"""Make the new voice with VoxCPM, and fit it to the original timing.

Ported from spy-ads voxcpm_api.py and the _timed_speech / _fit_cue_audio
methods of voxcpm_panel.py.

The hard part is not making speech, it is making speech that lands where the
old speech was. Reading the whole transcript as one block removes every
pause between sentences, so the dub finishes early and the picture no longer
matches. So each cue is spoken on its own and laid back at its own start
time, and only the length inside a cue is adjusted.

Order of preference when a line does not fit its slot:
  1. leave it alone            (it is close enough)
  2. change the speed a little (up to +15% / -18%)
  3. ask OpenAI for a shorter or longer line, then try again
Never cut the audio: a cut loses words, and a word lost is worse than a
sentence that runs a little long.
"""

from __future__ import annotations

import json
from pathlib import Path

from server.jobs import PipelineError
from server.steps.audio import (
    SOFT_SLOWDOWN,
    SOFT_SPEEDUP,
    duration,
    match_tempo,
    place_clips,
)
from server.steps.translate import (
    CANNOT_FIT,
    reconstruct_script,
    rephrase_for_duration,
)

# How far the spoken length may sit from the slot before we act.
RATIO_KEEP_LO = 0.92   # close enough, leave it
RATIO_KEEP_HI = 1.08
RATIO_SOFT_LO = 0.85   # a bit short or long, change the speed
RATIO_SOFT_HI = 1.15   # past these, ask for a different line

# One rewrite only. A second one drifts away from the meaning without
# fitting any better.
MAX_REWRITES = 1

# Voice presets, put in front of the text for /tts. Not used when cloning.
VOICE_PRESETS = {
    "male_young": "A young man, warm, clear and energetic voice",
    "male_middle": "A middle-aged man, low-pitched, warm and authoritative voice",
    "male_old": "An elderly man, deep, slightly raspy and slow voice",
    "female_young": "A young woman, bright, gentle and sweet voice",
    "female_middle": "A middle-aged woman, warm, confident and natural voice",
    "female_old": "An elderly woman, soft, mature and slightly raspy voice",
}


def with_voice_instruction(text: str, preset: str) -> str:
    """Put the voice description in front of the line."""
    description = VOICE_PRESETS.get(preset)
    if not description:
        raise PipelineError(f"Unknown voice preset: {preset}", code="invalid_input")
    body = (text or "").strip()
    if not body:
        raise PipelineError("There is no text to speak", code="invalid_input")
    return f"({description}){body}"


class VoxCPMModel:
    """Keeps the loaded TTS model between jobs."""

    name = "voxcpm"

    def __init__(self, model_id: str = "openbmb/VoxCPM2"):
        self.model_id = model_id
        self._model = None

    def load(self) -> None:
        from voxcpm import VoxCPM

        # No device argument here: VoxCPM2 chooses the card itself. Use
        # CUDA_VISIBLE_DEVICES to pin it to one.
        self._model = VoxCPM.from_pretrained(
            self.model_id, load_denoiser=False
        )

    def speak(self, text: str, out_wav: Path, cfg_value: float,
              timesteps: int, reference_wav: Path | None = None,
              reference_text: str | None = None) -> Path:
        """Say one line. With reference_wav it copies that voice.

        reference_text is what the reference wav says. VoxCPM2 needs both
        or the clone comes out wrong.
        """
        if self._model is None:
            raise PipelineError("The voice model is not loaded")

        import soundfile

        wav = self._model.generate(
            text=text,
            prompt_wav_path=str(reference_wav) if reference_wav else None,
            prompt_text=reference_text if reference_wav else None,
            cfg_value=cfg_value,
            inference_timesteps=timesteps,
        )
        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(out_wav, wav, self._model.tts_model.sample_rate)
        return out_wav


def cue_slots(cues: list[dict], video_seconds: float) -> list[dict]:
    """Work out the time slot of every cue.

    `target` is how long the new speech should be. `window` is how far it
    may run before it walks into the next cue. Silent cues still hold their
    place on the timeline, so the one before them cannot borrow their time.
    """
    slots = []
    count = len(cues)
    for index, cue in enumerate(cues):
        start = float(cue.get("speech_start", cue["start"]))
        end = float(cue.get("speech_end", cue["end"]))
        speech = max(end - start, 0.4)
        if index + 1 < count:
            following = cues[index + 1]
            next_start = float(following.get("speech_start", following["start"]))
            window = max(next_start - start, 0.4)
        else:
            window = max(video_seconds - start, speech)
        slots.append({
            "start": start,
            "end": end,
            # Never chase a target that reaches into the next cue.
            "target": min(speech, window),
            "window": window,
        })
    return slots


def fit_cue(
    line: str,
    slot: dict,
    work: Path,
    index: int,
    speak,
    rewrite=None,
    ctx=None,
) -> Path:
    """Speak one line and make it fit its slot. Returns the wav path.

    `speak(text, path)` makes the audio. `rewrite(text, seconds, shorter)`
    asks for a different line, or is None to skip that step.
    """
    label = f"Cue {index + 1}"
    target = float(slot["target"])
    window = float(slot["window"])
    # Hard limit: never grow into the next cue.
    cap = min(target, max(window - 0.05, 0.4))
    fit_path = work / f"cue_{index:03d}_fit.wav"

    def log(message: str):
        if ctx is not None:
            ctx.log(f"{label}: {message}")

    raw = speak(line, work / f"cue_{index:03d}.wav")
    spoken = duration(raw)
    ratio = spoken / target if target > 0 else 1.0
    log(f"spoke {spoken:.2f}s for a {target:.2f}s slot (ratio {ratio:.2f})")

    candidates = [(raw, spoken)]

    def predicted(length: float) -> float:
        """How long this take would be after the speed change."""
        if length <= 0 or cap <= 0:
            return length
        if RATIO_KEEP_LO <= length / cap <= RATIO_KEEP_HI:
            return length
        tempo = min(max(length / cap, SOFT_SLOWDOWN), SOFT_SPEEDUP)
        return length / tempo

    def stretch(path: Path, length: float, why: str) -> Path:
        share = length / cap if cap > 0 else 1.0
        if RATIO_KEEP_LO <= share <= RATIO_KEEP_HI:
            out = path
        elif share > 1.0:
            out, tempo = match_tempo(path, cap, fit_path, slowest=1.0)
            log(f"{why}: sped up by {tempo:.3f}")
        else:
            out, tempo = match_tempo(path, cap, fit_path, fastest=1.0)
            log(f"{why}: slowed down by {tempo:.3f}")

        # Last check: if it still reaches into the next cue, speed it up a
        # little more than the soft limit. Overlapping speech is worse.
        length_now = duration(out)
        if length_now > window * 1.001 and length_now > 0.05:
            needed = length_now / max(window - 0.05, 0.4)
            if needed > 1.02:
                clamped, tempo = match_tempo(
                    out,
                    max(window - 0.05, 0.4),
                    work / f"cue_{index:03d}_fit2.wav",
                    slowest=1.0,
                    fastest=max(SOFT_SPEEDUP, min(needed, 1.25)),
                )
                if duration(clamped) < length_now:
                    log(f"pushed to {tempo:.3f} to stay off the next cue")
                    out = clamped
        return out

    def best(why: str) -> Path:
        def score(candidate):
            _path, length = candidate
            after = predicted(length)
            over = max(0.0, after - window)
            return (over > 0.02, abs(after - cap), over)

        path, length = min(candidates, key=score)
        out = stretch(path, length, why)
        final = duration(out)
        if final > window * 1.02:
            log(f"still longer than its {window:.2f}s window; not cutting it")
        if final < target * 0.92:
            log(f"still short: {final:.2f}s of {target:.2f}s, silence follows")
        return out

    # Close enough already.
    if RATIO_KEEP_LO <= ratio <= RATIO_KEEP_HI:
        return stretch(raw, spoken, "clamp") if spoken > window * 1.001 else raw

    # A bit off: the speed change alone is enough.
    if RATIO_SOFT_LO <= ratio <= RATIO_SOFT_HI:
        return stretch(raw, spoken, "small fix")

    # Far off, and nobody can rewrite it for us.
    if rewrite is None or MAX_REWRITES < 1:
        return best("clamp")

    shorter = ratio > RATIO_SOFT_HI
    log(f"{'too long' if shorter else 'too short'}; asking for another line")
    new_line = rewrite(line, target, shorter)
    if new_line == CANNOT_FIT:
        log("no shorter wording exists, changing the speed instead")
        return best("cannot fit")

    log(f"new line: {new_line}")
    second = speak(new_line, work / f"cue_{index:03d}_r0.wav")
    second_length = duration(second)
    second_ratio = second_length / target if target > 0 else 1.0
    log(f"spoke {second_length:.2f}s (ratio {second_ratio:.2f})")
    candidates.append((second, second_length))

    # Asking for something shorter can overshoot into far too short.
    overshoot = shorter and second_ratio < RATIO_SOFT_LO
    improved = (
        abs(predicted(second_length) - cap) < abs(predicted(spoken) - cap) - 1e-6
    )
    if overshoot:
        log(f"the rewrite went too far ({ratio:.2f} to {second_ratio:.2f})")
        return best("overshoot")
    if not improved:
        log(f"the rewrite fits no better ({ratio:.2f} to {second_ratio:.2f})")
        return best("no better")

    if RATIO_KEEP_LO <= second_ratio <= RATIO_KEEP_HI:
        if second_length > window * 1.001:
            return stretch(second, second_length, "clamp")
        return second
    if RATIO_SOFT_LO <= second_ratio <= RATIO_SOFT_HI:
        return stretch(second, second_length, "after rewrite")
    return best("final")


def timed_speech(
    cues: list[dict],
    work: Path,
    video_seconds: float,
    speak,
    openai_key: str,
    target_lang: str,
    meta: dict | None = None,
    ctx=None,
) -> Path:
    """Translate, speak every cue, and lay them on one track.

    Returns a wav as long as the video, with the speech at the same moments
    as in the original.
    """
    if not openai_key:
        raise PipelineError(
            "An OpenAI key is needed, even for the same language: it repairs "
            "the transcript and shares the lines out between cues",
            code="internal",
        )

    if ctx is not None:
        ctx.step("Rewriting the script")

    script = reconstruct_script(cues, target_lang, openai_key, asr_meta=meta or {})
    (work / "dub_script.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if ctx is not None:
        _report_script(script, cues, ctx)

    lines = list(script["cue_translations"])
    if len(lines) != len(cues):
        raise PipelineError(
            f"Got {len(lines)} lines for {len(cues)} cues", code="internal"
        )

    language_name = script.get("output_lang_name") or ""
    master = script["master_meaning"]
    slots = cue_slots(cues, video_seconds)
    spoken_indices = [i for i, line in enumerate(lines) if (line or "").strip()]
    if not spoken_indices:
        raise PipelineError("There is nothing to say", code="internal")

    if ctx is not None:
        ctx.step(f"Making the voice ({len(spoken_indices)} lines)")

    clips: list[tuple[float, Path]] = []
    for position, index in enumerate(spoken_indices):
        if ctx is not None:
            ctx.check_cancel()
        slot = slots[index]
        line = lines[index].strip()

        previous = lines[spoken_indices[position - 1]].strip() if position else ""
        following = (
            lines[spoken_indices[position + 1]].strip()
            if position + 1 < len(spoken_indices)
            else ""
        )

        def rewrite(text: str, seconds: float, shorter: bool) -> str:
            return rephrase_for_duration(
                text, seconds, openai_key,
                master_meaning=master,
                prev_text=previous,
                next_text=following,
                shorter=shorter,
                target_lang=target_lang,
                lang_name=language_name,
            )

        fitted = fit_cue(line, slot, work, index, speak, rewrite, ctx)
        clips.append((slot["start"], fitted))

    return place_clips(clips, video_seconds, work / "speech_timed.wav")


def _report_script(script: dict, cues: list[dict], ctx) -> None:
    """Tell the user what the rewrite step decided."""
    if script.get("language_repaired"):
        ctx.log(
            f"Some lines came back in the wrong language and were redone: "
            f"{script.get('language_repair_indices')}"
        )
    if script.get("cues_filled"):
        ctx.log(f"Filled cues the transcript left empty: {script.get('cues_filled_indices')}")
    if script.get("trailing_cta_filled"):
        ctx.log("Filled the silent ending with the closing line")
    for index in script.get("garbled_silent_indices") or []:
        heard = ""
        if 0 <= index < len(cues):
            heard = (cues[index].get("text") or "").replace("\n", " ").strip()
        ctx.log(f"Cue {index + 1} was not understood, left silent: {heard[:100]}")
    ctx.log(f"Meaning: {script['master_meaning']}")
    ctx.log(f"Script: {script['master_translation']}")
    if script.get("uncertain_spans"):
        ctx.log(f"Unsure about: {script['uncertain_spans']}")
