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
  4. speed up harder (up to 1.4x) so it stays off the next cue
  5. cut the tail to the window — overlapping speech is worse than a lost word
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
    trim_audio,
)
from server.steps.translate import (
    CANNOT_FIT,
    pace_counts,
    reconstruct_script,
    rephrase_for_duration,
    score_pace,
    word_count,
)

# How far the spoken length may sit from the slot before we act.
RATIO_KEEP_LO = 0.92   # close enough, leave it
RATIO_KEEP_HI = 1.08
RATIO_SOFT_LO = 0.85   # a bit short or long, change the speed
RATIO_SOFT_HI = 1.15   # past these, ask for a different line
# A take this short vs the slot is B-roll, not a line that needs padding.
RATIO_HOLE = 0.35

# One rewrite only. A second one drifts away from the meaning without
# fitting any better.
MAX_REWRITES = 1

# Last speed-up before we cut. Faster than this sounds wrong; slower leaves
# the next cue overlapping.
HARD_SPEEDUP = 1.4

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

    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "cuda:0"):
        self.model_id = model_id
        self.device = device
        self._model = None

    def load(self) -> None:
        from voxcpm import VoxCPM

        self._model = VoxCPM.from_pretrained(
            self.model_id, device=self.device, load_denoiser=False
        )

    def speak(self, text: str, out_wav: Path, cfg_value: float,
              timesteps: int, reference_wav: Path | None = None) -> Path:
        """Say one line. With reference_wav it copies that voice.

        reference_wav_path is the cloning mode, the one the desktop tool
        calls through /clone, and it needs no transcript of the reference.
        Do not swap it for prompt_wav_path: that is continuation mode, it
        wants the exact words of the reference in prompt_text, and it
        sounds worse here. Both arguments need voxcpm 2.x — on 1.x this
        call raises TypeError, which is the real fault to fix.
        """
        if self._model is None:
            raise PipelineError("The voice model is not loaded")

        import soundfile

        wav = self._model.generate(
            text=text,
            reference_wav_path=str(reference_wav) if reference_wav else None,
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
    stats=None,
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
    if stats is not None:
        if ratio > RATIO_SOFT_HI:
            heard = "too_fast"
        elif ratio < RATIO_SOFT_LO:
            heard = "too_slow"
        else:
            heard = "ok"
        stats.update({
            "spoken": round(spoken, 3),
            "target": round(target, 3),
            "ratio": round(ratio, 3),
            "heard": heard,
        })

    candidates = [(raw, spoken, line)]

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

        # Last check: if it still reaches into the next cue, speed it up
        # harder, then cut the tail. Overlapping speech is worse.
        limit = max(window - 0.05, 0.4)
        length_now = duration(out)
        if length_now > window * 1.001 and length_now > 0.05:
            needed = length_now / limit
            if needed > 1.02:
                clamped, tempo = match_tempo(
                    out, limit, work / f"cue_{index:03d}_fit2.wav",
                    slowest=1.0,
                    fastest=max(SOFT_SPEEDUP, min(needed, HARD_SPEEDUP)),
                )
                if duration(clamped) < length_now:
                    log(f"pushed to {tempo:.3f} to stay off the next cue")
                    out = clamped
            length_now = duration(out)
            if length_now > window * 1.001:
                out = trim_audio(
                    out, limit, work / f"cue_{index:03d}_trim.wav",
                )
                log(f"cut to {duration(out):.2f}s to stay off the next cue")
        return out

    def best(why: str) -> Path:
        def score(candidate):
            _path, length, _text = candidate
            over = max(0.0, length - window)
            return (over > 0.02, abs(length - cap), over)

        path, length, text = min(candidates, key=score)
        if stats is not None:
            stats["spoken_text"] = text
        out = stretch(path, length, why)
        final = duration(out)
        if final < target * 0.92:
            log(f"still short: {final:.2f}s of {target:.2f}s, silence follows")
        return out

    def finish(path: Path, text: str) -> Path:
        if stats is not None:
            stats.setdefault("spoken_text", text)
        return path

    # Close enough already.
    if RATIO_KEEP_LO <= ratio <= RATIO_KEEP_HI:
        out = stretch(raw, spoken, "clamp") if spoken > window * 1.001 else raw
        return finish(out, line)

    # A bit off: the speed change alone is enough.
    if RATIO_SOFT_LO <= ratio <= RATIO_SOFT_HI:
        return finish(stretch(raw, spoken, "small fix"), line)

    # Far off, and nobody can rewrite it for us.
    # Tiny slots: the LLM replaces "Water" with a stolen full sentence.
    # A huge hole with almost no words is B-roll: stretch, leave silence.
    # A real sentence that TTS rushed (ratio 0.34, 7+ words) still needs
    # a longer line — that is a talking-head, not B-roll.
    thin = word_count(line) <= 4
    if rewrite is None or MAX_REWRITES < 1 or target < 0.8 or (
            ratio < RATIO_HOLE and thin):
        if target < 0.8 and rewrite is not None and MAX_REWRITES >= 1:
            log("slot too short to rewrite, changing the speed instead")
        elif ratio < RATIO_HOLE and thin:
            log("slot much longer than the take, leaving silence after")
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

    closer = abs(second_length - cap) < abs(spoken - cap) - 1e-6
    if not closer:
        log(f"the rewrite fits no better ({ratio:.2f} to {second_ratio:.2f})")
        return best("clamp")

    candidates.append((second, second_length, new_line))

    if RATIO_KEEP_LO <= second_ratio <= RATIO_KEEP_HI:
        if second_length > window * 1.001:
            return finish(stretch(second, second_length, "clamp"), new_line)
        return finish(second, new_line)
    if RATIO_SOFT_LO <= second_ratio <= RATIO_SOFT_HI:
        return finish(stretch(second, second_length, "after rewrite"), new_line)
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
    lines = list(script["cue_translations"])
    if len(lines) != len(cues):
        raise PipelineError(
            f"Got {len(lines)} lines for {len(cues)} cues", code="internal"
        )

    language_name = script.get("output_lang_name") or ""
    master = script["master_meaning"]
    slots = cue_slots(cues, video_seconds)
    # Word-count "too fast" is not VoxCPM's real duration. Do not rewrite
    # here; fit_cue measures the take and rewrites if the audio is off.
    scores = [
        score_pace(line, float(slots[i]["target"]))
        for i, line in enumerate(lines)
    ]
    script["cue_translations"] = lines
    script["pace"] = scores
    (work / "dub_script.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if ctx is not None:
        _report_script(script, cues, ctx)
        _report_pace(scores, ctx, "script")

    spoken_indices = [i for i, line in enumerate(lines) if (line or "").strip()]
    if not spoken_indices:
        raise PipelineError("There is nothing to say", code="internal")

    if ctx is not None:
        ctx.step(f"Making the voice ({len(spoken_indices)} lines)")

    clips: list[tuple[float, Path]] = []
    spoken_cues: list[dict] = []
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

        cue = cues[index]

        def speak_this(text, path, _cue=cue):
            return speak(text, path, _cue)

        heard = {}
        fitted = fit_cue(line, slot, work, index, speak_this, rewrite, ctx, heard)
        if index < len(scores):
            scores[index]["heard"] = heard.get("heard")
            scores[index]["spoken_s"] = heard.get("spoken")
            scores[index]["spoken_ratio"] = heard.get("ratio")
        clips.append((slot["start"], fitted))
        spoken_end = slot["start"] + duration(fitted)
        spoken_cues.append({
            "start": round(slot["start"], 3),
            "end": round(max(spoken_end, slot["start"] + 0.05), 3),
            "text": heard.get("spoken_text") or line,
        })

    for index, sub in enumerate(spoken_cues[:-1]):
        nxt = spoken_cues[index + 1]["start"]
        if sub["end"] > nxt:
            sub["end"] = nxt
    (work / "spoken_cues.json").write_text(
        json.dumps(spoken_cues, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    script["pace"] = scores
    (work / "dub_script.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if ctx is not None:
        _report_pace(scores, ctx, "heard")
    return place_clips(clips, video_seconds, work / "speech_timed.wav")


def _report_script(script: dict, cues: list[dict], ctx) -> None:
    """Tell the user what the rewrite step decided."""
    if script.get("language_repaired"):
        ctx.log(
            f"Some lines came back in the wrong language and were redone: "
            f"{script.get('language_repair_indices')}"
        )
    if script.get("cue_count_aligned") or script.get("cue_count_repaired"):
        ctx.log(
            "The rewrite dropped some leftover Whisper fragments; "
            "those slots were left silent"
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


def _report_pace(scores: list, ctx, when: str) -> None:
    """Log how many lines would sound rushed or dragged."""
    key = "heard" if when == "heard" else "verdict"
    counts = pace_counts(scores, key)
    label = "Heard after TTS" if when == "heard" else "Word-count estimate"
    ctx.log(
        f"{label}: {counts['ok']} ok, {counts['too_fast']} too fast, "
        f"{counts['too_slow']} too slow"
    )
    for index, score in enumerate(scores):
        verdict = score.get(key) or ""
        if verdict not in ("too_fast", "too_slow"):
            continue
        if when == "heard":
            ctx.log(
                f"Cue {index + 1} {verdict}: "
                f"spoke {score.get('spoken_s')}s "
                f"for a {score.get('slot_s')}s slot "
                f"(ratio {score.get('spoken_ratio')})"
            )
            continue
        after = score.get("after") or {}
        note = "rewrote" if score.get("rewritten") else "kept"
        ctx.log(
            f"Cue {index + 1} {verdict}: "
            f"{score['words']} words for {score['slot_s']}s "
            f"(~{score['estimated_s']}s spoken), {note}"
            + (
                f" → {after.get('verdict')}"
                if after.get("verdict") else ""
            )
        )
