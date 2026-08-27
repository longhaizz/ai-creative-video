"""One video in, one video out.

The whole job, in order:

    remove the burned-in subtitles   (optional, its own venv)
    split, diarize, read speech      (Open Dubbing venv)
    rewrite and translate            (OpenAI)
    say every line                   (VoxCPM, one cue at a time)
    move the mouth                   (LatentSync, optional)
    mix the new voice with the music
    put the sound back on the picture
    burn in the new subtitles        (optional)

Order matters in two places. Subtitles come off first, so every later step
works on a clean picture. Lip sync happens before the music is mixed in,
because it reads the sound to drive the mouth and music in that track would
confuse it.
"""

from __future__ import annotations

import time
from pathlib import Path

from server import config
from server.jobs import JobContext, PipelineError
from server.steps import audio, open_dubbing, subtitle, transcribe, vsr
from server.steps.lipsync import NoFaceError
from server.steps.synth import with_voice_instruction


class Models:
    """Everything that is loaded once and used by every job."""

    def __init__(self, voice, lipsync):
        self.voice = voice
        self.lipsync = lipsync

    def as_list(self):
        models = [self.voice]
        if self.lipsync is not None:
            models.append(self.lipsync)
        return models


def make_run_dub(models: Models):
    """Build the run_dub the JobRunner calls, holding the loaded models."""

    def run_dub(ctx: JobContext) -> Path:
        return _dub(ctx, models)

    return run_dub


def _source_video(work: Path) -> Path:
    found = sorted(work.glob("video.*"))
    if not found:
        raise PipelineError("The video was not saved", code="invalid_input")
    return found[0]


def _reference_audio(work: Path) -> Path | None:
    found = sorted(work.glob("reference_audio.*"))
    return found[0] if found else None


def _dub(ctx: JobContext, models: Models) -> Path:
    params = ctx.params
    work = ctx.workdir
    video = _source_video(work)

    # 1. Take the old subtitles off the picture.
    if params.remove_subtitle:
        ctx.step("Removing the old subtitles")
        video = vsr.remove_subtitles(
            video, work / "no_subs.mp4", params.vsr_mode,
            params.vsr_top, params.vsr_bottom, params.vsr_left, params.vsr_right,
            ctx=ctx,
        )
    ctx.check_cancel()

    video_seconds = audio.duration(video)
    width, height = audio.video_size(video)
    ctx.log(f"{video_seconds:.1f}s, {width}x{height}")

    # 2. Split, diarize and read speech in the Open Dubbing venv.
    segmented = open_dubbing.segment(
        video, work, params.whisper_model, ctx=ctx
    )
    cues = segmented["cues"]
    meta = segmented["meta"]
    vocals = segmented["vocals"]
    music = segmented["music"]
    ctx.check_cancel()

    ctx.step("Cleaning the music track")
    music = audio.suppress_vocal_bleed(music, vocals, work / "music_clean.wav")
    ctx.check_cancel()

    for index, cue in enumerate(cues, 1):
        reason = transcribe.cue_needs_review(cue)
        if reason:
            ctx.log(f"Cue {index} may be wrong ({reason}): {cue['text'][:80]}")
    ctx.check_cancel()

    uploaded = _reference_audio(work)
    if params.voice_mode == "original":
        if uploaded is not None:
            ctx.log(f"Copying the voice from {uploaded.name}")
        else:
            ctx.log("Copying the voice from each spoken cue")
    else:
        ctx.log(f"Using the {params.voice_mode} voice")

    def speak(text: str, out_wav: Path, cue: dict | None = None) -> Path:
        if params.voice_mode == "original":
            ref = uploaded
            if ref is None and cue is not None and cue.get("ref_wav"):
                ref = Path(cue["ref_wav"])
            if ref is None:
                ref = vocals
            return models.voice.speak(
                text, out_wav, params.cfg_value, params.inference_timesteps,
                reference_wav=ref,
            )
        return models.voice.speak(
            with_voice_instruction(text, params.voice_mode),
            out_wav, params.cfg_value, params.inference_timesteps,
        )

    # Imported here so the module still loads without an OpenAI key present.
    from server.steps.synth import timed_speech

    speech = timed_speech(
        cues, work, video_seconds, speak,
        config.OPENAI_API_KEY, params.target_lang, meta=meta, ctx=ctx,
    )
    ctx.log(f"Voice track: {audio.duration(speech):.1f}s of {video_seconds:.1f}s")
    ctx.check_cancel()

    # 6. Move the mouth. This reads `speech`, which is voice only: music in
    # that track would drive the mouth wrong, so the mix comes after.
    #
    # `speech` must be exactly as long as the video. LatentSync decides how
    # many frames to write from the length of the sound it is given, so a
    # short track silently cuts the end off the picture. place_clips() pads
    # to video_seconds for this reason; do not remove that.
    picture = video
    if params.lipsync:
        if models.lipsync is None:
            raise PipelineError(
                "Lip sync was requested but LatentSync is not loaded. "
                "Start with LOAD_LIPSYNC=1, or send lipsync=false.",
                code="invalid_input",
            )
        ctx.step("Matching the mouth to the new voice")
        # Log the numbers the model really got, and how long they cost. Both
        # come from the request, so a job that looks slow can be told apart
        # from a job that ignored its settings.
        ctx.log(
            f"LatentSync: steps={params.latentsync_steps} "
            f"guidance={params.latentsync_guidance}"
        )
        started = time.perf_counter()
        try:
            picture = models.lipsync.run_shots(
                video.resolve(),
                speech.resolve(),
                (work / "lipsync.mp4").resolve(),
                work / "shots",
                params.latentsync_steps,
                params.latentsync_guidance,
                ctx=ctx,
            )
            ctx.log(f"LatentSync took {time.perf_counter() - started:.1f}s")
        except NoFaceError as error:
            # Not a failure. Ad creatives often have no talking head.
            ctx.log(f"Skipping lip sync: {error}")
            picture = video
    ctx.check_cancel()

    # 7 and 8. Put the music back under the voice, then onto the picture.
    ctx.step("Mixing and putting it together")
    mixed = audio.mix_audio(speech, music, work / "final.wav", seconds=video_seconds)
    mixed = audio.make_audible(mixed, work / "final_loud.wav")
    ctx.log("Normalized the mix so the output is clearly audible")
    result = audio.mux_audio(picture, mixed, work / "result.mp4")

    # 9. Burn the new subtitles on last, so they sit on the final picture.
    if params.burn_subtitle:
        ctx.step("Burning in the subtitles")
        lines = _subtitle_cues(work, cues)
        result = subtitle.burn(
            result, lines, work / "result_subbed.mp4", width, height,
            font=params.subtitle_font,
            size=params.subtitle_size,
            position=params.subtitle_position,
            ctx=ctx,
        )

    ctx.step("Done")
    return result


def _subtitle_cues(work: Path, cues: list[dict]) -> list[dict]:
    """The lines that were actually spoken, timed to the dubbed audio.

    timed_speech writes spoken_cues.json after fit_cue. That is the source
    of truth: the script in dub_script.json can differ after a rewrite, and
    ASR windows are often longer than the new take.
    """
    import json

    spoken_path = work / "spoken_cues.json"
    if spoken_path.is_file():
        spoken = json.loads(spoken_path.read_text(encoding="utf-8"))
        return [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": (item.get("text") or "").strip(),
            }
            for item in spoken
            if (item.get("text") or "").strip()
        ]

    script = json.loads((work / "dub_script.json").read_text(encoding="utf-8"))
    lines = script.get("cue_translations") or []
    out = []
    for index, cue in enumerate(cues):
        text = (lines[index] if index < len(lines) else "").strip()
        if not text:
            continue
        out.append({
            "start": float(cue.get("speech_start", cue["start"])),
            "end": float(cue.get("speech_end", cue["end"])),
            "text": text,
        })
    return out
