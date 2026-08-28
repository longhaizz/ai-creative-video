"""One video in, one video out.

The whole job, in order:

    remove the burned-in subtitles   (optional, its own venv)
    split voice from music           (Demucs)
    read speech                      (Whisper, in-process)
    rewrite and translate            (OpenAI, three lengths a block)
    say every block                  (VoxCPM, best of a few takes)
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
from server.steps import audio, open_dubbing, separate, subtitle, transcribe, vsr
from server.steps.lipsync import NoFaceError, detect_scenes
from server.steps.synth import with_voice_instruction


class Models:
    """Everything that is loaded once and used by every job."""

    def __init__(self, voice, lipsync, whisper=None):
        self.voice = voice
        self.lipsync = lipsync
        self.whisper = whisper

    def as_list(self):
        models = [self.voice]
        if self.whisper is not None:
            models.append(self.whisper)
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

    # 2. Split voice from music, then read the mix with Whisper.
    mix = audio.extract_audio(video, work / "mix.wav")
    vocals, music = separate.separate(mix, work, ctx=ctx)
    ctx.check_cancel()
    cues, meta = transcribe.transcribe(
        models.whisper, mix, params.whisper_model, ctx=ctx
    )
    cues = open_dubbing.attach_refs(cues, vocals, work / "od")
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

    if models.whisper is None:
        ctx.log("No Whisper model: takes cannot be listened to, "
                "so the first one is kept whatever it says")

    def listen(wav, lang):
        """Hear a take back, to judge it and to time its sentences."""
        if models.whisper is None:
            return None
        return transcribe.listen(
            models.whisper, wav, lang, params.whisper_model)

    # Scene cuts are anchors: the dub is never allowed to drift across
    # one, because a cut is the moment a viewer checks lips against sound.
    scenes = detect_scenes(video)
    ctx.log(f"{len(scenes)} scene cuts")

    # Imported here so the module still loads without an OpenAI key present.
    from server.steps.synth import timed_speech

    speech = timed_speech(
        cues, work, video_seconds, speak,
        config.OPENAI_API_KEY, params.target_lang, meta=meta, ctx=ctx,
        listen=listen, scenes=scenes,
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

    timed_speech writes spoken_cues.json from the take it used, sentence by
    sentence. That is the only source of truth: the script can change after
    a rewrite, and the ASR window is the old speaker's timing, not ours.
    """
    import json

    spoken = json.loads(
        (work / "spoken_cues.json").read_text(encoding="utf-8"))
    return [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "text": (item.get("text") or "").strip(),
        }
        for item in spoken
        if (item.get("text") or "").strip()
    ]
