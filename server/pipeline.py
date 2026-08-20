"""One video in, one video out.

The whole job, in order:

    remove the burned-in subtitles   (optional, its own venv)
    split voice from music           (demucs)
    read the speech                  (whisper)
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

from pathlib import Path

from server import config
from server.jobs import JobContext, PipelineError
from server.steps import audio, separate, subtitle, transcribe, vsr
from server.steps.lipsync import NoFaceError
from server.steps.synth import with_voice_instruction


class Models:
    """Everything that is loaded once and used by every job."""

    def __init__(self, voice, whisper, lipsync):
        self.voice = voice
        self.whisper = whisper
        self.lipsync = lipsync

    def as_list(self):
        return [self.whisper, self.voice, self.lipsync]


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


def _joined_text(cues: list[dict]) -> str:
    """All the cue text as one line, for VoxCPM's prompt_text."""
    return " ".join(
        (cue.get("text") or "").strip()
        for cue in cues
        if (cue.get("text") or "").strip()
    )


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

    # 2. Split the voice from the music, so the music can be kept.
    track = audio.extract_audio(video, work / "in.wav")
    vocals, music = separate.separate(track, work, ctx=ctx)
    ctx.check_cancel()

    # The music stem still holds some of the old voice. It only shows in the
    # gaps, where it sounds like the old speaker never left.
    ctx.step("Cleaning the music track")
    music = audio.suppress_vocal_bleed(music, vocals, work / "music_clean.wav")
    ctx.check_cancel()

    # 3. Read the speech, then 4. rewrite it, then 5. say it.
    cues, meta = transcribe.transcribe(
        models.whisper, vocals, params.whisper_model, ctx=ctx
    )
    for index, cue in enumerate(cues, 1):
        reason = transcribe.cue_needs_review(cue)
        if reason:
            ctx.log(f"Cue {index} may be wrong ({reason}): {cue['text'][:80]}")
    ctx.check_cancel()

    uploaded_reference = _reference_audio(work)
    reference = uploaded_reference or vocals

    # VoxCPM2 clones a voice from the wav AND the words said in it, so the
    # reference always needs a transcript. For the vocals we already have
    # one; an uploaded sample has to be read first.
    if uploaded_reference is None:
        reference_text = _joined_text(cues)
    else:
        reference_cues, _ = transcribe.transcribe(
            models.whisper, uploaded_reference, params.whisper_model, ctx=ctx
        )
        reference_text = _joined_text(reference_cues)

    if params.voice_mode == "original":
        ctx.log(f"Copying the voice from {reference.name}")
    else:
        ctx.log(f"Using the {params.voice_mode} voice")

    def speak(text: str, out_wav: Path) -> Path:
        if params.voice_mode == "original":
            return models.voice.speak(
                text, out_wav, params.cfg_value, params.inference_timesteps,
                reference_wav=reference,
                reference_text=reference_text,
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
        ctx.step("Matching the mouth to the new voice")
        try:
            picture = models.lipsync.run(
                video.resolve(), speech.resolve(), (work / "lipsync.mp4").resolve(),
                steps=params.latentsync_steps,
                guidance=params.latentsync_guidance,
            )
        except NoFaceError as error:
            # Not a failure. Ad creatives often have no talking head.
            ctx.log(f"Skipping lip sync: {error}")
            picture = video
    ctx.check_cancel()

    # 7 and 8. Put the music back under the voice, then onto the picture.
    ctx.step("Mixing and putting it together")
    mixed = audio.mix_audio(speech, music, work / "final.wav", seconds=video_seconds)
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
    """The translated lines with the original timings.

    The text comes from dub_script.json, which the rewrite step wrote, so
    the subtitles say the same words the voice says.
    """
    import json

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
