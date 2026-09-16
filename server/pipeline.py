"""One video in, one video out.

The whole job, in order:

    read speech                      (Whisper, in-process)
    remove the burned-in subtitles   (optional, its own venv)
    split voice from music           (Demucs)
    rewrite and translate            (OpenAI, three lengths a block)
    say every block                  (VoxCPM, best of a few takes)
    move the mouth                   (LatentSync, optional)
    mix the new voice with the music
    put the sound back on the picture
    burn in the new subtitles        (optional)

With dub=false none of the voice work runs. The job keeps the original
sound and only redoes the subtitles: read the speech, take the old ones off,
put new ones on in the language already spoken. See _subtitle_only.

Order matters in three places. Speech is read first, because the subtitle
remover keeps only the text that was said. Subtitles come off next, so every
later step works on a clean picture. Lip sync happens before the music is mixed in,
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
from server.steps.synth import preset_voice


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
        kind = getattr(ctx.params, "job_kind", "dub")
        if kind == "clone":
            return _clone(ctx, models)
        if not getattr(ctx.params, "dub", True):
            return _subtitle_only(ctx, models)
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
    # Found before any GPU work: the request was checked, but the file can
    # still have gone since then.
    preset = (None if params.voice_mode == "original"
              else preset_voice(params.voice_mode))
    # Where the old subtitles sat, once step 1 has looked. Step 9 puts the
    # new ones there unless the client asked for a height of its own.
    detected_position = None

    # 0. Read the speech with Whisper, before the picture is touched. The
    # subtitle remover compares the text it finds with what was said, to
    # tell the subtitles from a logo or a price in the same band. The sound
    # is the same before and after the picture is cleaned, so every later
    # step reads the transcript it always did.
    #
    # Whisper reads the mix, not the vocals stem. Feeding it the stem was
    # tried, on the theory that music under the voice is what makes Whisper
    # write one short line for a whole window and skip the rest. It is the
    # wrong trade: most of these videos have no music at all, and demucs
    # still runs its four-stem split over the clean voice and rebuilds it,
    # artefacts and all. Clean speech came back transcribed as words that
    # do not exist in the language. Do not reshape the sound Whisper hears
    # to fix a fault that lives in how Whisper is called.
    mix = audio.extract_audio(video, work / "mix.wav")
    cues, meta = transcribe.transcribe(
        models.whisper, mix, params.whisper_model, ctx=ctx
    )
    ctx.check_cancel()

    # 1. Take the old subtitles off the picture. The same pass reads the
    # text that is staying, when the job asked for it to be translated.
    screen_path = _screen_text_path(work, params)
    if params.remove_subtitle:
        ctx.step("Removing the old subtitles")
        video, detected_position = vsr.remove_subtitles(
            video, work / "no_subs.mp4", params.vsr_mode,
            params.vsr_top, params.vsr_bottom, params.vsr_left, params.vsr_right,
            ctx=ctx,
            speech_cues=_speech_file(work, cues, meta, ctx),
            screen_text=screen_path,
        )
    elif screen_path is not None:
        _read_screen_only(video, work, params, cues, meta, screen_path, ctx)
    ctx.check_cancel()

    # 1b. Take the old hook off too, in the box the client drew. Its own
    # pass: the hook sits far from the subtitles, and one box around both
    # would send the remover over the middle of the picture as well.
    if params.hook_text:
        video = _remove_hook(video, work, params, ctx)
    ctx.check_cancel()

    video_seconds = audio.duration(video)
    width, height = audio.video_size(video)
    ctx.log(f"{video_seconds:.1f}s, {width}x{height}")

    # 2. Split voice from music. The mix is the one Whisper read in step 0.
    vocals, music = separate.separate(mix, work, ctx=ctx)
    ctx.check_cancel()
    # The file name travels with what Whisper heard, because both are what
    # the translator knows about the source. A title written by a person is
    # the one word in the job spelled the way it was meant.
    meta = {**meta, "source_title": params.source_title}
    cues = open_dubbing.attach_refs(
        cues, vocals, work / "od", one_speaker=params.speakers == 1)
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
    if preset is not None:
        ctx.log(f"Using the {params.voice_mode} voice")
        # The preset is what the user picked, so it wins over the upload.
        if uploaded is not None:
            ctx.log(f"Ignoring {uploaded.name}: voice_mode is "
                    f"{params.voice_mode}")
    elif uploaded is not None:
        ctx.log(f"Copying the voice from {uploaded.name}")
    else:
        ctx.log("Copying the voice from each spoken cue")

    def speak(text: str, out_wav: Path, cue: dict | None = None) -> Path:
        ref = preset or uploaded
        if ref is None and cue is not None and cue.get("ref_wav"):
            ref = Path(cue["ref_wav"])
        if ref is None:
            ref = vocals
        return models.voice.speak(
            text, out_wav, params.cfg_value, params.inference_timesteps,
            reference_wav=ref,
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
    if params.lipsync and models.lipsync is None:
        # Lip sync is off on this server (LOAD_LIPSYNC). The job goes on as
        # if the box was not ticked, instead of failing after minutes of work.
        ctx.log("Lip sync skipped: LatentSync is not loaded on this server")
    elif params.lipsync:
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

    # 9. Burn the new subtitles, the new hook and the translated screen text
    # on last, in one encode, so they sit on the final picture.
    hook = _hook(params, video_seconds)
    screen = _translated_screen_text(screen_path, params, meta, width, height, ctx)
    if params.burn_subtitle or hook is not None or screen:
        ctx.step("Burning in the subtitles")
        lines = _subtitle_cues(work, cues) if params.burn_subtitle else []
        result = subtitle.burn(
            result, lines, work / "result_subbed.mp4", width, height,
            font=params.subtitle_font,
            size=params.subtitle_size,
            position=_subtitle_position(params.subtitle_position, detected_position),
            hook=hook,
            screen=screen,
            ctx=ctx,
        )

    ctx.step("Done")
    return result


DEFAULT_SUBTITLE_POSITION = 0.75


def _hook(params, seconds: float) -> dict | None:
    """What subtitle.burn needs to draw the new hook, or None for no hook.

    It runs from the first frame to the last, because the hook it replaces
    was on screen the whole time.
    """
    if not params.hook_text:
        return None
    return {
        "text": params.hook_text,
        "top": params.hook_top,
        "bottom": params.hook_bottom,
        "left": params.hook_left,
        "right": params.hook_right,
        "font": params.hook_font,
        "size": params.hook_size,
        "colour": params.hook_colour,
        "align": params.hook_align,
        "prewrapped": params.hook_prewrapped,
        "colours": [c.strip() for c in params.hook_colours.split(",") if c.strip()],
        "end": seconds,
    }


def _remove_hook(video: Path, work: Path, params, ctx) -> Path:
    """Paint the old hook out of the box the client drew.

    The same remover as the subtitles, pointed at another box. Finding no
    text there is not a failure: it hands the video back untouched, and the
    new hook is written over a frame that was already clean.
    """
    ctx.step("Removing the old hook")
    cleaned, _ = vsr.remove_subtitles(
        video, work / "no_hook.mp4", params.vsr_mode,
        params.hook_top, params.hook_bottom,
        params.hook_left, params.hook_right,
        ctx=ctx,
    )
    return cleaned


def _subtitle_position(asked: float | None, detected: float | None) -> float:
    """Where to put the new subtitles, as a share of the frame height.

    What the client asked for wins, so the slider in the desktop tool still
    does what it says. With nothing asked, the new text goes where the old
    text was; with no old text found either, the lower third.
    """
    if asked is not None:
        return asked
    if detected is not None:
        return detected
    return DEFAULT_SUBTITLE_POSITION


def _subtitle_cues(work: Path, cues: list[dict]) -> list[dict]:
    """The lines that were actually spoken, timed to the dubbed audio.

    timed_speech writes spoken_cues.json from the take it used, sentence by
    sentence. That is the only source of truth: the script can change after
    a rewrite, and the ASR window is the old speaker's timing, not ours.
    """
    import json

    spoken = json.loads(
        (work / "spoken_cues.json").read_text(encoding="utf-8"))
    return _spoken_lines(spoken)


def _speech_file(work: Path, cues: list[dict], meta: dict, ctx) -> Path:
    """Write what Whisper heard for the subtitle remover, and log each line.

    The remover reads the text in each box it finds and keeps the line whose
    text was said. The language picks the model that can read that script.
    The file is written even when nothing was heard: the remover then knows
    the filter was asked for, and removes nothing instead of every text.
    """
    import json

    lines = _spoken_lines(cues)
    language = meta.get("language", "")
    for line in lines:
        ctx.log(f"Heard ({language}) {line['start']:.2f}-{line['end']:.2f}s: "
                f"{line['text']}")
    path = work / "speech_cues.json"
    path.write_text(
        json.dumps({"language": language, "cues": lines}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _spoken_lines(cues: list[dict]) -> list[dict]:
    """The cues that carry text, in the shape subtitle.burn wants."""
    return [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "text": (item.get("text") or "").strip(),
        }
        for item in cues
        if (item.get("text") or "").strip()
    ]


def _screen_text_path(work: Path, params) -> Path | None:
    """Where the remover writes the text it leaves on screen, or None.

    None means no translating: either nobody asked, or the target language
    is the one already on the picture, in which case writing the same words
    again would only put OCR mistakes on screen.
    """
    if not params.translate_screen_text:
        return None
    if (params.target_lang or "same").strip().lower() == "same":
        return None
    return work / "screen_text.json"


def _read_screen_only(video: Path, work: Path, params, cues, meta,
                      screen_path: Path, ctx) -> None:
    """Read the text on the picture without painting anything out.

    For a job that wants the on-screen text translated but the original
    subtitles left where they are. It runs no inpainting model, so it costs
    a pass of OCR and nothing else, and the video it was given is unchanged.
    """
    ctx.step("Reading the text on the picture")
    vsr.remove_subtitles(
        video, work / "detect_only.mp4", params.vsr_mode,
        params.vsr_top, params.vsr_bottom, params.vsr_left, params.vsr_right,
        ctx=ctx,
        speech_cues=_speech_file(work, cues, meta, ctx),
        screen_text=screen_path,
        detect_only=True,
    )


def _translated_screen_text(screen_path: Path | None, params, meta,
                            width: int, height: int, ctx) -> list[dict]:
    """The text found on the picture, translated, ready for subtitle.burn.

    A failure here loses the translation, not the job: everything before it
    took minutes on a GPU, and a video that comes back with its original
    on-screen text is worth more than no video at all.
    """
    if screen_path is None:
        return []
    pieces = vsr.read_screen_text(screen_path)
    kept = [piece for piece in pieces
            if not _inside_hook(piece, params, width, height)]
    if len(kept) < len(pieces):
        ctx.log(f"Leaving {len(pieces) - len(kept)} pieces of text in the hook "
                f"box alone: the hook says what goes there")
    if not kept:
        ctx.log("No text on the picture to translate")
        return []

    ctx.step("Translating the text on the picture")
    # Imported here, so reading this file does not need an OpenAI key.
    from server.steps.translate import translate_labels

    try:
        said = translate_labels(
            [piece["text"] for piece in kept], params.target_lang,
            config.OPENAI_API_KEY, asr_meta=meta, ctx=ctx,
        )
    except PipelineError as error:
        ctx.log(f"Could not translate the text on the picture: {error}")
        return []
    for piece, text in zip(kept, said):
        ctx.log(f"Screen text: {piece['text']!r} -> {text!r}")
        piece["text"] = text
    return kept


def _inside_hook(piece: dict, params, width: int, height: int) -> bool:
    """Does this piece of text touch the box the client drew for the hook?

    The client typed what that box is to say, so an automatic translation
    has no business drawing over it. Without this the old hook, read off
    the frame, would be translated and written on top of the new one.
    """
    if not params.hook_text or params.hook_left is None:
        return False
    xmin, xmax, ymin, ymax = piece["box"]
    return (xmin <= width * params.hook_right
            and xmax >= width * params.hook_left
            and ymin <= height * params.hook_bottom
            and ymax >= height * params.hook_top)


def _subtitle_only(ctx: JobContext, models: Models) -> Path:
    """No new voice: keep the original sound, only redo the subtitles.

    Many ad creatives need nothing said again. The old text comes off the
    picture and new text goes on, in the language that is already spoken.
    So Demucs, the translation and VoxCPM are all skipped, and the sound is
    never touched: the file that comes back still carries the audio it
    arrived with, sample for sample.
    """
    params = ctx.params
    work = ctx.workdir
    video = _source_video(work)
    # Where the old subtitles sat, if anything looked. Same rule as the dub
    # path: the new text goes back where the old text was.
    detected_position = None

    if params.burn_subtitle and models.whisper is None:
        raise PipelineError(
            "New subtitles were asked for, but this server has no "
            "Whisper model to read the speech with.",
            code="invalid_input",
        )

    # Read the speech before the picture is touched. The new lines come from
    # it, and the subtitle remover uses it to tell the subtitles from other
    # text. A job that only removes subtitles still pays for this read; with
    # no Whisper loaded it removes them by position alone, as it always did.
    cues, meta = [], {}
    # Screen text needs Whisper too: the language it hears is what picks the
    # model that can read the letters on the picture.
    wants_reading = (params.burn_subtitle or params.remove_subtitle
                     or params.translate_screen_text)
    if models.whisper is not None and wants_reading:
        cues, meta = transcribe.transcribe(
            models.whisper,
            audio.extract_audio(video, work / "mix.wav"),
            params.whisper_model,
            ctx=ctx,
        )
        ctx.check_cancel()

    screen_path = _screen_text_path(work, params)
    if params.remove_subtitle:
        ctx.step("Removing the old subtitles")
        video, detected_position = vsr.remove_subtitles(
            video, work / "no_subs.mp4", params.vsr_mode,
            params.vsr_top, params.vsr_bottom, params.vsr_left, params.vsr_right,
            ctx=ctx,
            speech_cues=(_speech_file(work, cues, meta, ctx)
                         if models.whisper is not None else None),
            screen_text=screen_path,
        )
    elif screen_path is not None:
        _read_screen_only(video, work, params, cues, meta, screen_path, ctx)
    ctx.check_cancel()

    if params.hook_text:
        video = _remove_hook(video, work, params, ctx)
    ctx.check_cancel()

    lines: list[dict] = []
    if params.burn_subtitle:
        lines = _spoken_lines(cues)
        if not lines:
            # Creatives carrying only music are common, and nothing here
            # could have known in advance. Removing the subtitles is the
            # expensive part and it already worked, so keep that and say why
            # the rest did not happen.
            ctx.log("Nobody speaks in this video, so there is nothing to burn")

    # The length is only read when there is a hook to hold for it.
    hook = _hook(params, audio.duration(video)) if params.hook_text else None
    width, height = audio.video_size(video)
    screen = _translated_screen_text(screen_path, params, meta, width, height, ctx)
    if lines or hook is not None or screen:
        ctx.step("Burning in the subtitles")
        video = subtitle.burn(
            video, lines, work / "result_subbed.mp4", width, height,
            font=params.subtitle_font,
            size=params.subtitle_size,
            position=_subtitle_position(
                params.subtitle_position, detected_position),
            hook=hook,
            screen=screen,
            ctx=ctx,
        )

    ctx.step("Done")
    return video


def _clone(ctx: JobContext, models: Models) -> Path:
    """Audio-only voice cloning: reference_audio + text -> wav."""
    params = ctx.params
    work = ctx.workdir

    reference_media = _reference_audio(work)
    if reference_media is None:
        raise PipelineError(
            "reference_audio was not saved",
            code="invalid_input",
        )

    ctx.step("Cloning voice and speaking text")

    # VoxCPM's `reference_wav_path` expects a WAV (or at least something
    # decodable as audio). Normalize to PCM WAV first so inputs like
    # mp3/mp4 are safe.
    # Not "reference_audio.wav": the upload may already carry that name,
    # and ffmpeg refuses when input and output are the same file.
    reference_wav = work / "reference_pcm.wav"
    audio.extract_audio(reference_media, reference_wav)

    out_wav = work / "result.wav"
    return models.voice.speak(
        params.text,
        out_wav,
        params.cfg_value,
        params.inference_timesteps,
        reference_wav=reference_wav,
    )
