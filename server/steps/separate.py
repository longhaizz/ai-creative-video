"""Split the sound into voice and music with demucs.

The dub replaces the voice, so the music of the original video has to
survive on its own track. Demucs is what pulls them apart.

It runs on the GPU here. The desktop tool forced -d cpu because a laptop
had nothing else; on the server this is the cheapest step on the card.
"""

from __future__ import annotations

from pathlib import Path

from server.jobs import PipelineError


def separate(audio: Path, work_dir: Path, ctx=None) -> tuple[Path, Path]:
    """Return (vocals, music). Both are wav files inside work_dir."""
    # Imported here, not at the top: the queue tests must keep running on a
    # machine with no torch and no demucs.
    from demucs.api import Separator, save_audio

    audio = Path(audio)
    out_dir = Path(work_dir) / "stems"
    out_dir.mkdir(parents=True, exist_ok=True)

    if ctx is not None:
        ctx.step("Splitting voice from music")

    separator = Separator(model="htdemucs", device="cuda")
    _origin, stems = separator.separate_audio_file(str(audio))
    if "vocals" not in stems:
        raise PipelineError("demucs returned no vocals stem")

    # htdemucs gives four stems. Everything that is not the voice is the
    # backing track, so add the other three together.
    music = sum(source for name, source in stems.items() if name != "vocals")

    vocals_path = out_dir / "vocals.wav"
    music_path = out_dir / "no_vocals.wav"
    save_audio(stems["vocals"], str(vocals_path), samplerate=separator.samplerate)
    save_audio(music, str(music_path), samplerate=separator.samplerate)
    return vocals_path, music_path
