"""Turn the voice samples in server/voices into the preset voices.

Each sample becomes server/voices/wav/<id>.wav: silence cut from both ends,
level evened out, at most MAX_SECONDS long. The id comes from the file name,
so "voice_preview_luke c.mp3" becomes luke_c. The wavs are made again on
every run; change the sample, not the wav.

Run from the repo root:

    python -m server.scripts.prepare_voices            # samples are clean
    python -m server.scripts.prepare_voices --demucs   # music under the voice
"""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from server import config
from server.steps.audio import clean_take, duration, trim_audio

SOURCES = Path(__file__).resolve().parents[1] / "voices"
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
# Shorter than this still clones, just less reliably; listen to it.
MIN_SECONDS = 5.0
# A longer reference makes every take slower and copies the voice no better.
MAX_SECONDS = 20.0


def voice_id(name: str) -> str:
    """The id a sample file becomes: lower case letters, digits and _."""
    stem = Path(name).stem.lower().removeprefix("voice_preview_")
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")


def prepare(source: Path, out_wav: Path, work: Path, demucs: bool) -> float:
    """Make one preset wav. Returns its length in seconds."""
    if demucs:
        from server.steps.separate import separate

        source, _music = separate(source, work)
    clean_take(source, out_wav)
    if duration(out_wav) > MAX_SECONDS:
        trim_audio(out_wav, MAX_SECONDS, out_wav)
    return duration(out_wav)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demucs", action="store_true",
                        help="take the voice out of music first (needs a GPU)")
    args = parser.parse_args()

    samples = sorted(p for p in SOURCES.iterdir()
                     if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)
    ids: dict[str, Path] = {}
    for sample in samples:
        vid = voice_id(sample.name)
        if not vid or vid == "original":
            raise SystemExit(f"{sample.name}: cannot be turned into a voice id")
        if vid in ids:
            raise SystemExit(f"{sample.name} and {ids[vid].name} both become {vid}")
        ids[vid] = sample

    config.VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for vid, sample in ids.items():
        with tempfile.TemporaryDirectory() as tmp:
            seconds = prepare(sample, config.VOICES_DIR / f"{vid}.wav",
                              Path(tmp), args.demucs)
        note = "  WARNING: short references clone less reliably" \
            if seconds < MIN_SECONDS else ""
        print(f"{vid}: {seconds:.1f}s{note}")
    print(f"{len(ids)} voices in {config.VOICES_DIR}")


if __name__ == "__main__":
    main()
