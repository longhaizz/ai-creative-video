"""ffmpeg jobs: read, cut, stretch, mix and mux audio.

Ported from spy-ads elevenlabs_api.py. The Windows-only parts are gone (no
hidden console window, no frozen-exe branch), and every error raises a
PipelineError so the job carries a code back to the client.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from server import config
from server.jobs import PipelineError

# Social/mobile ads: clearly audible without clipping.
OUTPUT_LUFS = -14.0


def run_ffmpeg(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(f"ffmpeg failed: {(result.stderr or '')[-300:]}")
    return result.stdout


def duration(path) -> float:
    """Length in seconds."""
    out = run_ffmpeg([
        config.FFPROBE_BIN, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ])
    try:
        return float(out.strip())
    except ValueError:
        raise PipelineError(
            f"Could not read the length of {Path(path).name}",
            code="invalid_input",
        )


def video_size(path) -> tuple[int, int]:
    out = run_ffmpeg([
        config.FFPROBE_BIN, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x", str(path),
    ])
    try:
        width, height = out.strip().split("x")[:2]
        return int(width), int(height)
    except ValueError:
        raise PipelineError("The file has no video stream", code="invalid_input")


def extract_audio(video, out_wav) -> Path:
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(video), "-vn", "-c:a", "pcm_s16le", str(out_wav),
    ])
    return out_wav


def suppress_vocal_bleed(music, vocals, out_wav) -> Path:
    """Push down the original voice that leaked into the music stem.

    Demucs usually leaves some voice in the accompaniment. While the new
    dub is silent, between sentences or at the end of the clip, that leak
    becomes audible and sounds like the old speaker is still there. Duck the
    music using the vocals stem as the side chain.
    """
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(music), "-i", str(vocals),
        "-filter_complex",
        "[0:a][1:a]sidechaincompress=threshold=0.02:ratio=8:attack=20:"
        "release=250:makeup=1:knee=2.5:link=average[a]",
        "-map", "[a]", "-c:a", "pcm_s16le", str(out_wav),
    ])
    return out_wav


def mix_audio(voice, music, out_wav, music_gain: float = 1.0, seconds=None) -> Path:
    """Put the new voice over the original music.

    normalize=0 stops amix from halving both inputs.

    `seconds` trims and pads both sides to the same length first, then mixes
    with duration=first. Without it, duration=longest would drag in the tail
    of the music stem, where the old voice leaks through.
    """
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if seconds and seconds > 0:
        span = f"{float(seconds):.3f}"
        run_ffmpeg([
            config.FFMPEG_BIN, "-y", "-loglevel", "error",
            "-i", str(voice), "-i", str(music),
            "-filter_complex",
            f"[0:a]apad,atrim=0:{span},asetpts=PTS-STARTPTS[v];"
            f"[1:a]volume={music_gain},apad,atrim=0:{span},"
            f"asetpts=PTS-STARTPTS[m];"
            f"[v][m]amix=inputs=2:duration=first:normalize=0[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(out_wav),
        ])
    else:
        run_ffmpeg([
            config.FFMPEG_BIN, "-y", "-loglevel", "error",
            "-i", str(voice), "-i", str(music),
            "-filter_complex",
            f"[1:a]volume={music_gain}[m];"
            f"[0:a][m]amix=inputs=2:duration=first:normalize=0[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(out_wav),
        ])
    return out_wav


def make_audible(src, dest) -> Path:
    """Raise the mix to a loud, even level so the output is easy to hear.

    EBU R128 at -14 LUFS matches phone / social-video playback. Always run:
    a quiet clone or a quiet music bed would otherwise leave the mp4 thin.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(src),
        "-af", f"loudnorm=I={OUTPUT_LUFS}:TP=-1.5:LRA=11",
        "-c:a", "pcm_s16le", str(dest),
    ])
    return dest


# Every take is brought to this level before it is placed, so one block is
# never louder than the next. The final mix is normalised again later.
TAKE_LUFS = -18.0
# Fade at both ends of a take. Long enough to kill the click of a hard cut,
# short enough that no one hears a fade.
EDGE_FADE = 0.02


def clean_take(src, dest) -> Path:
    """Make one take ready to sit next to another one.

    Three things, in one pass. The silence the model leaves before and after
    the words is removed, so the block starts speaking on the beat it was
    given. The level is evened out. Both edges get a very short fade, so
    joining two takes makes no click.

    The fades are done with areverse rather than afade=t=out, because
    fade-out needs the length of the audio and that is only known after the
    silence has been trimmed.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    trim = ("silenceremove=start_periods=1:start_silence=0.05:"
            "start_threshold=-45dB:detection=peak")
    fade = f"afade=t=in:st=0:d={EDGE_FADE}"
    run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(src),
        "-af", ",".join([
            trim,           # silence before the first word
            "areverse",
            trim,           # silence after the last word
            fade,           # which is the fade-out, we are reversed
            "areverse",
            fade,           # the fade-in
            f"loudnorm=I={TAKE_LUFS}:TP=-2:LRA=11",
            # loudnorm hands back 192kHz. Come straight back to the rate the
            # mix runs at, so nothing downstream carries four times the data.
            "aresample=44100",
        ]),
        "-c:a", "pcm_s16le", str(dest),
    ])
    return dest


def mux_audio(video, audio, out_mp4) -> Path:
    """Replace the sound track: picture from input 0, sound from input 1."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-shortest", str(out_mp4),
    ])
    return out_mp4


# -- fitting speech to a time slot -----------------------------------------

# Speeding up or slowing down by more than this starts to sound wrong.
SOFT_SPEEDUP = 1.15
SOFT_SLOWDOWN = 0.82


def tempo_for(speech_seconds: float, target_seconds: float,
              slowest: float = SOFT_SLOWDOWN,
              fastest: float = SOFT_SPEEDUP) -> float:
    """How much to speed the speech up, to make it last target_seconds.

    atempo above 1 plays faster, so the clip gets shorter.
    """
    if speech_seconds <= 0 or target_seconds <= 0:
        return 1.0
    return min(max(speech_seconds / target_seconds, slowest), fastest)


def match_tempo(speech, target_seconds: float, out_wav,
                slowest: float = SOFT_SLOWDOWN,
                fastest: float = SOFT_SPEEDUP) -> tuple[Path, float]:
    """Bring the speech near target_seconds by changing its speed only.

    Nothing is cut: a sentence that ends early keeps its silence, and one
    that runs long is squeezed, never truncated. Returns (path, tempo).
    """
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    speech = Path(speech)
    tempo = tempo_for(duration(speech), target_seconds, slowest, fastest)

    if abs(tempo - 1.0) < 0.02:
        if speech.resolve() != out_wav.resolve():
            shutil.copy2(speech, out_wav)
        return out_wav, 1.0

    # ffmpeg refuses to read and write the same file.
    target = out_wav
    if speech.resolve() == out_wav.resolve():
        target = out_wav.with_name(out_wav.stem + ".tmp" + out_wav.suffix)
    run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(speech),
        "-filter:a", f"atempo={tempo:.6f}",
        "-c:a", "pcm_s16le", str(target),
    ])
    if target != out_wav:
        target.replace(out_wav)
    return out_wav, tempo


def trim_audio(src, seconds: float, dest) -> Path:
    """Keep only the first `seconds` of src. Used so a cue cannot overlap the next."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    length = max(float(seconds), 0.05)
    target = dest
    if src.resolve() == dest.resolve():
        target = dest.with_name(dest.stem + ".tmp" + dest.suffix)
    run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", str(src), "-t", f"{length:.3f}",
        "-c:a", "pcm_s16le", str(target),
    ])
    if target != dest:
        target.replace(dest)
    return dest


def place_clips(clips: list[tuple[float, Path]], total_seconds: float,
                out_wav) -> Path:
    """Lay each clip at its own start time on one track of total_seconds.

    This is what keeps the pauses of the original video. Reading the whole
    transcript as one block removes every gap between sentences, so the dub
    finishes early and everything is bunched at the front.
    """
    if not clips:
        raise PipelineError("There is no speech to place")

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    filters: list[str] = []
    for index, (start, path) in enumerate(clips):
        inputs += ["-i", str(path)]
        delay = int(max(start, 0.0) * 1000)
        filters.append(f"[{index}:a]aresample=44100,adelay={delay}:all=1[d{index}]")
    filters.append(
        "".join(f"[d{i}]" for i in range(len(clips)))
        + f"amix=inputs={len(clips)}:duration=longest:normalize=0,apad[a]"
    )
    run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error", *inputs,
        "-filter_complex", ";".join(filters), "-map", "[a]",
        "-t", f"{total_seconds:.3f}", "-c:a", "pcm_s16le", str(out_wav),
    ])
    return out_wav
