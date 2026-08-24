"""Time LatentSync alone, at two settings, on the same clip.

Not a pytest file, on purpose: it needs the GPU and the checkpoint. It also
skips the rest of the pipeline (demucs, whisper, OpenAI, VoxCPM), so what it
prints is lip sync and nothing else. Use it when a job feels slow and you
want to know whether steps and guidance are really reaching the model.

    python -m server.tests.bench_lipsync clip.mp4 voice.wav
    python -m server.tests.bench_lipsync clip.mp4 voice.wav --runs 20:1.0 50:1.5

The audio drives how many frames are written, so pass a WAV as long as the
clip. Any voice track will do.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from server import config
from server.steps.lipsync import LipsyncModel


def _forwards(seconds: float, fps: int, num_frames: int, steps: int, guidance: float) -> int:
    """How many UNet passes a run costs, from the pipeline's own loop.

    ceil(frames / num_frames) chunks, `steps` denoising steps each, doubled
    when guidance > 1.0 because that turns classifier-free guidance on.
    """
    chunks = math.ceil(seconds * fps / num_frames)
    return chunks * steps * (2 if guidance > 1.0 else 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--runs", nargs="+", default=["50:1.5", "20:1.0"],
                    help="steps:guidance pairs, in the order to run them")
    ap.add_argument("--out-dir", type=Path, default=Path("bench_out"))
    args = ap.parse_args(argv)

    runs = []
    for pair in args.runs:
        steps, guidance = pair.split(":")
        runs.append((int(steps), float(guidance)))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = LipsyncModel(
        config.LATENTSYNC_DIR, config.LATENTSYNC_CONFIG, config.LATENTSYNC_CHECKPOINT
    )
    started = time.perf_counter()
    model.load()
    print(f"model loaded in {time.perf_counter() - started:.1f}s")

    fps = 25
    num_frames = model._config.data.num_frames
    baseline = None
    for steps, guidance in runs:
        out = args.out_dir / f"steps{steps}_g{guidance}.mp4"
        started = time.perf_counter()
        model.run(args.video.resolve(), args.audio.resolve(), out.resolve(),
                  steps=steps, guidance=guidance)
        took = time.perf_counter() - started
        baseline = baseline or took
        print(f"steps={steps:<3} guidance={guidance:<4} {took:7.1f}s "
              f"({took / baseline:.2f}x of the first run) -> {out}")

    # What the numbers should look like, so a flat result is obvious.
    print("\nExpected UNet passes (time is roughly proportional, plus a "
          "fixed cost for face detection and rebuilding the video):")
    seconds = None
    try:
        from server.steps import audio
        seconds = audio.duration(args.audio)
    except Exception:                      # ffprobe missing: skip the maths
        pass
    if seconds:
        for steps, guidance in runs:
            print(f"  steps={steps:<3} guidance={guidance:<4} "
                  f"{_forwards(seconds, fps, num_frames, steps, guidance)}")
    return 0


def _selfcheck():
    # 10s at 25fps in chunks of 16 = 16 chunks. Guidance > 1 doubles it.
    assert _forwards(10, 25, 16, 50, 1.5) == 1600
    assert _forwards(10, 25, 16, 50, 1.0) == 800
    assert _forwards(10, 25, 16, 20, 1.0) == 320
    print("bench_lipsync.py self-check OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        sys.exit(main())
