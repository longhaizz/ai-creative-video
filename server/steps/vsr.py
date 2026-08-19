"""Remove the subtitles that are burned into the video.

This step runs in its own venv, through a subprocess. The reason is numpy:
paddle here wants 2.2, LatentSync in the main venv wants 1.26, and the two
cannot live together. The worker waits for the subprocess while still
holding the GPU, so it is still one job at a time.

The command line comes from video_subtitle_remover.ipynb, which is the only
place these settings were tried on real ad creatives.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from server import config
from server.jobs import JobCancelled, PipelineError

# Only send a line to the job log this often. The tool prints a tqdm bar,
# and every redraw arrives here as its own line, so without this the log
# would be thousands of near-identical rows.
LOG_EVERY_SECONDS = 2.0

# Keep this many lines to explain a failure.
TAIL_LINES = 25


def probe_size(video: Path) -> tuple[int, int]:
    """Return (width, height) of the video."""
    result = subprocess.run(
        [
            config.FFPROBE_BIN,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PipelineError(
            f"Could not read the video: {result.stderr.strip()[:200]}",
            code="invalid_input",
        )
    streams = json.loads(result.stdout).get("streams") or []
    if not streams:
        raise PipelineError("The file has no video stream", code="invalid_input")
    return int(streams[0]["width"]), int(streams[0]["height"])


def area_to_pixels(
    width: int,
    height: int,
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> tuple[int, int, int, int]:
    """Turn a share of the frame into (ymin, ymax, xmin, xmax) in pixels.

    The client sends shares, not pixels, because it does not know the size
    of the video until the server opens it, and the same settings have to
    work for 720p and 1080p alike.
    """
    ymin = int(height * top)
    ymax = int(height * bottom)
    xmin = int(width * left)
    xmax = int(width * right)
    # One row and one column at least, or the tool has nothing to look at.
    return ymin, max(ymax, ymin + 1), xmin, max(xmax, xmin + 1)


def build_command(
    video: Path,
    out_path: Path,
    mode: str,
    area: tuple[int, int, int, int],
) -> list[str]:
    ymin, ymax, xmin, xmax = area
    return [
        str(config.VSR_PYTHON),
        "backend/main.py",
        "--input", str(video),
        "--output", str(out_path),
        "--subtitle-area-coords", str(ymin), str(ymax), str(xmin), str(xmax),
        "--inpaint-mode", mode,
    ]


def _child_env() -> dict:
    return {
        **os.environ,
        # No display on a server, and paddle pulls matplotlib in.
        "MPLBACKEND": "Agg",
        # Do not phone home to check the model host; the models are local.
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }


def remove_subtitles(
    video: Path,
    out_path: Path,
    mode: str,
    top: float,
    bottom: float,
    left: float,
    right: float,
    ctx=None,
) -> Path:
    """Paint over the burned-in subtitles. Returns out_path."""
    video = Path(video).resolve()
    out_path = Path(out_path).resolve()

    width, height = probe_size(video)
    area = area_to_pixels(width, height, top, bottom, left, right)
    command = build_command(video, out_path, mode, area)

    if ctx is not None:
        ctx.log(
            f"Removing subtitles ({mode}) in {width}x{height}, "
            f"area y={area[0]}-{area[1]} x={area[2]}-{area[3]}"
        )

    process = subprocess.Popen(
        command,
        cwd=str(config.VSR_DIR),
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    tail: list[str] = []
    last_log = 0.0
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            tail.append(line)
            del tail[:-TAIL_LINES]
            if ctx is None:
                continue
            # Cancel is checked here rather than on a timer, because the
            # tool prints often enough to come back here every second or so.
            if _cancelled(ctx):
                process.kill()
                raise JobCancelled()
            now = time.monotonic()
            if now - last_log >= LOG_EVERY_SECONDS:
                last_log = now
                ctx.log(line)
    finally:
        process.stdout.close()
        process.wait()

    if process.returncode != 0:
        raise PipelineError(
            "Subtitle removal failed:\n" + "\n".join(tail),
            code="internal",
        )
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise PipelineError("Subtitle removal produced no video")
    return out_path


def _cancelled(ctx) -> bool:
    try:
        ctx.check_cancel()
    except JobCancelled:
        return True
    return False
