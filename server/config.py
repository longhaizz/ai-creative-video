"""Settings read from environment variables.

Only the settings that something already uses. Keys for later steps
(OPENAI_API_KEY, VSR_PYTHON, VSR_REPO) are added when that step needs them.
"""

from __future__ import annotations

import os
from pathlib import Path

# Bearer token the client must send. If empty, the server refuses to start
# (see app.py) — better to fail at boot than to run an API with no lock.
API_KEY = os.getenv("API_KEY", "")

# Where result files live until the client downloads them or the TTL ends.
JOBS_DIR = Path(os.getenv("JOBS_DIR", "jobs"))

# When a job is older than this, delete both its state and its files. 1 hour.
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))

MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(200 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))

# -- outside programs ------------------------------------------------------
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

# -- video-subtitle-remover ------------------------------------------------
# It runs in its own venv (see server/requirements-vsr.txt), so we call it
# with that interpreter and from its own folder.
VSR_DIR = Path(os.getenv("VSR_DIR", "video-subtitle-remover"))
VSR_PYTHON = os.getenv("VSR_PYTHON", "/opt/venv-vsr/bin/python")

# -- LatentSync ------------------------------------------------------------
# The vendored source, and the two files it needs. The config and the
# checkpoint are named relative to the repo, the way upstream expects them.
LATENTSYNC_DIR = Path(os.getenv("LATENTSYNC_DIR", "LatentSync"))
LATENTSYNC_CONFIG = Path(
    os.getenv("LATENTSYNC_CONFIG", "configs/unet/stage2_512.yaml")
)
LATENTSYNC_CHECKPOINT = Path(
    os.getenv("LATENTSYNC_CHECKPOINT", "checkpoints/latentsync_unet.pt")
)
