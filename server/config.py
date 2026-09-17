"""Settings read from environment variables.

Only the settings that something already uses. Keys for later steps
(LLM_PROVIDER and its keys, VSR_PYTHON, VSR_REPO) are added when that step
needs them.
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

# What every take measured, kept between jobs so the length guess gets
# better. Deliberately NOT under JOBS_DIR: that folder is wiped at boot.
DURATION_DATA = Path(os.getenv("DURATION_DATA", "data/duration.csv"))

# The preset voices: one wav per voice, and the file name is the voice id.
# Made from the samples one folder up by server/scripts/prepare_voices.py.
VOICES_DIR = Path(
    os.getenv("VOICES_DIR", Path(__file__).resolve().parent / "voices" / "wav")
)

MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(200 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))

# The cap on a whole request, checked before the body is read at all. The
# two limits above are checked while saving, which is far too late: by then
# the body is already spooled to disk, token or no token.
# Room for a video, a voice sample and the form fields around them.
MAX_REQUEST_BYTES = int(
    os.getenv("MAX_REQUEST_BYTES", str(MAX_VIDEO_BYTES + MAX_AUDIO_BYTES + 1024 * 1024))
)

# The language model that rewrites and translates. The keys live here, on
# the server, so they never ship inside the desktop .exe.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# The key of the provider in use. Empty means that provider has no key.
LLM_API_KEY = GEMINI_API_KEY if LLM_PROVIDER == "gemini" else OPENAI_API_KEY
# Empty means the provider's default model, see steps/llm.py.
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
# A piece of screen text that OCR read with a lower score than this is
# sent to the model with a picture of it, so the model reads it again.
SCREEN_VISION_BELOW = float(os.getenv("SCREEN_VISION_BELOW", "0.95"))

# Set to 0 to start without the models. The API answers, and every job fails
# with "not built yet". Only useful for working on the HTTP side on a
# machine with no GPU.
LOAD_MODELS = os.getenv("LOAD_MODELS", "1") not in ("0", "false", "no")

# LatentSync is loaded unless this is set to 0. It sits on the GPU even when
# a job never asks for lip sync. While it is off, a job that sends
# lipsync=true runs as if the box was not ticked: no mouth work, no error.
LOAD_LIPSYNC = os.getenv("LOAD_LIPSYNC", "1") not in ("0", "false", "no")

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
