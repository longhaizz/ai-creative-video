#!/bin/bash
# ==========================================================================
# Every model weight the pipeline needs that pip does not install and git
# does not carry. Run once, before the first job. Safe to run again: each
# download checks its cache and skips what is already there.
#
#   ./download_models.sh                       # uses /opt/venv-main, /opt/venv-vsr
#   PY=/content/venv-main/bin/python VSR_PY=/content/venv-vsr/bin/python ./download_models.sh
#
# Without this, each model below is pulled on the first job that needs it:
# that job stalls for minutes, and fails if the box has no internet.
#
# What is NOT here: the OCR detector, Big-LAMA, ProPainter and STTN weights.
# Those are in the repo (video-subtitle-remover/backend/models), and the
# subtitle remover joins the split files on first use.
# ==========================================================================
set -euo pipefail

PY="${PY:-/opt/venv-main/bin/python}"
VSR_PY="${VSR_PY:-/opt/venv-vsr/bin/python}"
export HF_HOME="${HF_HOME:-/models/huggingface}"

cd "$(dirname "$0")"

# LatentSync. Named files in a fixed place: server/config.py looks for
# checkpoints/latentsync_unet.pt, and the config asks for whisper/tiny.pt.
echo "LatentSync checkpoints (about 1.3 GB)..."
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ByteDance/LatentSync-1.6",
    allow_patterns=["whisper/tiny.pt", "latentsync_unet.pt"],
    local_dir="LatentSync/checkpoints",
)
# scripts/inference.py loads this VAE by name from HF_HOME.
snapshot_download("stabilityai/sd-vae-ft-mse")
PY

# The LatentSync face detector. insightface downloads buffalo_l into
# checkpoints/auxiliary/models the first time FaceAnalysis is built, so
# building it once here is the download. CPU is enough for that.
echo "insightface buffalo_l..."
(cd LatentSync && "$PY" -c "
from insightface.app import FaceAnalysis
FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'],
             root='checkpoints/auxiliary', providers=['CPUExecutionProvider'])
")

# VoxCPM2 goes to HF_HOME, not into the repo. Pulling it now keeps the
# first job from stalling for several minutes with no log line.
echo "VoxCPM2 into $HF_HOME..."
"$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2', max_workers=1))"

# Every size a job may ask for (server/steps/transcribe.py WHISPER_MODELS).
# About 5 GB in all; medium is the default one.
echo "faster-whisper models into $HF_HOME..."
"$PY" - <<'PY'
from faster_whisper.utils import download_model
from server.steps.transcribe import WHISPER_MODELS
for size in WHISPER_MODELS:
    print("  ", size, download_model(size))
PY

# server/steps/separate.py uses htdemucs. It goes to the torch hub cache.
echo "demucs htdemucs..."
"$PY" -c "from demucs.pretrained import get_model; get_model('htdemucs')"

# OCR text recognition, one model per script. Unlike the detector these are
# not in the repo: PaddleOCR downloads them to ~/.paddlex/official_models.
# The list is the one speech_match.py picks from, so a new language there
# is picked up here too.
echo "PaddleOCR recognition models..."
(cd video-subtitle-remover && MPLBACKEND=Agg PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True "$VSR_PY" - <<'PY'
from paddleocr import TextRecognition
from backend.tools.speech_match import LATIN_REC_MODEL, REC_MODELS
for name in sorted(set(REC_MODELS.values()) | {LATIN_REC_MODEL}):
    print("  ", name)
    TextRecognition(model_name=name, device="cpu")
PY
)

echo
echo "Done. Checked in:"
ls -lh LatentSync/checkpoints/latentsync_unet.pt LatentSync/checkpoints/whisper/tiny.pt
ls -d LatentSync/checkpoints/auxiliary/models/buffalo_l
