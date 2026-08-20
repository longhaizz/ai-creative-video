#!/bin/bash
# ==========================================================================
# Everything the pipeline needs that pip does not install and git does not
# carry. Run once, before the first job.
#
#   ./download_models.sh                       # uses /opt/venv-main
#   PY=/content/venv-main/bin/python ./download_models.sh
#
# What is NOT here: the Big-LAMA and ProPainter weights. Those are in the
# repo, split into parts, and the subtitle remover joins them on first use.
# ==========================================================================
set -euo pipefail

PY="${PY:-/opt/venv-main/bin/python}"
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
PY

# VoxCPM2 goes to HF_HOME, not into the repo. Pulling it now keeps the
# first job from stalling for several minutes with no log line.
echo "VoxCPM2 into $HF_HOME..."
"$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2', max_workers=1))"

echo
echo "Done. Checked in:"
ls -lh LatentSync/checkpoints/latentsync_unet.pt LatentSync/checkpoints/whisper/tiny.pt
