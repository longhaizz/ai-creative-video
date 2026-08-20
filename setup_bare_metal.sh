#!/bin/bash
# ==========================================================================
# The whole bare-metal environment in one run: system packages, the two
# venvs, the model weights. Ubuntu 22.04, NVIDIA driver 525 or newer (13.x is
# fine, newer drivers still run the cu121 build).
#
#   ./setup_bare_metal.sh
#
# It is safe to run again: every step skips itself when it is already done.
# The long version, with the reasons behind each choice, is in
# server/docs/bare-metal.md.
#
# Not here: the .env file (yours) and PM2 (see ecosystem.config.js).
# ==========================================================================
set -euo pipefail

MAIN=/opt/venv-main
VSR=/opt/venv-vsr
export HF_HOME="${HF_HOME:-/models/huggingface}"

cd "$(dirname "$0")"

say() { echo; echo "=== $* ==="; }

# --- 0. preflight -----------------------------------------------------------
# The torch here is built for CUDA 12.1, and a newer driver runs it fine:
# drivers are backward compatible with older toolkits. A newer *card* is the
# problem. Blackwell (sm_120, RTX 50xx and B200) has no kernels in a cu121
# build, so torch loads and then fails on the first matmul. Check now, not
# an hour into the install.
say "preflight"
command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: install the driver first"; exit 1; }
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
if [ "${CAP:-0}" -ge 120 ] 2>/dev/null; then
    echo
    echo "This card is sm_$CAP (Blackwell). The pinned torch 2.5.1+cu121 has no"
    echo "kernels for it and every job will fail. It needs torch on cu128 or"
    echo "newer, which means new pins for torch, torchvision and torchaudio."
    exit 1
fi

# --- 1. system packages -----------------------------------------------------
# Two Pythons: paddle in the subtitle remover needs numpy 2.2, LatentSync
# needs 1.26, and the two ABIs do not share a process.
#
# 3.10 comes from deadsnakes even on a box that already has 3.12. The main
# venv cannot move to 3.12: mediapipe is pinned at 0.10.11, whose last
# wheels are cp311. The vsr venv is the one that wants 3.12.
say "system packages"
sudo apt-get update
sudo apt-get install -y --no-install-recommends software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
# build-essential and the -dev headers are for insightface: it ships source
# only, and its C++ extension needs a compiler and Python.h.
sudo apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    python3.12 python3.12-venv python3.12-dev \
    build-essential \
    ffmpeg libgl1 libglib2.0-0 \
    fonts-noto-core fontconfig curl

# --- 2. main venv -----------------------------------------------------------
say "main venv ($MAIN)"
[ -x "$MAIN/bin/python" ] || sudo python3.10 -m venv "$MAIN"
sudo chown -R "$(id -u):$(id -g)" "$MAIN"
"$MAIN/bin/pip" install --upgrade pip setuptools wheel
"$MAIN/bin/pip" install -r server/requirements.txt

# insightface, out of order and on its own. It has no wheel, and its setup.py
# imports numpy and Cython while pip builds it in an isolated env that has
# neither, so the build fails. Installing them here and passing
# --no-build-isolation lets the build see them. Cython must stay under 3.1:
# the .pyx in 0.7.3 does not compile with 3.1.
# Once it is installed the requirements line below finds it and moves on.
"$MAIN/bin/pip" install "numpy==1.26.4" "Cython<3.1"
"$MAIN/bin/pip" install insightface==0.7.3 --no-build-isolation
"$MAIN/bin/pip" install -r server/requirements-models.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# numpy must stay 1.26.x. A 2.x here means something pulled it up and
# LatentSync will blow up at run time, so stop now rather than at the first job.
"$MAIN/bin/python" - <<'PY'
import numpy, torch, sys
print("main venv:", torch.__version__, "cuda:", torch.cuda.is_available(), "numpy:", numpy.__version__)
if not numpy.__version__.startswith("1.26"):
    sys.exit(f"numpy must be 1.26.x in the main venv, got {numpy.__version__}")
PY

# --- 3. subtitle remover venv ----------------------------------------------
# Four pip calls, not one requirements file: three of them need their own
# index and pip takes only one --index-url per run. Order matters.
say "subtitle remover venv ($VSR)"
[ -x "$VSR/bin/python" ] || sudo python3.12 -m venv "$VSR"
sudo chown -R "$(id -u):$(id -g)" "$VSR"
"$VSR/bin/pip" install --upgrade pip setuptools wheel
"$VSR/bin/pip" install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu118
"$VSR/bin/pip" install paddlepaddle-gpu==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
"$VSR/bin/pip" install -r video-subtitle-remover/requirements.txt
"$VSR/bin/pip" install onnxruntime-gpu==1.20.1 \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

# The high performance inference plugin, about 1.2 GB, downloaded once.
MPLBACKEND=Agg PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    "$VSR/bin/paddlex" --install hpi-gpu

"$VSR/bin/python" - <<'PY'
import numpy, paddle, torch, sys
print("vsr venv:", torch.__version__, paddle.device.get_device(), "numpy:", numpy.__version__)
if not numpy.__version__.startswith("2.2"):
    sys.exit(f"numpy must be 2.2.x in the vsr venv, got {numpy.__version__}")
PY

# --- 4. model weights -------------------------------------------------------
say "model weights (HF_HOME=$HF_HOME)"
sudo mkdir -p "$HF_HOME"
sudo chown -R "$(id -u):$(id -g)" "$HF_HOME"
PY="$MAIN/bin/python" ./download_models.sh

say "done"
[ -f .env ] || echo "Warning: no .env yet. PM2 reads it for API_KEY and OPENAI_API_KEY."
cat <<'EOF'
Next:
  pm2 start ecosystem.config.js
  curl -H "Authorization: Bearer $API_KEY" localhost:8000/health

The first start takes 1-2 minutes: it loads the models.
EOF
