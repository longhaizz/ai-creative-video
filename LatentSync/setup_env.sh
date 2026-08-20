#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cho phép truyền đường dẫn Python từ bên ngoài.
if [[ -n "${LATENTSYNC_PYTHON:-}" ]]; then
    PYTHON_BIN="$LATENTSYNC_PYTHON"
elif [[ -x "/opt/venv-main/bin/python" ]]; then
    PYTHON_BIN="/opt/venv-main/bin/python"
elif [[ -x "/content/venv-main/bin/python" ]]; then
    PYTHON_BIN="/content/venv-main/bin/python"
else
    echo "Không tìm thấy venv-main."
    echo "Hãy đặt LATENTSYNC_PYTHON=/duong-dan/toi/python"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

# Xác nhận PyTorch nhận GPU.
"$PYTHON_BIN" -c "
import torch

print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError('PyTorch không nhận GPU')

print('GPU:', torch.cuda.get_device_name(0))
"

# Cài dependency vào đúng venv Python 3.10.
"$PYTHON_BIN" -m pip install -r requirements.txt

# Tải checkpoint bằng Python API mới.
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ByteDance/LatentSync-1.6",
    allow_patterns=[
        "whisper/tiny.pt",
        "latentsync_unet.pt",
    ],
    local_dir="checkpoints",
)

print("Đã tải checkpoint LatentSync")
PY

# Kiểm tra checkpoint.
test -s checkpoints/whisper/tiny.pt
test -s checkpoints/latentsync_unet.pt

echo "LatentSync setup hoàn tất"