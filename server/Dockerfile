# =========================================================================
# Dub server. One image, one GPU, three virtual environments.
#
# The second venv is not a style choice. The subtitle remover needs
# numpy 2.2 for paddle, LatentSync needs numpy 1.26, and the two ABIs do not
# mix: anything built against one crashes under the other. Their torch
# builds differ as well (2.7.0+cu118 against 2.5.1+cu121).
#
# They never share a process. The worker calls the subtitle remover as a
# subprocess and waits, still holding the GPU, so it is still one job at a
# time.
#
#   /opt/venv-main   python 3.10   FastAPI, LatentSync, VoxCPM
#   /opt/venv-vsr    python 3.12   video-subtitle-remover, paddle
#   /opt/venv-od     python 3.10   Demucs, VAD, Pyannote, Whisper
#
# Model weights are NOT in the image. They are several gigabytes and they
# change on their own schedule, so they are mounted; see docker-compose.yml.
# =========================================================================

# 12.1 to match torch cu121 in the main venv. The tag needs the "8" after
# cudnn: NVIDIA only dropped that number from CUDA 12.3 onwards, so the
# 12.1.1-cudnn-... that LatentSync/Dockerfile asks for does not exist.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# python3.12 is not in Ubuntu 22.04, and the subtitle remover was tested on
# 3.12 (see video_subtitle_remover.ipynb), so it comes from deadsnakes.
# libgl1 and libglib2.0-0 are for opencv; ffmpeg does every audio job.
# build-essential and the -dev headers are for insightface: source only, and
# its C++ extension needs a compiler and Python.h.
# fonts-noto-core carries the Vietnamese glyphs the burnt-in subtitles need;
# without it every accented letter comes out as an empty box.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        python3.12 python3.12-venv python3.12-dev \
        build-essential \
        ffmpeg libgl1 libglib2.0-0 \
        fonts-noto-core fontconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------------------------------------------------------
# Main venv. Requirements are copied first, so a code change does not throw
# away the layer that took twenty minutes to build.
# -------------------------------------------------------------------------
COPY server/requirements.txt server/requirements-models.txt server/
COPY LatentSync/requirements.txt LatentSync/

RUN python3.10 -m venv /opt/venv-main \
    && /opt/venv-main/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv-main/bin/pip install -r server/requirements.txt \
    && /opt/venv-main/bin/pip install "numpy==1.26.4" "Cython<3.1" \
    && /opt/venv-main/bin/pip install insightface==0.7.3 --no-build-isolation \
    && /opt/venv-main/bin/pip install -r server/requirements-models.txt \
        --extra-index-url https://download.pytorch.org/whl/cu121

# -------------------------------------------------------------------------
# Subtitle remover venv. The order and the three index URLs come from
# video_subtitle_remover.ipynb, where this was run for real. pip takes only
# one --index-url per call, which is why this is four calls and not one
# requirements file.
# -------------------------------------------------------------------------
COPY video-subtitle-remover/requirements.txt video-subtitle-remover/
COPY server/requirements-vsr.txt server/

RUN python3.12 -m venv /opt/venv-vsr \
    && /opt/venv-vsr/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv-vsr/bin/pip install torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && /opt/venv-vsr/bin/pip install paddlepaddle-gpu==3.0.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
    && /opt/venv-vsr/bin/pip install -r video-subtitle-remover/requirements.txt \
    && /opt/venv-vsr/bin/pip install onnxruntime-gpu==1.20.1 \
        --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/

# The high performance inference plugin, about 1.2 GB. MPLBACKEND keeps
# matplotlib away from a display it will not find, and the other variable
# stops paddle from calling home to check the model host.
RUN MPLBACKEND=Agg PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    /opt/venv-vsr/bin/paddlex --install hpi-gpu

# -------------------------------------------------------------------------
# Open Dubbing venv. pyannote's transformers stack does not sit next to
# LatentSync, so Demucs + VAD + Whisper run in their own interpreter.
# -------------------------------------------------------------------------
COPY server/requirements-open-dubbing.txt server/

RUN python3.10 -m venv /opt/venv-od \
    && /opt/venv-od/bin/pip install --upgrade pip \
    && /opt/venv-od/bin/pip install torch torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
    && /opt/venv-od/bin/pip install -r server/requirements-open-dubbing.txt

# -------------------------------------------------------------------------
# The code, last, so that editing it rebuilds only this layer.
# -------------------------------------------------------------------------
COPY server/ server/
COPY LatentSync/ LatentSync/
COPY VoxCPM/ VoxCPM/
COPY video-subtitle-remover/ video-subtitle-remover/

ENV VSR_PYTHON=/opt/venv-vsr/bin/python \
    VSR_DIR=/app/video-subtitle-remover \
    OPEN_DUBBING_PYTHON=/opt/venv-od/bin/python \
    LATENTSYNC_DIR=/app/LatentSync \
    JOBS_DIR=/app/jobs \
    HF_HOME=/models/huggingface

EXPOSE 8000

# One worker on purpose. A second one would take jobs from the same queue
# and put two of them on the same card.
CMD ["/opt/venv-main/bin/uvicorn", "server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
