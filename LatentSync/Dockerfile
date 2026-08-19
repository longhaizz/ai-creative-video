FROM nvidia/cuda:12.1.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3 install --no-cache-dir -r requirements.txt

COPY requirements-api.txt .
RUN pip3 install --no-cache-dir -r requirements-api.txt

COPY . .

RUN pip3 install --no-cache-dir "huggingface_hub[cli]" \
    && mkdir -p checkpoints \
    && huggingface-cli download ByteDance/LatentSync-1.6 \
        whisper/tiny.pt latentsync_unet.pt \
        --local-dir checkpoints

ENV MPLBACKEND=Agg
ENV HF_HOME=/models
ENV NVIDIA_VISIBLE_DEVICES=all

EXPOSE 8001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
