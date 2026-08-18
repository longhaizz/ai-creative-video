#!/usr/bin/env bash
# Step 1: test CLI inference inside image (GPU server)
set -euo pipefail

docker build -t latentsync:try .

mkdir -p data/in data/out
# Put face.mp4 and speech.wav in data/in/ before running

docker run --rm --gpus all \
  -v latentsync-hf:/models \
  -v "$(pwd)/data/in:/in" \
  -v "$(pwd)/data/out:/out" \
  latentsync:try \
  python -m scripts.inference \
    --unet_config_path configs/unet/stage2_512.yaml \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --enable_deepcache \
    --video_path /in/face.mp4 \
    --audio_path /in/speech.wav \
    --video_out_path /out/out.mp4

ls -lh data/out/out.mp4
