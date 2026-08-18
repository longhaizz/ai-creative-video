#!/usr/bin/env bash
# Run on GPU server after: docker compose up -d --build
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8001}"
KEY="${LATENTSYNC_API_KEY:?Set LATENTSYNC_API_KEY}"
VIDEO="${VIDEO:-assets/demo1_video.mp4}"
AUDIO="${AUDIO:-assets/demo1_audio.wav}"

curl -sf -H "Authorization: Bearer ${KEY}" "${BASE}/health" | python3 -m json.tool

curl -sf -X POST "${BASE}/sync" \
  -H "Authorization: Bearer ${KEY}" \
  -F "video=@${VIDEO}" \
  -F "audio=@${AUDIO}" \
  -F "inference_steps=20" \
  -F "guidance_scale=1.5" \
  -F "enable_deepcache=true" \
  --output /tmp/latentsync_result.mp4

ls -lh /tmp/latentsync_result.mp4
