# Dub server

Một video vào → một video ra. Toàn bộ pipeline (xoá sub → tách nhạc → ASR →
dịch → TTS → lipsync → mix → mux → burn sub) chạy ở đây, trên **một GPU, một
queue, một worker**.

Thiết kế và lộ trình: [`../PLAN.md`](../PLAN.md).

## Chạy dev

```bash
python -m venv .venv && . .venv/Scripts/activate    # Linux: . .venv/bin/activate
pip install -r server/requirements.txt

API_KEY=dev-key uvicorn server.app:app --reload
curl localhost:8000/health
```

Chạy từ **gốc repo**, không phải trong `server/` — `server` là package.

## Test

```bash
pytest server/tests
```

Không cần GPU. Bước 3-8 giữ nguyên tính chất này: toàn bộ logic queue/job/
auth/huỷ test được bằng một `run_dub` giả, chỉ pipeline thật mới cần GPU.

## Biến môi trường

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `API_KEY` | — | **Bắt buộc.** Thiếu thì server không khởi động. |
| `JOBS_DIR` | `jobs` | Nơi giữ file kết quả tới khi client tải hoặc hết TTL |
| `JOB_TTL_SECONDS` | `3600` | |
| `MAX_VIDEO_BYTES` | 200MB | |
| `MAX_AUDIO_BYTES` | 25MB | |

Các biến của bước sau (`OPENAI_API_KEY`, `VSR_PYTHON`, `VSR_REPO`) thêm vào
đúng lúc bước đó cần.

## Trạng thái

Mới xong bước 2 — khung rỗng, chỉ có `/health`.
