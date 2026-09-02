# Dub server

Một video vào → một video ra. Toàn bộ pipeline chạy ở đây, trên **một GPU,
một hàng đợi, một worker**:

```
xoá sub → Open Dubbing (Demucs + VAD + Pyannote + Whisper)
    → dịch (OpenAI) → đọc (VoxCPM, clone từng câu) → khớp miệng
    → trộn → mux → burn sub
```

Thiết kế và lộ trình: [`../PLAN.md`](../PLAN.md).

---

## Chạy production

```bash
cp .env.example .env      # điền API_KEY, OPENAI_API_KEY, HF_TOKEN
cd LatentSync && bash setup_env.sh && cd ..   # tải weight, ~1.3GB
docker compose up -d --build
```

Lần khởi động đầu mất 1–2 phút để nạp model. Không dùng Docker được thì
xem [`docs/bare-metal.md`](docs/bare-metal.md).

## Chạy dev (không cần GPU)

Chạy từ **gốc repo**, không phải trong `server/` — `server` là package.

```bash
pip install -r server/requirements.txt
```

`requirements.txt` **cố ý không có torch**. Toàn bộ hàng đợi, API, phần huỷ,
phép canh giờ và phần phụ đề đều test được trên laptop. Muốn chạy job thật
thì thêm `requirements-models.txt`.

**PowerShell** — không có cú pháp `VAR=value cmd`, phải set biến trước:

```powershell
$env:API_KEY = "dev-key"
$env:LOAD_MODELS = "0"
uvicorn server.app:app --reload
```

**Bash**:

```bash
API_KEY=dev-key LOAD_MODELS=0 uvicorn server.app:app --reload
```

`LOAD_MODELS=0` cho server chạy mà không nạp model. API trả lời bình thường,
mọi job fail ngay với *"The dub pipeline is not built yet"*.

## Test

```bash
pytest server/tests
```

Không cần GPU. Một ca cần `ffmpeg` trên PATH, thiếu thì tự bỏ qua.

Chạy thật cả chuỗi (cần GPU + model + OpenAI key):

```bash
python -m server.tests.smoke_run_dub clip.mp4 --lipsync --burn-subtitle
```

Đây không phải file pytest — tên không bắt đầu bằng `test_` nên không bao
giờ chạy nhầm.

## API

```
POST   /dub                  → 202 {job_id}
POST   /speak                → 202 {job_id}
GET    /jobs/{id}?since=N    → {status, step, log, queue_position, error_code}
GET    /jobs/{id}/result     → mp4 hoặc wav, rồi job bị xoá
DELETE /jobs/{id}            → huỷ
GET    /health               → {status, models_loaded, gpu}
```

Mọi endpoint cần `Authorization: Bearer <API_KEY>`, kể cả `/health`.

`status`: `queued | running | done | failed | cancelled`
`error_code`: `no_face | invalid_input | internal`

`no_face` **không phải hỏng** — creative quảng cáo thường không có
talking-head, pipeline bỏ qua lipsync và vẫn xuất video.

### POST /dub

Một video vào → một video lồng tiếng ra.

```bash
JOB=$(curl -s -H "Authorization: Bearer $API_KEY" \
    -F "video=@clip.mp4" -F "lipsync=true" -F "target_lang=vi" \
    localhost:8000/dub | jq -r .job_id)

curl -H "Authorization: Bearer $API_KEY" "localhost:8000/jobs/$JOB?since=0"
curl -H "Authorization: Bearer $API_KEY" "localhost:8000/jobs/$JOB/result" -o out.mp4
```

### POST /speak

Một đoạn text + một file audio mẫu giọng → một file WAV. Server clone giọng
từ audio mẫu (VoxCPM) rồi đọc text — không cần video, không dịch, không khớp
miệng. Cùng hàng đợi và cùng luồng job với `/dub`.

| Field | Bắt buộc | Mặc định | Ghi chú |
|---|---|---|---|
| `text` | có | — | Nội dung cần đọc (1–8000 ký tự) |
| `audio` | có | — | File mẫu giọng (wav/mp3, tối đa 25MB) |
| `cfg_value` | không | `2.0` | Độ bám text, 1.0–3.0 |
| `inference_timesteps` | không | `10` | Số bước sinh, 5–30 |

```bash
JOB=$(curl -s -H "Authorization: Bearer $API_KEY" \
    -F "text=Xin chào, đây là giọng đã clone" \
    -F "audio=@voice_sample.wav" \
    localhost:8000/speak | jq -r .job_id)

curl -H "Authorization: Bearer $API_KEY" "localhost:8000/jobs/$JOB?since=0"
curl -H "Authorization: Bearer $API_KEY" "localhost:8000/jobs/$JOB/result" -o out.wav
```

Khác với `/dub`: `/speak` **chỉ clone từ audio mẫu**, không dùng preset giọng
(`male_young`, `female_old`, …). Preset chỉ có trên `/dub` qua `voice_mode`.

Tham số đầy đủ: [`schemas.py`](schemas.py) (`DubParams`, `SpeakParams`).

## Biến môi trường

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `API_KEY` | — | **Bắt buộc.** Thiếu thì server không khởi động |
| `OPENAI_API_KEY` | — | Cần cả khi giữ nguyên ngôn ngữ |
| `LOAD_MODELS` | `1` | `0` = chạy không model |
| `LOAD_LIPSYNC` | `1` | `0` = không nạp LatentSync (T4 / card 16GB) |
| `OPEN_DUBBING_PYTHON` | `/opt/venv-od/bin/python` | venv Demucs/VAD/Pyannote/Whisper |
| `HF_TOKEN` | — | **Bắt buộc cho job.** Pyannote gated |
| `JOBS_DIR` | `jobs` | |
| `JOB_TTL_SECONDS` | `3600` | |
| `MAX_VIDEO_BYTES` | 200MB | kiểm lúc lưu |
| `MAX_AUDIO_BYTES` | 25MB | kiểm lúc lưu |
| `MAX_REQUEST_BYTES` | 226MB | kiểm **trước khi đọc body** |
| `VSR_PYTHON` | `/opt/venv-vsr/bin/python` | |
| `VSR_DIR` | `video-subtitle-remover` | |
| `LATENTSYNC_DIR` | `LatentSync` | |
| `LATENTSYNC_CONFIG` | `configs/unet/stage2_512.yaml` | |
| `LATENTSYNC_CHECKPOINT` | `checkpoints/latentsync_unet.pt` | |
| `FFMPEG_BIN` / `FFPROBE_BIN` | `ffmpeg` / `ffprobe` | |

## Ba điều đáng biết trước khi sửa code

**Một GPU, một worker.** `JobRunner.start()` gọi lần hai là `RuntimeError`,
và uvicorn phải chạy `--workers 1`. Worker thứ hai sẽ đặt hai job lên cùng
một card.

**Bộ xoá phụ đề và Open Dubbing chạy venv riêng.** numpy/transformers của paddle và pyannote không sống chung với LatentSync. Cả hai được gọi bằng `subprocess`.

**Trạng thái nằm trong RAM.** Restart là mất job. Đúng thiết kế: mấy phút
GPU đã đốt thì database cũng không tua lại được.

## Bố cục

```
app.py         HTTP, auth, tạo app
jobs.py        JobRunner — hàng đợi, worker, trạng thái, TTL
pipeline.py    run_dub — ghép 9 bước
limits.py      chặn upload quá cỡ trước khi chạm đĩa
schemas.py     tham số POST /dub và POST /speak
steps/         mỗi file một bước
tests/         101 ca, không cần GPU
```

`run_dub` được **truyền vào** `JobRunner`, không import. Đó là chỗ cắt duy
nhất: test đưa vào một pipeline giả và kiểm được toàn bộ phần còn lại mà
không cần card nào.
