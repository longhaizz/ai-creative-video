# Plan: Dub Server — gộp VoxCPM + LatentSync + VSR thành một job API

Trạng thái: đã chốt thiết kế, chưa code.
Repo: `longhaizz/ai-creative-video`

---

## 1. Vấn đề

Tool `spy-ads-creative-desktop-tool` đang chạy phần nặng **trên máy user**: demucs (CPU), faster-whisper (process con riêng vì `ctranslate2 + Qt = ACCESS_VIOLATION`), Wav2Lip (96×96, chất lượng thấp). User phải tự tải model weight, .exe phình to, và mỗi máy chạy một tốc độ khác nhau.

Phần đã đẩy lên server thì lại nửa vời: VoxCPM có API nhưng client **không gửi Bearer token** nên gọi là 401; LatentSync có API nhưng chưa ai gọi. Hai server dùng chung một GPU nhưng mỗi cái giữ `threading.Lock` riêng, không biết nhau — cùng nhận request là OOM.

Và mọi thứ đều đồng bộ: một request lipsync giữ HTTP connection suốt nhiều phút, rớt mạng là mất trắng.

## 2. Giải pháp

**Một video vào → một video ra.** Client upload video + tham số, nhận `job_id`, poll tới khi xong, tải mp4 về. Toàn bộ 8 bước xử lý chạy server-side trong **một process, một queue, một worker, một lock** trên GPU 48GB.

Máy user chỉ còn cần `ffmpeg`.

## 3. User stories

1. Là người dựng creative, tôi muốn chọn nhiều video rồi bấm một nút, để nhận về video đã lồng tiếng mà không phải cài model gì.
2. Là người dựng creative, tôi muốn tích "Xoá sub" để phụ đề cháy sẵn trong video gốc biến mất trước khi lồng tiếng.
3. Là người dựng creative, tôi muốn tích "Lipsync" để khẩu hình khớp giọng mới, và bỏ tích khi video không có mặt người.
4. Là người dựng creative, tôi muốn tích "Tạo sub" để video ra có sẵn phụ đề tiếng đích.
5. Là người dựng creative, tôi muốn chọn giọng đọc (giữ giọng gốc hoặc 6 preset nam/nữ × trẻ/trung/già), để hợp với nội dung quảng cáo.
6. Là người dựng creative, tôi muốn chọn ngôn ngữ đích, để phát hành ở thị trường khác.
7. Là người dựng creative, tôi muốn mỗi lựa chọn kỹ thuật đều có nhãn tiếng Việt kèm khuyến nghị, để không phải hiểu con số nghĩa là gì.
8. Là người dựng creative, tôi muốn thấy log chạy theo thời gian thực, để biết nó đang ở bước nào chứ không phải đứng hình.
9. Là người dựng creative, tôi muốn biết job của mình đang xếp thứ mấy trong hàng, để ước lượng thời gian chờ.
10. Là người dựng creative, tôi muốn bấm Stop huỷ được job, để không phí GPU khi chọn nhầm.
11. Là người dựng creative, tôi muốn nộp cả lô video một lượt, để không phải ngồi canh từng cái.
12. Là người dựng creative, tôi muốn một video lỗi không làm hỏng cả lô, để lấy được phần còn lại.
13. Là người dựng creative, tôi muốn video không có mặt người vẫn ra kết quả (bỏ qua lipsync, không tính là lỗi), vì creative quảng cáo thường không có talking-head.
14. Là người dựng creative, tôi muốn chọn vùng quét sub (mặc định dưới màn hình), để xử lý được cả video có sub nằm giữa khung.
15. Là người dựng creative, tôi muốn chọn thuật toán xoá sub, để đổi cách khi kết quả mặc định chưa sạch.
16. Là người dựng creative, tôi muốn chỉnh font/cỡ/vị trí phụ đề, để khớp bộ nhận diện của khách.
17. Là người dựng creative, tôi muốn chọn model Whisper, để đổi giữa nhanh và chính xác.
18. Là người dựng creative, tôi muốn upload một file WAV giọng mẫu, để nhân bản đúng giọng đó thay vì giọng trong video.
19. Là người dựng creative, tôi muốn nhạc nền của video gốc còn nguyên sau khi thay giọng.
20. Là người dựng creative, tôi muốn giọng mới rơi đúng vào nhịp câu của video gốc, không bị dồn hết về đầu clip.
21. Là admin, tôi muốn API key và base URL nằm trong `global-configs.yaml` dạng mã hoá, để user không sửa và không lộ.
22. Là admin, tôi muốn OpenAI key nằm trên server, để nó không được phát tán theo mỗi bản .exe.
23. Là admin, tôi muốn server chỉ nhận request có Bearer token đúng.
24. Là admin, tôi muốn `GET /health` cho biết model nào đã nạp xong, để biết khi nào nhận việc được.
25. Là admin, tôi muốn file kết quả tự xoá sau 1 giờ, để đĩa không đầy dần.
26. Là admin, tôi muốn deploy bằng một `docker compose up`.
27. Là admin, tôi muốn có hướng dẫn dựng bare-metal, để chạy được trên máy không dùng Docker.
28. Là dev, tôi muốn test được toàn bộ logic queue/job/auth/huỷ mà không cần GPU.
29. Là dev, tôi muốn code mình viết tách hẳn khỏi cây source upstream, để biết chỗ nào là của mình khi cần nâng cấp upstream.
30. Là dev, tôi muốn lỗi "không dò được mặt" phân biệt được với lỗi thật bằng mã máy đọc, không phải so chuỗi tiếng Anh của upstream.

## 4. Quyết định triển khai

### Kiến trúc

- **Một process** giữ GPU. Một `queue.Queue` không giới hạn độ sâu, một worker thread, một lock. Submit luôn nhận `202` — không bao giờ trả `429`.
- **Preload lúc khởi động**: VoxCPM2, LatentSync (UNet3D + VAE + whisper riêng), faster-whisper, demucs. VRAM 48GB thừa sức.
- **VSR chạy `subprocess`** sang venv riêng, worker giữ lock trong lúc chờ → vẫn đúng một job một lúc. Lý do tách: numpy `1.26.4` (LatentSync) vs `2.2.5` (VSR) xung đột ABI, cộng thêm paddle và torch `2.7.0+cu118` vs `2.5.1+cu121`.
- **Job state trong RAM**, file kết quả trong `jobs/`, TTL 1h, xoá sau khi client tải xong, quét dọn lúc khởi động. Restart mất job đang chạy — chấp nhận, vì job đang chạy có DB cũng không cứu được.

### Code layout

Thư mục mới `server/` ở gốc repo, được track thật. Ba cây upstream giữ nguyên, `server/` import vào. Xoá `VoxCPM/api/` và `LatentSync/api/` (code của mình đang nằm lẫn trong cây upstream nên không được track).

Cả ba upstream đều **vendor thẳng** vào repo (không submodule) — xem Step 1 để biết lý do.

### Patch upstream (có chủ đích, ghi chú lý do tại chỗ)

- **LatentSync**: tách phần dựng pipeline khỏi `scripts/inference.py:main()`. Hiện mỗi lần gọi dựng lại `Audio2Feature` + `AutoencoderKL` + `UNet3DConditionModel` rồi `.to("cuda")` — 30–60s thuần overhead mỗi job.
- **VSR**: thêm `box_thresh=0.80, thresh=0.45` vào `TextDetection(...)` trong `backend/tools/subtitle_detect.py`. Notebook `video_subtitle_remover.ipynb` vá bằng tay lúc runtime; bản vendor trong repo chưa có.

### API

```
POST   /dub                  → 202 {job_id}
GET    /jobs/{id}?since=N    → {status, step, log[], queue_position, error, error_code}
GET    /jobs/{id}/result     → mp4
DELETE /jobs/{id}            → huỷ
GET    /health               → {status, models_loaded, gpu}
```

- Auth: `Authorization: Bearer <API_KEY>` trên mọi endpoint.
- `status`: `queued | running | done | failed | cancelled`
- `error_code`: `no_face | invalid_input | internal` — server map `RuntimeError("Face not detected")` của LatentSync thành `no_face`.
- `since=N` trả về các dòng log từ chỉ số N, để client poll không kéo lại cả lịch sử.
- Không có `progress` %. `queue_position` là thứ user cần thật.
- Upload video ≤ 200MB, WAV giọng mẫu ≤ 25MB.
- Secrets qua biến môi trường: `API_KEY`, `OPENAI_API_KEY`.

### Tham số `POST /dub`

| Field | Ghi chú |
|---|---|
| `video` | file, bắt buộc |
| `reference_audio` | file WAV, tuỳ chọn — không có thì dùng vocals của chính video |
| `voice_mode` | `original` hoặc 1 trong 6 preset |
| `cfg_value`, `inference_timesteps` | VoxCPM |
| `target_lang` | ngôn ngữ đích |
| `whisper_model` | server chốt danh sách cho phép; tự retry `large-v3` khi ASR kém |
| `remove_subtitle` | bool |
| `vsr_mode` | `sttn-det` (mặc định) / `sttn-auto` / `lama` / `propainter` |
| `vsr_area` | 4 số % — mặc định y 60–96%, x 3–97% |
| `lipsync` | bool |
| `latentsync_steps`, `latentsync_guidance` | mặc định 20 / 1.5, `enable_deepcache=True` |
| `burn_subtitle` | bool |
| `subtitle_style` | font / cỡ / vị trí |

### Các bước trong job `/dub`

```
xoá sub (VSR, nếu bật)
  → demucs tách vocals / nhạc nền
  → whisper ASR
  → OpenAI rewrite + dịch
  → VoxCPM từng cue (in-process, giữ vòng ratio/rewrite hiện có)
  → LatentSync (nếu bật; không thấy mặt thì bỏ qua, không tính lỗi)
  → mix nhạc nền
  → mux
  → burn sub tiếng đích (nếu bật)
  → mp4
```

Huỷ: cờ hợp tác, kiểm ở ranh giới từng bước và mỗi vòng cue. Job `queued` huỷ tức thì; job `running` dừng ở điểm kiểm gần nhất (đang giữa một lần `run_inference` LatentSync thì phải chờ hết bước đó).

### Client

Xoá: tab ElevenLabs, `elevenlabs_api.py`, `elevenlabs_dialog.py`, `wav2lip_api.py`, tab Subtitle riêng, `openai_translate_api.py`, dependency demucs + faster-whisper + wav2lip, và process con ASR. Phần canh/ghép giọng trong `voxcpm_api.py` (`fit_tempo`, `match_tempo`, `place_segments`, `asr_needs_review`) chuyển sang server.

Lưu ý: **`torch` vẫn phải ở lại .exe** — `classifier_*.py` và CLIP dùng. Chỉ gỡ được demucs, faster-whisper, và cụm wav2lip.

`voxcpm_api.py` thành client job thuần: submit → poll → download.

UI: một panel, một nút chạy, ba checkbox (**Xoá sub · Lipsync · Tạo sub**). Mọi lựa chọn kỹ thuật là dropdown có nhãn cụ thể theo pattern `CFG_CHOICES` đang có (vd `"Cân bằng — khuyến nghị (20)"`). Bỏ khỏi UI: Base URL, OpenAI key, cụm weight/detector/height Wav2Lip.

`base_url` + `api_key` vào `global-configs.yaml` dạng `ENC(...)`, dùng lại cơ chế giải mã AES sẵn có trong `utils.py`.

Nhiều video → nộp hết job một lượt để queue server xếp hàng, poll 2 giây/lần, log server append thẳng vào ô log.

### Deploy

Một image Docker, hai venv bên trong:
- `/opt/venv-main` — VoxCPM + LatentSync + faster-whisper + demucs + FastAPI (torch 2.5.1+cu121, numpy 1.26.4)
- `/opt/venv-vsr` — VSR (py3.12, torch 2.7.0+cu118, paddlepaddle-gpu 3.0.0, numpy 2.2.5, onnxruntime-gpu 1.20.1, `paddlex --install hpi-gpu`)

Kèm docs dựng bare-metal hai venv + systemd.

## 5. Quyết định về test

**Đúng một seam mới:**

```
run_dub(ctx) -> Path
# ctx.params · ctx.workdir · ctx.log(msg) · ctx.step(name) · ctx.check_cancel()
```

Thuần Python, toàn bộ 8 bước GPU nằm sau nó. `JobRunner` nhận nó dạng inject.

| Tầng | Cách test | Cần GPU |
|---|---|---|
| HTTP + queue + job state | `TestClient`, `run_dub` giả (sleep / raise / phun log): auth 401, submit 202, poll `since`, `queue_position`, huỷ khi `queued`, huỷ khi `running`, map `no_face`, TTL, chặn >200MB | Không |
| Pipeline thật | Gọi thẳng `run_dub` với 1 clip ngắn — smoke test | Có |
| Client | `_selfcheck()` như prior art hiện có, thêm fake HTTP cho vòng submit→poll→download | Không |

Test chỉ chạm hành vi bên ngoài: status code, JSON trả về, file ra có tồn tại và khác rỗng. Không assert vào cấu trúc job state nội bộ.

Không cắt seam thấp hơn thành từng bước (`remove_subtitle`, `separate`, `transcribe`, ...) — mỗi bước là một model thật, test chúng là test upstream chứ không phải test code mình.

Prior art: `_selfcheck()` dùng `assert` trong `voxcpm_api.py`, `wav2lip_api.py`, `subtitle_api.py`, chạy bằng `python <file>.py`. Không có test framework ở spy-ads; giữ nguyên phong cách đó cho client, dùng `pytest` + `TestClient` cho `server/`.

## 6. Ngoài phạm vi

- Cho user sửa transcript / nghe thử từng cue trước khi ghép. Nếu sau này cần thì phải chẻ `/dub` thành nhiều job có `session_id` — thiết kế hiện tại cố tình không làm.
- Đa GPU, nhiều worker song song. Một GPU, một worker.
- Job sống sót qua restart server.
- Fallback offline khi mất mạng — bỏ Wav2Lip local nghĩa là không có lipsync offline nữa.
- Chạy VSR không kèm dub (nó là bước trong `/dub`, không phải job độc lập).
- Tối ưu upload (resume, chunk, nén).
- Multi-tenant, quota, rate limit theo user.

## 7. Ghi chú

- **VRAM 48GB** là tiền đề của gần như mọi quyết định preload. GPU nhỏ hơn thì phải xem lại phần preload.
- **numpy 1.26 vs 2.2** là lý do duy nhất VSR không nằm chung process. Nếu upstream VSR nới pin xuống 1.26 thì gộp được và bỏ luôn venv thứ hai.
- Notebook `video_subtitle_remover.ipynb` là nguồn sự thật cho cách gọi VSR: CLI `backend/main.py --input --output --subtitle-area-coords ymin ymax xmin xmax --inpaint-mode sttn-det`, chứ không phải `SubtitleRemover(...).run()` in-process.
- Tab ElevenLabs và Subtitle **đã bị comment sẵn** trong `ui.py:177,181` — chỉ Downloader + VoxCPM đang bật. Việc xoá chỉ là dọn xác.
- Ba notebook (`LatentSync_1_6.ipynb`, `voxcpm.ipynb`, `video_subtitle_remover.ipynb`) cố ý **không** track. Chúng là nguồn sự thật cho cách gọi model — ai đụng vào giai đoạn C nên đọc trước, nhưng phải xin file riêng.
- Toàn bộ công việc nằm trên branch `dub-server`, không phải `main`.

---

# Các bước triển khai

Mỗi step là một commit chạy được. Không step nào để repo ở trạng thái gãy.

Ký hiệu: 🆕 thêm file · ✏️ sửa file · 🗑️ xoá file

---

## Giai đoạn A — Dọn nền

### Step 1 — Vendor upstream, gỡ gitlink mồ côi ✅ `e05ee70`

Kế hoạch ban đầu là dựng `.gitmodules`. Không làm được: `VoxCPM/` và `LatentSync/` **không phải git repo** — không có `.git` bên trong, chỉ còn sót entry gitlink `160000` từ lần `git add` cũ. Hậu quả: ai clone repo về nhận **hai thư mục rỗng**.

Chuyển sang vendor thẳng, vì Step 6 đằng nào cũng phải patch `LatentSync/scripts/inference.py` (submodule không cho sửa tại chỗ), `video-subtitle-remover/` đã vendor sẵn, và tổng chỉ 17MB.

- [x] Branch `dub-server`
- [x] `git rm --cached VoxCPM LatentSync` — gỡ entry gitlink, không đụng file trên đĩa
- [x] `git add VoxCPM LatentSync` — 215 file source thật
- [x] 🆕 `.gitattributes` — `* text=auto eol=lf`. Source vendor có shell script chạy trong Docker/Linux; CRLF trên Windows làm hỏng shebang ở giai đoạn D
- [x] ✏️ `.gitignore` — `video_subtitle_remover.ipynb/` có dấu `/` thừa nên không match file nào; bỏ dấu `/`, gom về mục Notebooks
- [x] Kiểm không có `.env` / `.pt` / checkpoint nào lọt vào
- [x] **Xong khi**: clone thử ra thư mục sạch → `VoxCPM/src`, `LatentSync/scripts/inference.py`, `LatentSync/configs` đều có mặt ✓

Lệch so với plan gốc: **không track lại notebook** — giữ nguyên ignore theo ý chủ repo.

### Step 2 — Dựng khung `server/` ✅ `f713df5`

- [x] 🆕 `server/__init__.py`
- [x] 🆕 `server/config.py` — chỉ `API_KEY`, `JOBS_DIR`, `JOB_TTL_SECONDS`, `MAX_VIDEO_BYTES`, `MAX_AUDIO_BYTES`. `OPENAI_API_KEY` / `VSR_PYTHON` / `VSR_REPO` để bước 7-8 thêm khi có người dùng thật
- [x] 🆕 `server/app.py` — chỉ `GET /health`; `lifespan` chết ngay nếu thiếu `API_KEY`; tắt `docs_url` / `redoc_url` / `openapi_url`
- [x] 🆕 `server/requirements.txt` — fastapi, uvicorn, python-multipart + khối `[DEV]` pytest, httpx. **Chưa** cài torch/VoxCPM/LatentSync để bước 3-5 test được trên máy không GPU
- [x] 🆕 `server/tests/test_health.py` — 3 ca
- [x] 🆕 `server/README.md`
- [x] 🗑️ `VoxCPM/api/`, `LatentSync/api/`, `LatentSync/requirements-api.txt`, `LatentSync/test_api.sh`. Giữ `LatentSync/test_cli.sh` (test CLI upstream, vẫn dùng được)
- [x] ✏️ `.gitignore` — thêm `__pycache__/`, `*.py[cod]`, `.venv/`, `.pytest_cache/`, `jobs/`. Repo chưa có mục nào cho Python nên `.pyc` suýt bị commit
- [x] **Xong khi**: `pytest server/tests` 3 passed, và `uvicorn server.app:app` thật trả `{"status":"ok","models_loaded":[],"gpu":null}` ✓

Bỏ so với plan gốc: `server/tests/__init__.py` và `conftest.py` — pytest không cần, và chưa có fixture nào để chứa. Thêm khi bước 3 có `run_dub` giả.

Nợ nhỏ: `starlette.testclient` cảnh báo `httpx` sắp bị thay bằng `httpx2`. Chưa đổi vì chưa gãy.

---

## Giai đoạn B — Hạ tầng job (không cần GPU)

### Step 3 — Job store + queue + worker ✅ `1ac3334`

- [x] 🆕 `server/jobs.py` — `Job` dataclass, `JobRunner` (store + `queue.Queue` + một worker thread), `JobContext`, `JobCancelled`, `PipelineError`
  - [x] `submit()` / `get()` / `cancel()` / `queue_position()` / `snapshot(since)` / `drop()` / `purge_expired()`
  - [x] TTL: gọi khi `submit()` và sau mỗi job, **không** dùng thread hẹn giờ. Quét sạch `JOBS_DIR` lúc `start()`
  - [x] Xoá file ngay khi `drop()` (client tải xong)
- [x] 🆕 `server/tests/conftest.py` — fixture `make_runner` + helper `wait_until`
- [x] 🆕 `server/tests/fake_pipeline.py` — `FakePipeline` ghi lại thời điểm chạy để phát hiện chồng lấn, `failing_pipeline`, `crashing_pipeline`
- [x] 🆕 `server/tests/test_jobs.py` — 12 ca
- [x] **Xong khi**: `pytest server/tests` 15 passed, không cần GPU ✓

**Đổi seam so với plan gốc.** `run_dub(params, workdir, on_log, should_cancel)` → **`run_dub(ctx) -> Path`**, với `ctx.workdir` / `ctx.params` / `ctx.log()` / `ctx.step()` / `ctx.check_cancel()`. Lý do: bốn callback rời đã thiếu chỗ báo `step`, và mỗi nhu cầu mới lại phải nới chữ ký. `check_cancel()` **ném** `JobCancelled` thay vì trả bool, nên pipeline chỉ cần rắc một dòng giữa các bước.

**Bug bắt được nhờ test**: `queue_position` ban đầu sắp thứ tự bằng `created_at`. `time.time()` trên Windows chỉ nhích mỗi ~15.6ms, nên ba job nộp liên tiếp trùng timestamp và không cái nào đứng trước cái nào. Sửa bằng cách duyệt theo thứ tự chèn của `dict` — hàng đợi vốn đã có thứ tự, không cần suy ra từ đồng hồ.

Kiểm bằng mutation: cho worker chạy song song (mỗi job một thread) → 5 test đỏ. Test có răng thật.

### Step 4 — Endpoint + auth ✅ `7cb8d64`

- [x] 🆕 `server/schemas.py` — `DubParams` (17 field, whitelist bằng `Literal`, `ge/le` cho số, validator vùng quét phải là hộp thật) + `DubRequest` thêm hai file
- [x] 🆕 `server/auth.py` — `HTTPBearer` + `secrets.compare_digest`
- [x] 🆕 `server/uploads.py` — đếm dung lượng **trong lúc đọc** chứ không tin `Content-Length`, whitelist đuôi file, xoá file dở khi lỗi
- [x] ✏️ `server/app.py` — `create_app(run_dub)`; `POST /dub`, `GET /jobs/{id}?since=N`, `GET /jobs/{id}/result`, `DELETE /jobs/{id}`, `GET /health`
- [x] ✏️ `server/jobs.py` — tách `submit()` thành `create()` + `enqueue()`
- [x] 🆕 `server/tests/test_api.py` — 21 ca
- [x] 🗑️ `server/tests/test_health.py` — `/health` giờ cần token, `test_api.py` đã phủ
- [x] **Xong khi**: `pytest server/tests` 34 passed; và uvicorn thật: không token → 401, `POST /dub` → 202 + job_id, poll ra `error_code: internal` (pipeline chưa dựng), `cfg_value=99` → 422 ✓

**Hai chỗ phải chữa mới chạy được:**

1. **Upload đua với worker.** `submit()` cũ vừa tạo job vừa xếp hàng trong một lệnh, nên worker có thể nhấc job lên trong lúc video còn đang upload. Tách thành `create()` → ghi file → `enqueue()`.

2. **Form model bị lồng.** FastAPI chỉ trải phẳng form model khi nó là body param **duy nhất**. Có `UploadFile` đứng cạnh thì mọi tham số chui vào field `params` và request nào cũng 422. Chuyển file vào trong `DubRequest`; `settings()` trả `DubParams` sạch để pipeline không cầm file handle đang mở.

Placeholder `not_built_yet` ném `PipelineError("The dub pipeline is not built yet")` — app import được và chạy được trước khi có bước 8.

### Step 5 — Huỷ hợp tác ✅ `e145818`

Cơ chế đã dựng sẵn ở bước 3 (`cancel()` bật cờ, `ctx.check_cancel()` ném `JobCancelled`, `_run_one` bắt) và bước 4 phơi ra `DELETE`. Bước này bổ sung thứ còn thiếu: **bằng chứng job đang chạy dừng thật**.

- [x] ✏️ `server/tests/fake_pipeline.py` — `steps_done` đếm số bước đã xong, `started` là `threading.Event` để test huỷ đúng lúc pipeline đã thật sự vào chạy (huỷ sớm quá thì chỉ chứng minh lại ca `queued`)
- [x] 🆕 `server/tests/test_cancel.py` — 9 ca, gom cả mức `JobRunner` lẫn mức HTTP
  - job đang chờ: không bao giờ chạm pipeline
  - job đang chạy: dừng giữa chừng (`steps_done < 10`), không sinh kết quả, không để lại file
  - huỷ rồi worker vẫn chạy tiếp job sau
  - huỷ job đã xong / job lạ / huỷ hai lần → từ chối
- [x] ✏️ Dời hết test huỷ khỏi `test_jobs.py` và `test_api.py` về một chỗ
- [x] **Xong khi**: `pytest server/tests` 39 passed ✓

Mutation check: cho `check_cancel()` không ném gì → đúng hai ca "dừng giữa chừng" đỏ. Bảy ca kia vẫn xanh, đúng như mong đợi — chúng kiểm nhánh dự phòng trong `_run_one` (pipeline phớt lờ cờ mà chạy hết thì vẫn bị đánh dấu `cancelled` và xoá file).

**Giới hạn đã biết**: huỷ chỉ ăn ở ranh giới bước. Bước 7 (VSR chạy `subprocess`) và bước 8 (vòng TTS từng cue) phải tự thêm điểm kiểm bên trong, nếu không một lượt LatentSync dài vẫn phải chờ hết.

Hết giai đoạn B: toàn bộ logic mới đã có test, chưa đụng GPU dòng nào.

---

## Giai đoạn C — Nối model (cần GPU)

### Step 6 — Patch LatentSync để preload ✅ `170bd24` (chưa chạy trên GPU)

- [x] ✏️ `LatentSync/scripts/inference.py` — `build_pipeline(config, ckpt_path, enable_deepcache) -> (pipeline, dtype)`; `main()` gọi lại nó nên CLI cũ không đổi hành vi. Trả cả `dtype` vì lúc chạy phải truyền `weight_dtype` đúng giá trị đó
- [x] 🆕 `server/steps/lipsync.py` — `LipsyncModel` giữ pipeline giữa các job; `NoFaceError` (mã `no_face`)
- [x] ✏️ `server/config.py` — `LATENTSYNC_DIR` / `LATENTSYNC_CONFIG` / `LATENTSYNC_CHECKPOINT`
- [x] ✏️ `server/app.py` — `create_app(run_dub, models=())`, nạp trước request đầu tiên, `/health` báo tên model + tên GPU
- [x] 🆕 `server/tests/test_models.py` — 5 ca, dùng `FakeModel`, không cần GPU
- [x] `pytest server/tests` 44 passed; `import server.app` chạy được trên máy **không có torch**
- [ ] **Xong khi**: gọi lipsync hai lần liên tiếp trên GPU, lần thứ hai không còn 30–60s load ← *chưa kiểm được, cần máy GPU + checkpoint*

**Cạm bẫy phải xử lý: đường dẫn tương đối.** `build_pipeline` đọc `"configs"`, `"checkpoints/whisper/small.pt"`, và config trỏ `latentsync/utils/mask.png` — tất cả tính từ gốc repo LatentSync. Nên cả `load()` lẫn `run()` đều bọc trong `chdir(repo_dir)`. `chdir` là toàn process, chỉ an toàn vì đúng một worker chạy một job — đã ghi `ponytail:` tại chỗ, có worker thứ hai thì phải đổi sang truyền đường dẫn tuyệt đối vào upstream.

**Giữ được tính chất không-cần-GPU**: `lipsync.py` không import torch/latentsync ở mức module, `_gpu_name()` bọc `import torch` trong `try/except ImportError`. 44 test vẫn chạy trên laptop.

**Bug thứ tự do test mới bắt được**: job bị huỷ được đánh dấu `cancelled` **trước khi** xoá file, nên client poll thấy `cancelled` mà thư mục còn nằm đó. Gộp thành `_cancel_and_clean()` — xoá file trước, đổi trạng thái sau.

### Step 7 — Venv VSR + wrapper subprocess ✅ `cb7a0e7` (chưa chạy trên GPU)

- [x] ✏️ `video-subtitle-remover/backend/tools/subtitle_detect.py` — `box_thresh=0.80, thresh=0.45`. Bản vendor thiếu, notebook vá tay lúc runtime. Không có nó thì detector quét cả logo và chữ trong artwork
- [x] 🆕 `server/requirements-vsr.txt` — giữ **thứ tự cài và 3 index URL riêng**; pip chỉ nhận một `--index-url` nên không diễn tả được bằng requirements thường, phải ghi thành lệnh trong comment
- [x] 🆕 `server/steps/vsr.py` — `probe_size()` (ffprobe), `area_to_pixels()`, `build_command()`, `remove_subtitles()`
- [x] ✏️ `server/config.py` — `VSR_DIR`, `VSR_PYTHON`, `FFPROBE_BIN`, `FFMPEG_BIN`
- [x] 🆕 `server/tests/test_vsr.py` — 8 ca, không cần paddle
- [x] `pytest server/tests` 54 passed
- [ ] **Xong khi**: xoá sub một clip thật, kết quả khớp notebook ← *chưa kiểm được, cần venv VSR + GPU*

**Client gửi vùng quét theo % chứ không theo pixel** — nó không biết kích thước video, và cùng một cấu hình phải đúng cho cả 720p lẫn 1080p. Server `ffprobe` lấy kích thước rồi quy đổi.

**Log bị tiết chế còn 1 dòng / 2 giây.** VSR vẽ thanh tqdm; Python dịch `\r` thành `\n` nên mỗi lần vẽ lại thành một dòng riêng — không chặn thì log job phình hàng nghìn dòng gần giống nhau.

**Huỷ**: kiểm cờ mỗi lần đọc được một dòng stdout rồi `process.kill()`. VSR in đủ dày nên không cần timer riêng.

**Test nhắm hai chỗ sai mà không ai báo**: phép quy đổi %, và **thứ tự toạ độ** trên dòng lệnh (`YMIN YMAX XMIN XMAX`). Sai thứ tự thì nó bôi nhầm chỗ trên khung hình mà không có lỗi nào bắn ra.

Một test viết sai đã bị chính nó bắt: tôi khẳng định video gấp đôi thì toạ độ gấp đôi chính xác, nhưng `int()` cắt thập phân nên lệch 1 pixel (`0.96×360=345.6→345` vs `0.96×720=691.2→691`). Sửa test thành "sai lệch ≤ 1 pixel", không sửa code.

### Step 8 — `run_dub` thật ✅ `39d8527` (chưa chạy trên GPU)

- [x] 🆕 `server/steps/audio.py` (213) — ffmpeg + `tempo_for` / `match_tempo` / `place_clips`
- [x] 🆕 `server/steps/separate.py` (43) — demucs, `device="cuda"` thay vì ép CPU
- [x] 🆕 `server/steps/transcribe.py` (199) — faster-whisper in-process, `asr_quality`, retry `large-v3`, `cue_needs_review`; giữ nhiều model trong `dict`
- [x] 🆕 `server/steps/translate.py` (1277) — **copy nguyên** `openai_translate_api.py`, chỉ đổi lớp lỗi và nguồn key
- [x] 🆕 `server/steps/synth.py` (369) — `VoxCPMModel`, `cue_slots`, `fit_cue`, `timed_speech`
- [x] 🆕 `server/pipeline.py` (195) — `run_dub(ctx)`, 9 bước, `check_cancel` ở mỗi ranh giới và mỗi vòng cue, bắt `NoFaceError` → bỏ qua lipsync
- [x] ✏️ `server/app.py` — `build_production_app()`; `LOAD_MODELS=0` để chạy được ở máy không GPU
- [x] 🆕 `server/tests/smoke_run_dub.py` (120) — chạy tay, không phải file pytest
- [x] 🆕 `server/requirements-models.txt` — tách riêng phần GPU, giữ `requirements.txt` chạy được trên laptop
- [x] 🆕 `server/tests/test_synth.py` (267) — 21 ca, thay TTS bằng tiếng sine ffmpeg nên cả cây quyết định chạy thật
- [x] `pytest server/tests` 101 passed; `pyflakes` sạch
- [ ] **Xong khi**: smoke test 1 clip ngắn, đủ 9 bước, ra mp4 xem được ← *cần GPU*

**Đối chiếu với bản client**: mọi ngưỡng, mọi nhánh, mọi công thức của `_fit_cue_audio` và `_timed_speech` khớp 1-1. Chỗ duy nhất khác cấu trúc là gộp hai nhánh "hơi dài" / "hơi ngắn" thành một — tương đương vì nhánh `KEEP` đứng trước đã `return`, và cả hai nhánh client đều gọi cùng một `apply_tempo`. Bỏ `_blob_speech` (chỉ `raise`, đã chết từ trước).

**Hai test tôi viết sai và phải sửa**, cả hai đều lộ ra hành vi thật chưa ai ghi:
1. Hai tầng đổi tốc độ **nhân dồn**: 1.15 × 1.25 ≈ **1.44×**, vượt xa `SOFT_SPEEDUP`.
2. Ngay cả 1.44 vẫn có thể không đủ — code **chấp nhận tràn** thay vì cắt mất chữ.

**Ràng buộc ngầm đã ghi comment**: LatentSync quyết định ghi bao nhiêu khung hình theo **độ dài audio** (`num_inferences = ceil(len(whisper_chunks) / num_frames)`). `speech` ngắn hơn video là nó âm thầm cắt cụt đuôi hình. `place_clips` pad đúng `video_seconds` nên hiện không sao — nhưng liên kết đó không nhìn thấy từ chỗ nào cả. *(Đúng cái bẫy `voxcpm_api.py` cũ từng ghi chú cho Wav2Lip.)*

### Vá kèm — chặn upload quá cỡ trước khi chạm đĩa 🆕 `server/limits.py`

Phát hiện khi trả lời câu hỏi "gửi video 4GB thì sao". Đo trên server thật:

| | Trước | Sau |
|---|---|---|
| Byte gửi lên | 62.914.772 | **0** |
| TEMP tăng | +200 MB | **0 MB** |

`MAX_VIDEO_BYTES` ở bước 4 kiểm **trong lúc `save_upload` đọc** — quá muộn. Starlette đọc và ghi trọn body ra file tạm **trước cả tầng auth**, nên request **không token** cũng ghi được bao nhiêu tuỳ thích vào đĩa. Ai chạm tới cổng đều làm đầy đĩa được.

`BodySizeLimit` bọc ngoài toàn app, hai lớp: đọc `Content-Length` (từ chối trước khi chạm byte nào) **và** đếm luồng khi chảy (vì header đó do client viết, có thể thiếu hoặc nói dối). 7 test, ca quan trọng nhất là `test_the_limit_applies_without_a_token`.

### Step 9 — Burn sub ✅ `e066913` (làm trước step 8; còn thiếu dòng nối vào pipeline)

- [x] 🆕 `server/steps/subtitle.py` — `wrap_text_lines`, `split_cue`, `normalize_cues`, `_ass_time`, `write_ass`, `burn`
- [x] 🆕 `server/tests/test_subtitle.py` — 19 ca, trong đó 1 ca chạy ffmpeg thật
- [x] `server/schemas.py` đã có `subtitle_font` / `subtitle_size` / `subtitle_position` từ bước 4
- [x] **Xong khi**: burn thật một clip 1280×720 → chữ hiện đúng vị trí, đúng ngắt dòng, audio giữ nguyên codec, thời lượng không đổi ✓ (đã trích khung hình xem tận mắt)
- [x] ✏️ `server/pipeline.py` — chèn bước burn sau mux ✅ *(làm ở bước 8, `39d8527`)*. Chữ lấy từ `dub_script.json` mà bước dịch đã ghi, nên phụ đề nói đúng những gì giọng nói

**Hai thay đổi so với bản client:**

1. **Làm việc trên cue trong bộ nhớ**, không qua file `.srt`. Bản desktop phải ghi ra đĩa rồi đọc lại vì các bước là những chương trình riêng; ở server thì cue đi thẳng từ bước dịch sang.
2. **Chỉ giữ tham số client gửi được** — font, cỡ, vị trí. Màu chữ, viền, nền bị chôn cứng: code cũ có mang chúng nhưng chưa dropdown nào set.

**Đặt chữ bằng ASS `\pos` chứ không dùng margin của ffmpeg.** Một tỉ lệ chiều cao ("85% từ trên xuống") rơi đúng chỗ trên cả 720p lẫn 1080p; margin tính bằng pixel thì không.

**Test nhắm chỗ sai mà không ai báo**: độ dài dòng · cắt cue phải lấp kín đúng khoảng thời gian cũ, không hở · định dạng centisecond · và **hai cách một giá trị phá hỏng file ASS** — xuống dòng thật làm đứt dòng `Dialogue`, dấu phẩy trong tên font làm lệch mọi trường của `Style`.

**Một ca chạy ffmpeg thật** (bỏ qua nếu máy không có). Đường dẫn `.ass` được nhét vào chuỗi `-vf`, nơi dấu `:` phân tách tuỳ chọn và `'` kết thúc giá trị — escape sai thì filter đứt đôi, chỉ chạy thật mới lộ.

---

## Giai đoạn D — Deploy

### Step 10 — Docker ✅ `164401d` (chưa build trên máy GPU)

- [x] 🆕 `server/Dockerfile` — base `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`, hai venv (py3.10 chính, py3.12 cho VSR từ deadsnakes), ffmpeg + libgl1 + libglib2.0-0, `paddlex --install hpi-gpu`
- [x] 🆕 `docker-compose.yml` — `--gpus all`, volume cho weight/cache/jobs, healthcheck có token, `start_period: 300s` vì nạp model lâu
- [x] 🆕 `.env.example`
- [x] 🆕 `.dockerignore` — loại weight, notebook, spy-ads khỏi build context
- [x] ✏️ `.gitignore` — thêm `!.env.example`; luật `.env.*` đang nuốt luôn cái template không thể rò rỉ gì
- [x] 🗑️ `VoxCPM/docker-compose*.yml`, `LatentSync/docker-compose.yml`, hai `.env.example` của API cũ — chúng khai key không còn tồn tại
- [x] `docker compose config` hợp lệ; `docker build --check` không cảnh báo
- [ ] **Xong khi**: `docker compose up` trên máy sạch → `/health` xanh, chạy được một job thật ← *cần máy GPU*

**Sửa một tag sai của upstream**: `LatentSync/Dockerfile` dùng `12.1.1-cudnn-runtime-ubuntu22.04` — **không tồn tại**. NVIDIA chỉ bỏ số `8` sau `cudnn` từ CUDA 12.3 trở đi. Đã hỏi registry để xác minh, không đoán.

### Step 11 — Docs bare-metal ✅ `164401d`

- [x] 🆕 `server/docs/bare-metal.md` — hai venv theo đúng thứ tự của notebook, tải checkpoint, systemd unit, nginx, mục gỡ lỗi
- [x] ✏️ `server/README.md` — viết lại: production / dev không GPU / test / API / biến môi trường
- [ ] **Xong khi**: làm theo docs trên máy sạch, ra kết quả như Docker ← *cần máy GPU*

Docs mang theo hai phép kiểm đáng giá nhất: **numpy phải ra 1.26.4 ở venv chính và 2.2.5 ở venv VSR** (sai là hỏng lúc chạy chứ không phải lúc cài), và **`--workers 1` là bắt buộc** — code chặn được `start()` hai lần trong một process, nhưng không chặn được hai process.

---

## Giai đoạn E — Client (`spy-ads-creative-desktop-tool/`)

### Step 12 — `voxcpm_api.py` thành client job ✅ (chưa gọi server thật)

- [x] ✏️ `global-configs.yaml` — thêm khối `dub: { base_url: "ENC(...)", api_key: "ENC(...)" }`, sinh bằng `encrypt_keys.py`
- [x] ✏️ `encrypt_configs.py` — thêm `base_url` và `api_key` của khối `dub` vào danh sách field cần giải mã
- [x] ✏️ `main.py` — thêm hai key mới vào danh sách decrypt (chỗ đang liệt kê `telemetry.api_key`, `remake.*`)
- [x] ✏️ `voxcpm_api.py` — viết lại thành client job:
  - [x] `submit_dub(...) -> job_id` (multipart), `poll(job_id, since) -> dict`, `download(job_id, out_path)`, `cancel(job_id)`
  - [x] Header `Authorization: Bearer`, giữ `_force_ipv4()`, giữ `_explain_http()`
  - [x] 🗑️ bỏ `fit_tempo`, `fit_audio`, `match_tempo`, `stretch_if_long`, `place_segments`, `asr_needs_review`, `with_voice_instruction`, `plain_text_from_srt`, `fit_to_video`, `tts`, `clone` (đã sang server)
  - [x] Giữ `VOICE_MODE_LABELS`, `CFG_CHOICES`, `TIMESTEP_CHOICES` (UI cần) + thêm `LATENTSYNC_STEP_CHOICES`, `GUIDANCE_CHOICES`, `VSR_MODE_CHOICES`, `VSR_AREA_CHOICES`
  - [x] ✏️ `_selfcheck()` — fake HTTP cho vòng submit → poll → download
- [ ] **Xong khi**: `python voxcpm_api.py` xanh, và gọi được server thật từ script rời

`dub.base_url` / `dub.api_key` để **rỗng** trong YAML: chưa có server thật để lấy URL và key. Điền plaintext rồi chạy `encrypt_configs.py` (đã nhận hai leaf-key này) là xong — rỗng thì `DubClient` báo "chưa cấu hình" chứ không sập app.

`voxcpm_panel.py` **gãy import** cho tới hết step 13, vì nó còn `from voxcpm_api import tts, clone, match_tempo, ...` — đúng những thứ step 12 xoá. Step 13 viết lại panel nên không vá tạm ở đây.

### Step 13 — Panel mới ✅ (chưa chạy trên UI thật)

- [x] ✏️ `voxcpm_panel.py` — viết lại:
  - [x] 🗑️ `DubWorker` cũ (demucs / whisper / synth / lipsync local) → worker mới chỉ upload + poll + download
  - [x] Nộp cả lô job một lượt, poll 2s/lần, log server append vào ô log
  - [x] Stop → `DELETE` mọi job của lô
  - [x] Ba checkbox: Xoá sub · Lipsync · Tạo sub
  - [x] Dropdown mới: LatentSync steps, guidance, VSR mode, VSR area (+ 4 ô % khi chọn *Tuỳ chỉnh*), style sub
  - [x] 🗑️ ô Base URL, ô OpenAI key, cụm chọn thư mục weight / model / detector / out_height
  - [x] ✏️ QSettings: bỏ `voxcpm/base_url`, `voxcpm/openai_key`, `voxcpm/wav2lip_*`; thêm key cho các dropdown mới
- [x] ✏️ `ui.py` — đổi nhãn tab `"VoxCPM"` → `"Dubbing"`
- [ ] **Xong khi**: chạy lô 3 video qua UI, log hiện đúng, Stop huỷ được

`ui.py` phải giữ luôn `global_configs` (`self.global_configs`) để truyền `configs.dub` vào panel — panel không còn ô Base URL nên phải lấy từ config.

QSettings của tab mới nằm dưới tiền tố `dub/`, không phải `voxcpm/`: key cũ trùng tên nhưng khác kiểu (`target_lang` là chuỗi, `cfg_value` là float) làm `settings.value(..., type=int)` ném ngay lúc dựng tab. `_saved_index()` còn ép int trong try/except cho chắc.

`_ask()` bỏ `ask_yes_no` của `elevenlabs_dialog`, dùng `QMessageBox.question` — step 14 xoá file đó.

`_selfcheck()` trong `voxcpm_panel.py` chạy `DubWorker` với client giả: nộp cả lô, log kèm tên video, `queue_position`, job `no_face` không chặn job khác, Stop huỷ mọi job còn dở. Máy này không có PyQt5 nên self-check chạy qua stub Qt; trên máy build thì `python voxcpm_panel.py` là đủ.

### Step 14 — Dọn code ✅ (chưa build .exe)

- [x] 🗑️ `elevenlabs_api.py`
- [x] 🗑️ `elevenlabs_dialog.py`
- [x] 🗑️ `wav2lip_api.py`
- [x] 🗑️ `openai_translate_api.py`
- [x] 🗑️ `subtitle_panel.py`
- [x] 🗑️ `subtitle_api.py`
- [x] 🗑️ `subtitle_worker.py`
- [x] ✏️ `ui.py` — xoá dòng import + hai `addTab` đã comment (`:177`, `:181`), xoá `self.elevenlabs_panel` / `self.subtitle_panel`
- [x] ✏️ `main.py:58` — xoá nhánh `from subtitle_api import _cli`
- [x] ✏️ `requirements.txt` — gỡ khối `[ELEVENLABS]` (demucs, julius, lameenc, sphn, einops…), khối `[SUBTITLE]` (faster-whisper), khối `[WAV2LIP]` (batch-face, opencv-transforms, sixdrepnet, `wav2lip-trip @ git+...`). **Giữ** torch/torchaudio/torchvision — `classifier_*.py` và CLIP còn dùng.
- [x] ✏️ `_build_exe.ps1` — bỏ hidden-import / data của các module đã xoá nếu có
- [x] ✏️ `README.md`, `GEMINI.md` — cập nhật mô tả kiến trúc
- [ ] **Xong khi**: build .exe thành công, nhẹ hơn hẳn, Downloader + Dubbing chạy đúng

`main.py` mất luôn cổng `--subtitle-cli` (process con ASR) — nó chỉ tồn tại để gọi `subtitle_api._cli`. `_build_exe.ps1` bỏ nguyên khối build `subtitle-asr.exe`, bỏ hidden-import `faster_whisper`/`demucs`/`wav2lip` và collect-data `faster_whisper`/`batch_face`/`wav2lip`/`demucs`.

**Giữ `--collect-data=whisper`**: `classifier_video.py` import `whisper` (openai-whisper, khác faster-whisper) để lấy transcript cho bước lọc policy. Gỡ nhầm là .exe chết khi quét audio.

---

**Thứ tự bắt buộc**: A → B → C → D → E. Giai đoạn B phải xong trước C, để khi nối GPU vào thì mọi thứ quanh nó đã được test rồi — lỗi lúc đó chắc chắn nằm ở model, không phải ở hạ tầng.
