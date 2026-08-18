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

### Step 4 — Endpoint + auth

- [ ] 🆕 `server/schemas.py` — pydantic model cho `DubParams` (đủ bảng tham số ở §4), whitelist `whisper_model` / `vsr_mode`, ràng buộc `ge/le` cho các số
- [ ] 🆕 `server/auth.py` — `HTTPBearer` + `secrets.compare_digest`, port từ `VoxCPM/api/main.py` cũ
- [ ] 🆕 `server/uploads.py` — `save_upload()` giới hạn dung lượng theo chunk + whitelist đuôi file, port từ `LatentSync/api/main.py` cũ
- [ ] ✏️ `server/app.py`
  - [ ] `POST /dub` → 202 `{job_id}`
  - [ ] `GET /jobs/{id}?since=N`
  - [ ] `GET /jobs/{id}/result` → `FileResponse`
  - [ ] `DELETE /jobs/{id}`
  - [ ] `GET /health` → `{status, models_loaded, gpu}`
  - [ ] `dependencies=[Depends(require_api_key)]`, tắt `docs_url` / `openapi_url`
- [ ] 🆕 `server/tests/test_api.py` — 401 không token, 202 submit, poll `since`, 413 quá cỡ, 415 sai đuôi, 404 job lạ
- [ ] **Xong khi**: `pytest server/tests/` xanh với `run_dub` giả

### Step 5 — Huỷ hợp tác

- [ ] ✏️ `server/jobs.py` — `should_cancel()` closure truyền vào `run_dub`; job `queued` bị huỷ thì worker bỏ qua không chạy
- [ ] 🆕 `server/tests/test_cancel.py` — huỷ job `queued` (tức thì), huỷ job `running` (dừng trong vòng một bước), huỷ job đã `done` (không đổi gì)
- [ ] **Xong khi**: cả ba ca xanh

Hết giai đoạn B: toàn bộ logic mới đã có test, chưa đụng GPU dòng nào.

---

## Giai đoạn C — Nối model (cần GPU)

### Step 6 — Patch LatentSync để preload

- [ ] ✏️ `LatentSync/scripts/inference.py` — tách phần dựng `Audio2Feature` + `AutoencoderKL` + `UNet3DConditionModel` + `LipsyncPipeline` ra hàm `build_pipeline(config, ckpt_path, enable_deepcache)`; `main()` gọi lại nó để CLI cũ không gãy. Ghi comment nêu lý do patch.
- [ ] 🆕 `server/steps/lipsync.py` — giữ pipeline đã preload, hàm `run(video, audio, out, steps, guidance, seed, on_log, should_cancel)`; bắt `RuntimeError("Face not detected")` → raise `NoFaceError` riêng
- [ ] ✏️ `server/app.py` — preload trong `lifespan`, `models_loaded` trong `/health`
- [ ] **Xong khi**: gọi lipsync hai lần liên tiếp, lần thứ hai không còn 30–60s load

### Step 7 — Venv VSR + wrapper subprocess

- [ ] ✏️ `video-subtitle-remover/backend/tools/subtitle_detect.py` — thêm `box_thresh=0.80, thresh=0.45` vào `TextDetection(...)`, comment nêu nguồn (notebook)
- [ ] 🆕 `server/requirements-vsr.txt` — torch 2.7.0+cu118, torchvision 0.22.0, paddlepaddle-gpu 3.0.0, onnxruntime-gpu 1.20.1 + `-r video-subtitle-remover/requirements.txt`
- [ ] 🆕 `server/steps/vsr.py` — `subprocess` gọi `$VSR_PYTHON backend/main.py --input --output --subtitle-area-coords ... --inpaint-mode ...`, đọc stdout đẩy vào `on_log`, kiểm `should_cancel` để `kill()`, env `MPLBACKEND=Agg` + `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`
- [ ] 🆕 `server/steps/vsr.py` — hàm quy đổi `vsr_area` từ % sang pixel theo kích thước video thật
- [ ] **Xong khi**: xoá sub một clip thật, kết quả khớp notebook

### Step 8 — `run_dub` thật

- [ ] 🆕 `server/steps/audio.py` — port `_run`, `duration`, `ffmpeg_bin`, `extract_audio`, `mix_audio`, `mux_audio`, `suppress_vocal_bleed` từ `spy-ads-creative-desktop-tool/elevenlabs_api.py`
- [ ] 🆕 `server/steps/separate.py` — port `separate_stems` (demucs), bỏ nhánh frozen/.exe, đổi `-d cpu` → GPU
- [ ] 🆕 `server/steps/transcribe.py` — faster-whisper in-process (không cần tách process vì không có Qt), port `asr_quality` + `load_asr_json` + `parse_srt` từ `subtitle_api.py`, giữ logic retry `large-v3`
- [ ] 🆕 `server/steps/translate.py` — port từ `openai_translate_api.py`, key lấy từ env
- [ ] 🆕 `server/steps/synth.py` — VoxCPM in-process; port `fit_tempo`, `fit_audio`, `match_tempo`, `stretch_if_long`, `place_segments`, `asr_needs_review`, `with_voice_instruction`, `VOICE_PRESETS` từ `voxcpm_api.py`; port vòng ratio/rewrite từ `voxcpm_panel.py:_timed_speech`
- [ ] 🆕 `server/pipeline.py` — `run_dub(ctx) -> Path`, ghép 8 bước, gọi `ctx.check_cancel()` ở ranh giới mỗi bước và mỗi vòng cue; bắt `NoFaceError` → log warn + bỏ qua lipsync (không fail)
- [ ] ✏️ `server/app.py` — thay `run_dub` giả bằng bản thật
- [ ] 🆕 `server/tests/smoke_run_dub.py` — clip ngắn thật, chạy tay, không nằm trong `pytest` mặc định
- [ ] ✏️ `server/requirements.txt` — thêm torch 2.5.1+cu121, numpy 1.26.4, faster-whisper, demucs, soundfile, openai, `-r VoxCPM/requirements.txt`, `-r LatentSync/requirements.txt`
- [ ] **Xong khi**: smoke test 1 clip ngắn, đủ 8 bước, ra mp4 xem được

### Step 9 — Burn sub

- [ ] 🆕 `server/steps/subtitle.py` — port `normalize_style`, `normalize_srt_file`, sinh ASS `\pos` theo % khung hình, ≤32 ký tự/dòng ≤2 dòng, burn bằng ffmpeg — từ `subtitle_api.py`
- [ ] ✏️ `server/pipeline.py` — chèn bước burn sau mux, dùng cue tiếng đích đã có sẵn
- [ ] ✏️ `server/schemas.py` — `subtitle_style` (font, cỡ, vị trí)
- [ ] **Xong khi**: bật `burn_subtitle`, mp4 ra có sub tiếng đích đúng style

---

## Giai đoạn D — Deploy

### Step 10 — Docker

- [ ] 🆕 `server/Dockerfile` — base CUDA, `/opt/venv-main` từ `requirements.txt`, `/opt/venv-vsr` từ `requirements-vsr.txt` + `paddlex --install hpi-gpu`, cài ffmpeg + libgl1 + libglib2.0-0
- [ ] 🆕 `docker-compose.yml` ở gốc — mount `jobs/`, mount cache model, `--gpus all`, `env_file: .env`
- [ ] 🆕 `.env.example` — `API_KEY=`, `OPENAI_API_KEY=`
- [ ] ✏️ `.gitignore` — thêm `.env`, `jobs/`
- [ ] 🗑️ `VoxCPM/docker-compose.yml`, `LatentSync/docker-compose.yml` nếu không còn dùng (hoặc để nguyên nếu là của upstream)
- [ ] **Xong khi**: `docker compose up` trên máy sạch → `/health` xanh, chạy được một job thật

### Step 11 — Docs bare-metal

- [ ] 🆕 `server/docs/bare-metal.md` — dựng hai venv theo đúng thứ tự cài của notebook, tải checkpoint, systemd unit mẫu
- [ ] ✏️ `server/README.md` — trỏ sang cả hai đường Docker và bare-metal
- [ ] **Xong khi**: làm theo docs trên máy sạch, ra kết quả như Docker

---

## Giai đoạn E — Client (`spy-ads-creative-desktop-tool/`)

### Step 12 — `voxcpm_api.py` thành client job

- [ ] ✏️ `global-configs.yaml` — thêm khối `dub: { base_url: "ENC(...)", api_key: "ENC(...)" }`, sinh bằng `encrypt_keys.py`
- [ ] ✏️ `encrypt_configs.py` — thêm `base_url` và `api_key` của khối `dub` vào danh sách field cần giải mã
- [ ] ✏️ `main.py` — thêm hai key mới vào danh sách decrypt (chỗ đang liệt kê `telemetry.api_key`, `remake.*`)
- [ ] ✏️ `voxcpm_api.py` — viết lại thành client job:
  - [ ] `submit_dub(...) -> job_id` (multipart), `poll(job_id, since) -> dict`, `download(job_id, out_path)`, `cancel(job_id)`
  - [ ] Header `Authorization: Bearer`, giữ `_force_ipv4()`, giữ `_explain_http()`
  - [ ] 🗑️ bỏ `fit_tempo`, `fit_audio`, `match_tempo`, `stretch_if_long`, `place_segments`, `asr_needs_review`, `with_voice_instruction`, `plain_text_from_srt`, `fit_to_video`, `tts`, `clone` (đã sang server)
  - [ ] Giữ `VOICE_MODE_LABELS`, `CFG_CHOICES`, `TIMESTEP_CHOICES` (UI cần) + thêm `LATENTSYNC_STEP_CHOICES`, `GUIDANCE_CHOICES`, `VSR_MODE_CHOICES`, `VSR_AREA_CHOICES`
  - [ ] ✏️ `_selfcheck()` — fake HTTP cho vòng submit → poll → download
- [ ] **Xong khi**: `python voxcpm_api.py` xanh, và gọi được server thật từ script rời

### Step 13 — Panel mới

- [ ] ✏️ `voxcpm_panel.py` — viết lại:
  - [ ] 🗑️ `DubWorker` cũ (demucs / whisper / synth / lipsync local) → worker mới chỉ upload + poll + download
  - [ ] Nộp cả lô job một lượt, poll 2s/lần, log server append vào ô log
  - [ ] Stop → `DELETE` mọi job của lô
  - [ ] Ba checkbox: Xoá sub · Lipsync · Tạo sub
  - [ ] Dropdown mới: LatentSync steps, guidance, VSR mode, VSR area (+ 4 ô % khi chọn *Tuỳ chỉnh*), style sub
  - [ ] 🗑️ ô Base URL, ô OpenAI key, cụm chọn thư mục weight / model / detector / out_height
  - [ ] ✏️ QSettings: bỏ `voxcpm/base_url`, `voxcpm/openai_key`, `voxcpm/wav2lip_*`; thêm key cho các dropdown mới
- [ ] ✏️ `ui.py` — đổi nhãn tab `"VoxCPM"` → `"Dubbing"`
- [ ] **Xong khi**: chạy lô 3 video qua UI, log hiện đúng, Stop huỷ được

### Step 14 — Dọn code

- [ ] 🗑️ `elevenlabs_api.py`
- [ ] 🗑️ `elevenlabs_dialog.py`
- [ ] 🗑️ `wav2lip_api.py`
- [ ] 🗑️ `openai_translate_api.py`
- [ ] 🗑️ `subtitle_panel.py`
- [ ] 🗑️ `subtitle_api.py`
- [ ] 🗑️ `subtitle_worker.py`
- [ ] ✏️ `ui.py` — xoá dòng import + hai `addTab` đã comment (`:177`, `:181`), xoá `self.elevenlabs_panel` / `self.subtitle_panel`
- [ ] ✏️ `main.py:58` — xoá nhánh `from subtitle_api import _cli`
- [ ] ✏️ `requirements.txt` — gỡ khối `[ELEVENLABS]` (demucs, julius, lameenc, sphn, einops…), khối `[SUBTITLE]` (faster-whisper), khối `[WAV2LIP]` (batch-face, opencv-transforms, sixdrepnet, `wav2lip-trip @ git+...`). **Giữ** torch/torchaudio/torchvision — `classifier_*.py` và CLIP còn dùng.
- [ ] ✏️ `_build_exe.ps1` — bỏ hidden-import / data của các module đã xoá nếu có
- [ ] ✏️ `README.md`, `GEMINI.md` — cập nhật mô tả kiến trúc
- [ ] **Xong khi**: build .exe thành công, nhẹ hơn hẳn, Downloader + Dubbing chạy đúng

---

**Thứ tự bắt buộc**: A → B → C → D → E. Giai đoạn B phải xong trước C, để khi nối GPU vào thì mọi thứ quanh nó đã được test rồi — lỗi lúc đó chắc chắn nằm ở model, không phải ở hạ tầng.
