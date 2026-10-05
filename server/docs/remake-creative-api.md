# API tab Remake Creatives

Tab **Remake Creatives** trên tool spy ads không chạy model trên máy user. Nó gửi video lên dub server, hỏi trạng thái, rồi tải file về. Tài liệu này là đúng các endpoint tab đó đang gọi.

Base URL và API key nằm trong **⚙ Cài đặt**. Ô để trống thì tool dùng server mặc định trong `global-configs.yaml`. Mọi request đều cần header:

```
Authorization: Bearer <API_KEY>
```

Thiếu hoặc sai key thì `401`, kể cả `GET /health`.

Server chạy **một GPU, một hàng đợi, một job một lúc**. Nộp nhiều video thì job sau đứng chờ, không bị từ chối. Job nằm trong RAM khoảng 1 giờ (`JOB_TTL_SECONDS`, mặc định 3600). Tải xong hoặc hết giờ thì job biến mất. Restart server thì job đang chạy cũng mất.

`POST /speak` cũng nằm trên server này, nhưng tab Remake Creatives **không gọi**. Tab chỉ dùng các endpoint bên dưới.

## Luồng

```
GET  /health                 kiểm tra kết nối (nút Kiểm tra)
GET  /voices                 danh sách giọng preset
POST /dub                    nộp video, nhận job_id
GET  /jobs/{id}?since=N      hỏi trạng thái, mỗi 2 giây
GET  /jobs/{id}/result       tải file khi status = done
DELETE /jobs/{id}            huỷ (nút Stop)
```

Tool nộp hết video trong lô trước, rồi mới poll. Không gửi lại `POST /dub` khi mạng chớp: POST không được retry, vì retry sẽ xếp thêm một bản copy lên GPU.

Bool gửi dạng chuỗi `true` / `false`, không phải `True`.

## GET /health

Nút **Kiểm tra** trong Cài đặt.

```bash
curl -H "Authorization: Bearer $API_KEY" "$BASE/health"
```

```json
{
  "status": "ok",
  "models_loaded": ["voxcpm", "whisper", "latentsync"],
  "gpu": "NVIDIA ...",
  "worker_alive": true
}
```

`worker_alive: false` nghĩa là hàng đợi không còn ai xử lý. Job mới sẽ nằm mãi ở `queued`.

## GET /voices

Nút làm mới danh sách giọng. `original` không nằm trong danh sách này: tool luôn thêm sẵn. Mọi id khác phải có trong response, nếu không `POST /dub` trả `422`.

```bash
curl -H "Authorization: Bearer $API_KEY" "$BASE/voices"
```

```json
[
  {"id": "kai_clean", "label": "Kai Clean"}
]
```

Gửi id đó vào `voice_mode`. `original` thì server nhái giọng trong video, hoặc giọng file `reference_audio` nếu có upload WAV.

## POST /dub

`multipart/form-data`. Trả `202`:

```json
{"job_id": "…"}
```

Video bắt buộc, tối đa 200MB. Đuôi nhận: `.mp4`, `.mov`, `.mkv`, `.webm`. WAV giọng mẫu tối đa 25MB, đuôi `.wav`, `.mp3`, `.m4a`, `.flac`. Cả request (video + WAV + form) tối đa khoảng 226MB, chặn trước khi ghi đĩa: quá cỡ là `413`. Đuôi lạ là `415`.

```bash
curl -H "Authorization: Bearer $API_KEY" \
  -F "video=@clip.mp4" \
  -F "dub=true" \
  -F "target_lang=VI" \
  -F "voice_mode=original" \
  -F "remove_subtitle=true" \
  -F "burn_subtitle=true" \
  -F "name_from_content=false" \
  "$BASE/dub"
```

### Việc tool gửi

Bỏ một field thì server dùng mặc định trong cột phải. `source_title` không gửi: server tự lấy từ tên file upload để gợi ý cho bước dịch.

| Field | UI | Mặc định | Ghi chú |
|---|---|---|---|
| `video` | danh sách video | bắt buộc | file |
| `reference_audio` | WAV khi giọng Original | không có | file, chỉ khi user chọn WAV |
| `dub` | Lồng tiếng | `true` | `false` = giữ tiếng gốc, chỉ sửa hình |
| `target_lang` | Target language | `same` | mã bên dưới |
| `voice_mode` | giọng | `original` | hoặc `id` từ `GET /voices` |
| `cfg_value` | Cài đặt → Bám giọng mẫu | `2.0` | `1.0`–`3.0` |
| `inference_timesteps` | Cài đặt → Độ kỹ giọng đọc | `10` | `5`, `10`, `20` hoặc `30` |
| `speakers` | Cài đặt → Số người nói | bỏ qua | chỉ gửi `1` khi user chọn 1. `0` / Tự động thì không gửi |
| `whisper_model` | Cài đặt → Whisper | `medium` | `tiny` `base` `small` `medium` `large-v3` |
| `remove_subtitle` | Xoá sub | `false` | |
| `vsr_mode` | Cài đặt → Kiểu xoá chữ | `sttn-det` | `sttn-det` `sttn-auto` `lama` `propainter` |
| `vsr_top` `vsr_bottom` `vsr_left` `vsr_right` | vùng quét | `0.60` `0.96` `0.03` `0.97` | tỉ lệ khung, `0`–`1`. Trên phải nhỏ hơn dưới, trái nhỏ hơn phải |
| `translate_screen_text` | dịch chữ trên hình | `false` | chữ không phải phụ đề: tiêu đề, giá, nút |
| `screen_text_min_seconds` | thời gian chữ thường | `1.0` nếu không gửi | chỉ gửi khi dịch chữ trên hình. `0`–`30`. Ô trên tool bắt đầu từ `0` |
| `screen_text_small_min_seconds` | thời gian chữ nhỏ | `3.0` nếu không gửi | chữ dưới 3.5% chiều cao khung. Ô trên tool bắt đầu từ `0` |
| `screen_text_inpaint` | xoá chữ bằng LAMA | `false` | tắt thì che bằng hình chữ nhật trắng |
| `lipsync` | Lipsync | `false` | `dub=false` mà `lipsync=true` thì `422` |
| `latentsync_steps` | Cài đặt → Số vòng lặp | `50` | `1`–`100`. Tool mặc định ô này là `20` |
| `latentsync_guidance` | Cài đặt → Bám audio | `1.5` | `1.0`–`3.0` |
| `burn_subtitle` | Tạo sub | `false` | |
| `subtitle_font` | cố định | `Noto Sans` | tool luôn gửi `Noto Sans` |
| `subtitle_size` | cỡ sub | bỏ qua | `8`–`200`. Bỏ qua thì server tự lấy 56px trên khung cao 1920, rồi scale |
| `subtitle_position` | vị trí sub | bỏ qua | `0`–`1`, tâm dòng theo chiều cao. Bỏ qua thì server đặt lại chỗ sub cũ |
| `name_from_content` | Cài đặt → Tự đặt tên theo nội dung | `false` | xem mục tên file |
| `hook_text` và `hook_top` `hook_bottom` `hook_left` `hook_right` | Thay text hook | không gửi | có chữ thì phải có đủ 4 cạnh. Thiếu một phía là `422` |
| `hook_font` | phông hook | `Noto Sans` | tool gửi `Arial` |
| `hook_size` | cỡ hook | bỏ qua | `8`–`200` |
| `hook_colour` | màu | `#FFFFFF` | `#` + 6 ký tự hex |
| `hook_align` | căn | `center` | `left` `center` `right` |
| `hook_prewrapped` | tool đã ngắt dòng | `false` | `true` thì server vẽ đúng các dòng đã gửi |
| `hook_colours` | màu từng dòng | `""` | các màu cách nhau bằng dấu phẩy, thiếu thì dùng `hook_colour` |

`target_lang` tool đang có: `same`, `HI`, `ID`, `TL`, `AR`, `TR`, `TH`, `EN`, `VI`, `ZH`, `DA`, `NL`, `FI`, `FR`, `DE`, `EL`, `HE`, `IT`, `JA`, `KO`, `NB`, `PL`, `PT`, `RU`, `ES`, `SV`. `same` nghĩa là giữ ngôn ngữ Whisper nghe được, không dịch sang ngôn ngữ khác.

`dub=false` thì mọi thông số giọng bị bỏ qua. Job phải làm ít nhất một việc trên hình: `remove_subtitle`, `burn_subtitle`, `hook_text`, hoặc `translate_screen_text`. Không có gì để làm thì `422`.

Không thấy mặt người thì bước lipsync bị bỏ qua và job vẫn ra video. Creative quảng cáo thường không có talking-head.

### Tên file

`name_from_content=false` (mặc định, ô trong Cài đặt tắt): response job **không** có `output_name`. Tool tự đặt tên theo file gốc, ngôn ngữ, và các bước đã bật, ví dụ `clip_vi_lipsyn_remove_sub.mp4`.

`name_from_content=true`: sau khi lồng tiếng, server đặt một tiêu đề 40–80 ký tự từ lời thoại đã dịch, viết bằng ngôn ngữ đích. Giữ dấu và khoảng trắng. Khi đặt được, `GET /jobs/{id}` thêm:

```json
"output_name": "Kem chống nắng giúp da hết thâm chỉ sau bảy ngày dùng"
```

Không có đuôi `.mp4`. File tải về dùng tên đó. Video không lồng tiếng, không có lời, hoặc model không viết được tiêu đề đúng độ dài thì không có `output_name`, tool giữ tên cũ. Lỗi đặt tên không làm hỏng job.

## GET /jobs/{id}

`since` là số dòng log client đã có. Lần đầu gửi `0`. Lần sau gửi `log_offset` của lần trước để chỉ nhận dòng mới. Tool poll mỗi 2 giây.

```bash
curl -H "Authorization: Bearer $API_KEY" "$BASE/jobs/$JOB?since=0"
```

```json
{
  "job_id": "…",
  "status": "running",
  "step": "Making the voice (4 blocks)",
  "log": ["Voice track: 12.4s of 15.0s"],
  "log_offset": 18,
  "error": null,
  "error_code": null,
  "queue_position": 0
}
```

`queue_position` chỉ có khi job còn `queued`. `0` nghĩa là không còn job nào xếp trước nó.

`status`: `queued`, `running`, `done`, `failed`, `cancelled`.

Hết `queued` và `running` thì dừng poll. `done` thì tải file. `failed` thì đọc `error` và `error_code`:

| `error_code` | Nghĩa |
|---|---|
| `invalid_input` | video hoặc tham số không xử lý được |
| `no_face` | mã dành cho trường hợp không thấy mặt mà job bị đánh hỏng. Pipeline hiện tại bỏ qua lipsync chứ không dùng mã này để fail cả job |
| `internal` | lỗi khác |

Job không còn (hết hạn, đã tải, chưa từng có) thì `404`.

## GET /jobs/{id}/result

Chỉ khi `status` là `done`. Trả file (`video/mp4` với job dub). Header `Content-Disposition` mang tên file: `output_name` nếu có, không thì chính `job_id`.

Tải xong server **xoá job**. Gọi lần hai là `404`. Gọi khi job chưa `done` là `409`.

```bash
curl -H "Authorization: Bearer $API_KEY" \
  "$BASE/jobs/$JOB/result" -o out.mp4
```

## DELETE /jobs/{id}

Nút Stop. Job đang chờ thì dừng ngay. Job đang chạy thì dừng ở ranh giới bước gần nhất. Một bước GPU đã bắt đầu thì phải chờ bước đó chạy nốt.

```bash
curl -X DELETE -H "Authorization: Bearer $API_KEY" "$BASE/jobs/$JOB"
```

```json
{"job_id": "…", "cancelling": true}
```

Job đã `done`, `failed`, hoặc `cancelled` thì `409`.

## Một job giống tool đang bấm chạy

Lồng tiếng sang tiếng Việt, xoá sub, tạo sub, không lipsync, không tự đặt tên:

```bash
JOB=$(curl -s -H "Authorization: Bearer $API_KEY" \
  -F "video=@clip.mp4" \
  -F "dub=true" \
  -F "target_lang=VI" \
  -F "voice_mode=original" \
  -F "cfg_value=2.0" \
  -F "inference_timesteps=10" \
  -F "whisper_model=medium" \
  -F "remove_subtitle=true" \
  -F "vsr_mode=sttn-det" \
  -F "vsr_top=0.60" -F "vsr_bottom=0.96" \
  -F "vsr_left=0.03" -F "vsr_right=0.97" \
  -F "lipsync=false" \
  -F "burn_subtitle=true" \
  -F "subtitle_font=Noto Sans" \
  -F "name_from_content=false" \
  "$BASE/dub" | jq -r .job_id)

curl -H "Authorization: Bearer $API_KEY" "$BASE/jobs/$JOB?since=0"
curl -H "Authorization: Bearer $API_KEY" "$BASE/jobs/$JOB/result" -o out.mp4
```
