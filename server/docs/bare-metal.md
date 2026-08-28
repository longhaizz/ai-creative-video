# Dựng server không dùng Docker

Dành cho máy GPU cài thẳng. Docker vẫn là đường khuyến nghị — xem
[docker-compose.yml](../../docker-compose.yml). Tài liệu này để dành cho lúc
không dùng được Docker, hoặc khi cần mò lỗi trong môi trường thật.

Mọi lệnh chạy từ **gốc repo**.

---

## 0. Trước khi bắt đầu

- Ubuntu 22.04, GPU NVIDIA, driver đủ cho CUDA 12.1
- **48GB VRAM** là giả định của thiết kế. Card nhỏ hơn thì phải bỏ preload — xem mục *Card nhỏ hơn* ở cuối
- Ổ đĩa: ~20GB cho hai venv, ~5GB cho model weight

```bash
nvidia-smi                    # phải thấy card
python3.10 --version
```

## 1. Gói hệ thống

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv \
    python3.12 python3.12-venv \
    ffmpeg libgl1 libglib2.0-0
```

**Vì sao hai bản Python.** Bộ xoá phụ đề cần numpy 2.2 cho paddle, còn LatentSync cần numpy 1.26 — hai ABI không sống chung một process. Nó được thử nghiệm trên Python 3.12 (xem `video_subtitle_remover.ipynb`), phần còn lại chạy 3.10.

`ffmpeg` bắt buộc: mọi thao tác audio đều gọi nó.

## 2. Venv chính

```bash
python3.10 -m venv /opt/venv-main
/opt/venv-main/bin/pip install --upgrade pip setuptools wheel

/opt/venv-main/bin/pip install -r server/requirements.txt
/opt/venv-main/bin/pip install -r server/requirements-models.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

Kiểm:

```bash
/opt/venv-main/bin/python -c "import torch, numpy; \
    print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
# mong đợi: 2.5.1+cu121 True 1.26.4
```

`numpy` phải là **1.26.x**. Ra 2.x là có gói nào đó kéo bản mới lên, và LatentSync sẽ nổ lúc chạy.

## 3. Venv cho bộ xoá phụ đề

Thứ tự quan trọng, và ba trong bốn lệnh cần index riêng. `pip` chỉ nhận một `--index-url` mỗi lần chạy — đó là lý do phải tách ra chứ không gói vào một file requirements.

```bash
python3.12 -m venv /opt/venv-vsr
/opt/venv-vsr/bin/pip install --upgrade pip setuptools wheel

# 1. torch cu118 (khác venv chính, có chủ ý)
/opt/venv-vsr/bin/pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 2. paddle
/opt/venv-vsr/bin/pip install paddlepaddle-gpu==3.0.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 3. phần còn lại
/opt/venv-vsr/bin/pip install -r video-subtitle-remover/requirements.txt

# 4. onnxruntime GPU
/opt/venv-vsr/bin/pip install onnxruntime-gpu==1.20.1 \
    --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

Rồi plugin inference tốc độ cao (~1.2GB, tải một lần):

```bash
MPLBACKEND=Agg PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    /opt/venv-vsr/bin/paddlex --install hpi-gpu
```

Kiểm:

```bash
/opt/venv-vsr/bin/python -c "import torch, paddle, numpy; \
    print(torch.__version__, paddle.device.get_device(), numpy.__version__)"
# mong đợi: 2.7.0+cu118 gpu:0 2.2.5
```

numpy ở đây phải là **2.2.5**, khác hẳn venv chính. Hai con số khác nhau chính là lý do có hai venv.

## 4. Tải model weight

```bash
cd LatentSync && bash setup_env.sh && cd ..
ls -la LatentSync/checkpoints/latentsync_unet.pt      # ~1.3GB
ls -la LatentSync/checkpoints/whisper/                # small.pt hoặc tiny.pt
```

VoxCPM, faster-whisper, demucs và VAE tự tải từ Hugging Face lần chạy đầu. Đặt `HF_HOME` để chúng không nằm rải rác:

```bash
export HF_HOME=/models/huggingface
```

## 5. Cấu hình

```bash
cp .env.example .env
$EDITOR .env          # điền API_KEY, OPENAI_API_KEY
```

Sinh khoá:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 6. Chạy thử

```bash
set -a && source .env && set +a
export VSR_PYTHON=/opt/venv-vsr/bin/python

/opt/venv-main/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Lần khởi động đầu mất 1–2 phút vì nạp model. Xong thì:

```bash
curl -H "Authorization: Bearer $API_KEY" localhost:8000/health
```

Mong đợi:

```json
{"status":"ok","models_loaded":["voxcpm","whisper","latentsync"],"gpu":"NVIDIA ..."}
```

`models_loaded` rỗng nghĩa là `LOAD_MODELS=0` còn sót trong môi trường.

## 7. Chạy thật một clip

Trước khi dựng systemd, chạy tay để chắc chắn cả chuỗi hoạt động:

```bash
set -a && source .env && set +a
export VSR_PYTHON=/opt/venv-vsr/bin/python

/opt/venv-main/bin/python -m server.tests.smoke_run_dub clip.mp4 \
    --lipsync --remove-subtitle --burn-subtitle --lang vi
```

Nó in mốc thời gian từng bước, kể cả thời gian nạp từng model. Đây là chỗ xác nhận việc preload có tác dụng: chạy hai lần liên tiếp, lần thứ hai **không** được mất thêm 30–60 giây nạp LatentSync.

## 8. systemd

`/etc/systemd/system/dub.service`:

```ini
[Unit]
Description=Dub server
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User=dub
WorkingDirectory=/srv/dub
EnvironmentFile=/srv/dub/.env
Environment=VSR_PYTHON=/opt/venv-vsr/bin/python
Environment=HF_HOME=/models/huggingface
ExecStart=/opt/venv-main/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=10
# Nạp model mất 1-2 phút; đừng để systemd tưởng nó treo.
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now dub
journalctl -u dub -f
```

**`--workers 1` là bắt buộc.** Worker thứ hai sẽ lấy job từ cùng hàng đợi và đặt hai job lên cùng một card. Code có chặn `start()` gọi hai lần trong một process, nhưng nó không thể chặn hai process.

**Restart là mất job.** Trạng thái nằm trong RAM; job đang chạy và đang chờ đều biến mất. Đúng thiết kế — 3 phút GPU đã đốt thì có database cũng không tua lại được.

---

## Chạy sau reverse proxy

`server/limits.py` đã từ chối request quá cỡ trước khi đọc body, nên server tự đứng một mình được. Nhưng có nginx thì nên chặn sớm hơn nữa:

```nginx
location / {
    client_max_body_size 250m;
    proxy_pass http://127.0.0.1:8000;
    proxy_read_timeout 3600s;     # job chạy hàng phút; poll thì nhanh
    proxy_request_buffering off;   # đừng ghi đệm 200MB lên đĩa nginx
}
```

## Card nhỏ hơn 48GB

Thiết kế giả định nạp sẵn **mọi** model. Thường trú ~12–16GB, lúc diffusion 512 đỉnh ~25GB. Card nhỏ hơn thì:

- Dùng `LATENTSYNC_CONFIG=configs/unet/stage2.yaml` (256 thay vì 512)
- Hoặc bỏ preload LatentSync, cho nó nạp theo từng job — mất 30–60 giây mỗi job, tức là quay lại đúng vấn đề mà bước 6 đã đi sửa

## Gỡ lỗi hay gặp

| Triệu chứng | Nguyên nhân |
|---|---|
| `numpy.dtype size changed` | numpy sai bản. Kiểm lại mục 2 và 3 — mỗi venv một con số riêng |
| Job fail `error_code: internal`, log nhắc paddle | Sai `VSR_PYTHON`, hoặc quên bước `paddlex --install hpi-gpu` |
| Job fail `error_code: no_face` | Không phải lỗi. Video không có mặt người; bỏ tick Lipsync |
| `/health` trả `models_loaded: []` | `LOAD_MODELS=0` còn trong môi trường |
| `RuntimeError: You must set the API_KEY` | Chưa `source .env`, hoặc systemd thiếu `EnvironmentFile` |
| Nạp model xong hết VRAM | Card nhỏ hơn 48GB — xem mục trên |
