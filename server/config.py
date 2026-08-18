"""Cấu hình đọc từ biến môi trường.

Chỉ khai báo thứ đã có người dùng. Các key của bước sau (OPENAI_API_KEY,
VSR_PYTHON, VSR_REPO) thêm vào đúng lúc bước đó cần.
"""

from __future__ import annotations

import os
from pathlib import Path

# Bearer token client phải gửi. Rỗng = server từ chối khởi động (xem app.py) —
# thà chết lúc boot còn hơn chạy một API không khoá cửa.
API_KEY = os.getenv("API_KEY", "")

# Nơi giữ file kết quả tới khi client tải về hoặc hết TTL.
JOBS_DIR = Path(os.getenv("JOBS_DIR", "jobs"))

# Job quá hạn thì xoá cả state lẫn file. 1 giờ.
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))

MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(200 * 1024 * 1024)))
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
