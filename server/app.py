"""Dub server — một GPU, một queue, một worker.

Bước 2 mới dựng khung: chỉ có /health. Queue/job/endpoint vào ở bước 3-4,
model vào ở bước 6-9.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from server import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.API_KEY:
        raise RuntimeError("Phải set biến môi trường API_KEY")
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Dub API",
    version="0.1.0",
    lifespan=lifespan,
    # Không phơi schema ra ngoài: API này chỉ phục vụ đúng một client mình viết.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": [],  # bước 6-9 sẽ điền
        "gpu": None,          # bước 6 (lúc có torch) sẽ điền
    }
