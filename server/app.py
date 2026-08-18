"""Dub server — one GPU, one queue, one worker.

Step 2 only builds the frame: just /health. The queue, jobs and endpoints
come in steps 3-4. The models come in steps 6-9.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from server import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.API_KEY:
        raise RuntimeError("You must set the API_KEY environment variable")
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Dub API",
    version="0.1.0",
    lifespan=lifespan,
    # Do not publish the schema: this API serves only one client, and we
    # wrote that client ourselves.
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": [],  # steps 6-9 will fill this
        "gpu": None,          # step 6 will fill this, once torch is installed
    }
