import argparse
import os
import secrets
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from omegaconf import OmegaConf
from starlette.background import BackgroundTask

from scripts.inference import main as run_inference

API_KEY = os.getenv("LATENTSYNC_API_KEY", "")
CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

inference_lock = threading.Lock()
bearer_scheme = HTTPBearer(auto_error=False)


def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not secrets.compare_digest(credentials.credentials, API_KEY)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _acquire_gpu():
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="GPU is busy; retry later",
            headers={"Retry-After": "5"},
        )


def _remove_file(path: str | None):
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _save_upload(upload: UploadFile, allowed: set[str], max_bytes: int, label: str) -> str:
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"{label} must be one of: {', '.join(sorted(allowed))}",
        )

    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"{label} exceeds the {max_bytes // (1024 * 1024)} MB limit",
                    )
                output.write(chunk)
        return path
    except Exception:
        _remove_file(path)
        raise


def _build_args(
    video_path: str,
    audio_path: str,
    output_path: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    enable_deepcache: bool,
) -> argparse.Namespace:
    argv = [
        "--inference_ckpt_path",
        CHECKPOINT_PATH.absolute().as_posix(),
        "--video_path",
        video_path,
        "--audio_path",
        audio_path,
        "--video_out_path",
        output_path,
        "--inference_steps",
        str(inference_steps),
        "--guidance_scale",
        str(guidance_scale),
        "--seed",
        str(seed),
        "--temp_dir",
        "temp",
    ]
    if enable_deepcache:
        argv.append("--enable_deepcache")

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")
    return parser.parse_args(argv)


def _run_sync(
    video_path: str,
    audio_path: str,
    inference_steps: int,
    guidance_scale: float,
    seed: int,
    enable_deepcache: bool,
) -> str:
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    config = OmegaConf.load(CONFIG_PATH)
    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )
    args = _build_args(
        video_path,
        audio_path,
        output_path,
        inference_steps,
        guidance_scale,
        seed,
        enable_deepcache,
    )
    run_inference(config, args)
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        _remove_file(output_path)
        raise HTTPException(status_code=500, detail="Inference produced no output")
    return output_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("LATENTSYNC_API_KEY must be set")
    yield


app = FastAPI(
    title="LatentSync API",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/health")
def health():
    return JSONResponse(
        {
            "status": "ok",
            "config": CONFIG_PATH.as_posix(),
            "checkpoint": CHECKPOINT_PATH.as_posix(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )


@app.post("/sync")
def sync(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    inference_steps: Annotated[int, Form(ge=10, le=50)] = 20,
    guidance_scale: Annotated[float, Form(ge=1.0, le=3.0)] = 1.5,
    enable_deepcache: Annotated[bool, Form()] = True,
    seed: Annotated[int, Form()] = 1247,
):
    _acquire_gpu()
    video_path: str | None = None
    audio_path: str | None = None
    output_path: str | None = None
    try:
        video_path = _save_upload(video, ALLOWED_VIDEO_EXTENSIONS, MAX_VIDEO_BYTES, "Video")
        audio_path = _save_upload(audio, ALLOWED_AUDIO_EXTENSIONS, MAX_AUDIO_BYTES, "Audio")
        output_path = _run_sync(
            video_path,
            audio_path,
            inference_steps,
            guidance_scale,
            seed,
            enable_deepcache,
        )
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="sync.mp4",
            background=BackgroundTask(_remove_file, output_path),
        )
    finally:
        inference_lock.release()
        _remove_file(video_path)
        _remove_file(audio_path)
