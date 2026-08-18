import os
import secrets
import tempfile
import threading
from typing import Annotated
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from voxcpm import VoxCPM

MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
DEVICE = os.getenv("DEVICE", "cuda:0")
API_KEY = os.getenv("VOXCPM_API_KEY", "")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

model: VoxCPM | None = None
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


def _get_model() -> VoxCPM:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return model


def _acquire_gpu():
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="GPU is busy; retry later",
            headers={"Retry-After": "5"},
        )


def _save_upload(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Audio must be .wav, .mp3, .m4a, or .flac",
        )

    fd, path = tempfile.mkstemp(suffix=suffix)
    size = 0
    try:
        with os.fdopen(fd, "wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Audio file exceeds the 25 MB limit",
                    )
                output.write(chunk)
        return path
    except Exception:
        _remove_file(path)
        raise


def _remove_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not API_KEY:
        raise RuntimeError("VOXCPM_API_KEY must be set")
    model = VoxCPM.from_pretrained(
        MODEL_ID,
        device=DEVICE,
        load_denoiser=False,
    )
    yield


app = FastAPI(
    title="VoxCPM2 API",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _wav_response(wav, filename: str) -> FileResponse:
    current_model = _get_model()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, wav, current_model.tts_model.sample_rate)
        return FileResponse(
            path,
            media_type="audio/wav",
            filename=filename,
            background=BackgroundTask(_remove_file, path),
        )
    except Exception:
        _remove_file(path)
        raise


@app.get("/health")
def health():
    current_model = _get_model()
    return JSONResponse(
        {
            "status": "ok",
            "model": MODEL_ID,
            "sample_rate": current_model.tts_model.sample_rate,
            "device": DEVICE,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    cfg_value: float = Field(default=2.0, ge=1.0, le=3.0)
    inference_timesteps: int = Field(default=10, ge=5, le=20)


@app.post("/tts")
def tts(req: TTSRequest):
    _acquire_gpu()
    try:
        wav = _get_model().generate(
            text=req.text,
            cfg_value=req.cfg_value,
            inference_timesteps=req.inference_timesteps,
        )
        return _wav_response(wav, "tts.wav")
    finally:
        inference_lock.release()


@app.post("/clone")
def clone_voice(
    text: Annotated[str, Form(min_length=1, max_length=1000)],
    reference_audio: UploadFile = File(...),
    cfg_value: Annotated[float, Form(ge=1.0, le=3.0)] = 2.0,
    inference_timesteps: Annotated[int, Form(ge=5, le=20)] = 10,
):
    _acquire_gpu()
    ref_path: str | None = None
    try:
        ref_path = _save_upload(reference_audio)
        wav = _get_model().generate(
            text=text,
            reference_wav_path=ref_path,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        return _wav_response(wav, "clone.wav")
    finally:
        inference_lock.release()
        if ref_path:
            _remove_file(ref_path)


@app.post("/hifi-clone")
def hifi_clone(
    text: Annotated[str, Form(min_length=1, max_length=1000)],
    prompt_text: Annotated[str, Form(min_length=1, max_length=1000)],
    reference_audio: UploadFile = File(...),
    control: Annotated[str, Form(max_length=1000)] = "",
    cfg_value: Annotated[float, Form(ge=1.0, le=3.0)] = 2.0,
    inference_timesteps: Annotated[int, Form(ge=5, le=20)] = 10,
):
    _acquire_gpu()
    ref_path: str | None = None
    try:
        ref_path = _save_upload(reference_audio)
        final_text = f"({control.strip()}){text}" if control.strip() else text
        wav = _get_model().generate(
            text=final_text,
            prompt_wav_path=ref_path,
            prompt_text=prompt_text,
            reference_wav_path=ref_path,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        return _wav_response(wav, "hifi_clone.wav")
    finally:
        inference_lock.release()
        if ref_path:
            _remove_file(ref_path)
