"""Dub server — one GPU, one queue, one worker.

The HTTP layer is thin on purpose: it saves the uploads, hands the job to
the JobRunner, and reads job state back out. All the real work sits behind
run_dub, which is passed into create_app().
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from server import config
from server.auth import require_api_key
from server.jobs import DONE, JobContext, JobRunner, PipelineError
from server.limits import BodySizeLimit
from server.schemas import DubRequest
from server.uploads import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, save_upload


def not_built_yet(context: JobContext) -> Path:
    """Stand-in until step 8 brings the real pipeline."""
    raise PipelineError("The dub pipeline is not built yet", code="internal")


def _gpu_name() -> str | None:
    """The GPU we are on, or None. torch is imported here, not at the top,
    so the server still starts on a machine that has no torch at all."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def create_app(run_dub=not_built_yet, models=()) -> FastAPI:
    """Build the app around one pipeline function.

    Tests pass a fake here. That is the only seam they need: everything in
    this file can then be checked with no GPU and no model files.

    `models` is a list of objects with a `name` and a `load()`. They are
    loaded once, before the first request, and stay in memory for every job
    after that. Loading them is why the server takes a minute to come up.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not config.API_KEY:
            raise RuntimeError("You must set the API_KEY environment variable")
        app.state.models_loaded = []
        for model in models:
            model.load()
            app.state.models_loaded.append(model.name)
        runner = JobRunner(run_dub)
        runner.start()
        app.state.runner = runner
        try:
            yield
        finally:
            runner.stop()

    app = FastAPI(
        title="Dub API",
        version="0.1.0",
        lifespan=lifespan,
        dependencies=[Depends(require_api_key)],
        # Do not publish the schema: this API serves only one client, and we
        # wrote that client ourselves.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Wrapped around everything, so it runs before routing and before the
    # token is checked. Without it, an upload of any size is written to disk
    # first and refused afterwards.
    app.add_middleware(BodySizeLimit, limit=config.MAX_REQUEST_BYTES)

    def get_runner() -> JobRunner:
        return app.state.runner

    def get_job_or_404(job_id: str):
        job = get_runner().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return job

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "models_loaded": app.state.models_loaded,
            "gpu": _gpu_name(),
        }

    @app.post("/dub", status_code=202)
    def dub(form: Annotated[DubRequest, Form()]):
        runner = get_runner()
        job = runner.create(form.settings())
        try:
            save_upload(
                form.video,
                job.workdir,
                VIDEO_EXTENSIONS,
                config.MAX_VIDEO_BYTES,
                "video",
            )
            if form.reference_audio is not None and form.reference_audio.filename:
                save_upload(
                    form.reference_audio,
                    job.workdir,
                    AUDIO_EXTENSIONS,
                    config.MAX_AUDIO_BYTES,
                    "reference_audio",
                )
        except Exception:
            # A job that never got its files must not sit in the store.
            runner.drop(job.id)
            raise
        runner.enqueue(job.id)
        return {"job_id": job.id}

    @app.get("/jobs/{job_id}")
    def job_state(job_id: str, since: Annotated[int, Query(ge=0)] = 0):
        get_job_or_404(job_id)
        state = get_runner().snapshot(job_id, since=since)
        if state is None:  # purged between the two calls
            raise HTTPException(status_code=404, detail="No such job")
        return JSONResponse(state)

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        runner = get_runner()
        job = get_job_or_404(job_id)
        if job.status != DONE:
            raise HTTPException(
                status_code=409,
                detail=f"Job is {job.status}, so there is no result to send",
            )
        # Forget the job once the file has gone out. Nobody downloads the
        # same dub twice, and the disk is not free.
        return FileResponse(
            job.result_path,
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
            background=BackgroundTask(runner.drop, job_id),
        )

    @app.delete("/jobs/{job_id}")
    def cancel_job(job_id: str):
        get_job_or_404(job_id)
        cancelled = get_runner().cancel(job_id)
        if not cancelled:
            raise HTTPException(
                status_code=409, detail="Job has already finished"
            )
        return {"job_id": job_id, "cancelling": True}

    return app


def build_production_app() -> FastAPI:
    """The real server: every model loaded, the real pipeline behind it.

    With LOAD_MODELS=0 it starts with nothing loaded instead, so the HTTP
    side can be worked on where there is no GPU. Jobs then fail at once.
    With LOAD_LIPSYNC=0 it still loads voice, but not LatentSync.
    Whisper now runs in the Open Dubbing venv, not here.
    """
    if not config.LOAD_MODELS:
        return create_app()

    from server.pipeline import Models, make_run_dub
    from server.steps.synth import VoxCPMModel

    lipsync = None
    if config.LOAD_LIPSYNC:
        from server.steps.lipsync import LipsyncModel

        lipsync = LipsyncModel(
            config.LATENTSYNC_DIR,
            config.LATENTSYNC_CONFIG,
            config.LATENTSYNC_CHECKPOINT,
        )
    models = Models(
        voice=VoxCPMModel(),
        lipsync=lipsync,
    )
    return create_app(make_run_dub(models), models=models.as_list())


app = build_production_app()
