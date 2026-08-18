"""Job queue for the dub server.

One GPU means one worker. Jobs wait in a plain queue.Queue and run one at a
time, in the order they arrived. Submitting a job always works: the client
never gets "busy, try again", it just gets a place in the line.

Everything is kept in memory. If the server restarts, running jobs die. A
database would not save them either, because the GPU work is lost anyway.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from server import config

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


class JobCancelled(Exception):
    """The pipeline raises this when the user cancels a running job."""


class PipelineError(Exception):
    """A failure we can explain to the user.

    `code` is a short machine word (see error_code in the API), so the client
    can branch on it instead of reading English text from some upstream
    library. That text can change at any time; our code cannot.
    """

    def __init__(self, message: str, code: str = "internal"):
        super().__init__(message)
        self.code = code


@dataclass
class Job:
    id: str
    params: Any
    workdir: Path
    status: str = QUEUED
    step: str = ""
    log: list[str] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    result_path: Path | None = None
    cancelled: bool = False
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class JobContext:
    """What the pipeline gets. It hides the job store from the pipeline.

    The pipeline only needs three things: a place to write files, a way to
    talk to the user, and a way to stop early.
    """

    def __init__(self, runner: "JobRunner", job_id: str, workdir: Path, params: Any):
        self._runner = runner
        self.job_id = job_id
        self.workdir = workdir
        self.params = params

    def log(self, message: str) -> None:
        self._runner._append_log(self.job_id, message)

    def step(self, name: str) -> None:
        """Name the current step, and write it to the log as well."""
        self._runner._set_step(self.job_id, name)

    def check_cancel(self) -> None:
        """Call this between steps. It raises if the user pressed Stop."""
        if self._runner._is_cancelled(self.job_id):
            raise JobCancelled()


RunDub = Callable[[JobContext], Path]


class JobRunner:
    """Holds the jobs, the queue and the single worker thread.

    `run_dub` is passed in, not imported. That is the one seam of this
    server: tests give it a fake and can then check the queue, the job state
    and the cancel path without a GPU.
    """

    def __init__(
        self,
        run_dub: RunDub,
        jobs_dir: Path | None = None,
        ttl_seconds: int | None = None,
    ):
        self._run_dub = run_dub
        self._jobs_dir = Path(jobs_dir if jobs_dir is not None else config.JOBS_DIR)
        self._ttl = ttl_seconds if ttl_seconds is not None else config.JOB_TTL_SECONDS
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Clean old files, then start the worker.

        Nothing is in memory at boot, so any folder left in the jobs
        directory belongs to a job that died in a restart. Delete it.
        """
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        for path in self._jobs_dir.iterdir():
            _remove(path)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._queue.put(None)  # tells the worker to return
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # -- public API --------------------------------------------------------

    def submit(self, params: Any) -> Job:
        self.purge_expired()
        job_id = uuid.uuid4().hex
        workdir = self._jobs_dir / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, params=params, workdir=workdir)
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. Returns False if it already ended.

        A queued job is dropped when the worker reaches it. A running job
        stops at its next check_cancel().
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in (DONE, FAILED, CANCELLED):
                return False
            job.cancelled = True
            return True

    def queue_position(self, job_id: str) -> int | None:
        """How many jobs are waiting in front of this one. 0 means next.

        We walk the dict, because a dict keeps insertion order and submit()
        inserts in queue order. Do not sort by created_at: on Windows
        time.time() only moves every ~15ms, so jobs sent in the same moment
        share a timestamp and none of them looks earlier than the others.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != QUEUED:
                return None
            position = 0
            for other in self._jobs.values():
                if other.id == job_id:
                    return position
                if other.status == QUEUED:
                    position += 1
            return None

    def snapshot(self, job_id: str, since: int = 0) -> dict | None:
        """The job state as the API sends it.

        `since` is how many log lines the client already has, so a client
        that polls every 2 seconds does not pull the whole log every time.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            log = job.log[since:]
            state = {
                "job_id": job.id,
                "status": job.status,
                "step": job.step,
                "log": log,
                "log_offset": len(job.log),
                "error": job.error,
                "error_code": job.error_code,
            }
        position = self.queue_position(job_id)
        if position is not None:
            state["queue_position"] = position
        return state

    def drop(self, job_id: str) -> None:
        """Forget a job and delete its files. Used after a download."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None:
            _remove(job.workdir)

    def purge_expired(self) -> None:
        """Delete jobs that ended more than TTL seconds ago.

        ponytail: called on submit and after each job, not on a timer. On an
        idle server old files stay a bit longer, which costs disk but never
        correctness. Add a timer thread only if disk actually fills up.
        """
        now = time.time()
        expired = []
        with self._lock:
            for job in list(self._jobs.values()):
                if job.status in (QUEUED, RUNNING):
                    continue
                ended = job.finished_at or job.created_at
                if now - ended >= self._ttl:
                    expired.append(self._jobs.pop(job.id))
        for job in expired:
            _remove(job.workdir)

    # -- internals ---------------------------------------------------------

    def _append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.log.append(message)

    def _set_step(self, job_id: str, name: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.step = name
                job.log.append(name)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return job is not None and job.cancelled

    def _finish(self, job: Job, status: str, **fields) -> None:
        with self._lock:
            job.status = status
            job.finished_at = time.time()
            for key, value in fields.items():
                setattr(job, key, value)

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:  # sentinel from stop()
                return
            job = self.get(job_id)
            if job is None:  # dropped or purged while waiting
                continue
            self._run_one(job)
            self.purge_expired()

    def _run_one(self, job: Job) -> None:
        if self._is_cancelled(job.id):
            # Cancelled while it was still waiting. Never touch the GPU.
            self._finish(job, CANCELLED)
            _remove(job.workdir)
            return

        with self._lock:
            job.status = RUNNING

        context = JobContext(self, job.id, job.workdir, job.params)
        try:
            result = self._run_dub(context)
        except JobCancelled:
            self._finish(job, CANCELLED)
            _remove(job.workdir)
        except PipelineError as error:
            self._finish(job, FAILED, error=str(error), error_code=error.code)
        except Exception as error:  # noqa: BLE001 - any bug must land in the job
            self._finish(job, FAILED, error=str(error), error_code="internal")
        else:
            if self._is_cancelled(job.id):
                # The pipeline ignored the flag and ran to the end. The user
                # asked to stop, so we still call it cancelled.
                self._finish(job, CANCELLED)
                _remove(job.workdir)
            else:
                self._finish(job, DONE, result_path=Path(result))


def _remove(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except FileNotFoundError:
        pass
