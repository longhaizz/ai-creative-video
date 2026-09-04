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

# How often the sweeper wakes up to look for expired jobs. A tick while a
# job runs costs one lock and a walk over a handful of dict entries, so it
# can be short: the expensive part, deleting the files, is skipped then.
SWEEP_SECONDS = 60.0


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
        sweep_seconds: float = SWEEP_SECONDS,
    ):
        self._run_dub = run_dub
        self._jobs_dir = Path(jobs_dir if jobs_dir is not None else config.JOBS_DIR)
        self._ttl = ttl_seconds if ttl_seconds is not None else config.JOB_TTL_SECONDS
        self._sweep = sweep_seconds
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._sweeper: threading.Thread | None = None
        self._stopping = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Clean old files, then start the one worker.

        Nothing is in memory at boot, so any folder left in the jobs
        directory belongs to a job that died in a restart. Delete it.

        Calling this twice is a bug, and a loud one: a second worker would
        take jobs from the same queue, so two jobs would load models onto the
        same GPU and run out of memory. The old thread would also keep
        running with nobody holding it, and the disk clean-up above would
        delete the files of the job that is running right now.
        """
        if self._thread is not None:
            raise RuntimeError("JobRunner.start() was already called")
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        for path in self._jobs_dir.iterdir():
            _remove(path)
        self._stopping.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._sweeper = threading.Thread(target=self._sweep_forever, daemon=True)
        self._sweeper.start()

    def worker_alive(self) -> bool:
        """Is the one worker still there? /health reports this.

        Without it a dead worker looks exactly like a healthy idle server
        from the outside: jobs are still accepted, they just never run.
        """
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()  # wakes the sweeper out of its wait
        self._queue.put(None)  # tells the worker to return
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self._sweeper is not None:
            self._sweeper.join(timeout=timeout)
            self._sweeper = None

    # -- public API --------------------------------------------------------

    def create(self, params: Any) -> Job:
        """Make a job and its folder, but do NOT queue it yet.

        The caller still has to write the uploaded files into job.workdir.
        If we queued here, the worker could pick the job up while the video
        is still being uploaded.
        """
        job_id = uuid.uuid4().hex
        workdir = self._jobs_dir / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, params=params, workdir=workdir)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def enqueue(self, job_id: str) -> None:
        """Put a created job in the line. The files must be on disk by now."""
        self._queue.put(job_id)

    def submit(self, params: Any) -> Job:
        """create() + enqueue(), for callers with no files to write."""
        job = self.create(params)
        self.enqueue(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Ask a job to stop. Returns False if it already ended.

        A waiting job ends here and now: only raising the flag would leave it
        counted as queued until the worker reaches it, so every job behind it
        would report a line longer than it really is. A running job cannot be
        ended from here; it stops at its next check_cancel().

        The record stays in the store, with only the files removed. The client
        must still be able to read "cancelled" back, and purge_expired() takes
        the record away later, once finished_at is old enough.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in (DONE, FAILED, CANCELLED):
                return False
            job.cancelled = True
            waiting = job.status == QUEUED
            if waiting:
                job.status = CANCELLED
                job.finished_at = time.time()
        if waiting:
            # The worker still pulls the id off the queue later, sees the flag
            # and cleans up again. _remove() of a gone folder is a no-op.
            _remove(job.workdir)
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

        Only the records in memory are read. A folder this store never heard
        of is never touched, so a job whose folder exists but whose record is
        not in place yet cannot be deleted from under the thread making it.
        Folders left by a dead run are wiped by start() instead.
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

    def _busy(self) -> bool:
        """Is any job waiting or running?"""
        with self._lock:
            return any(job.status in (QUEUED, RUNNING)
                       for job in self._jobs.values())

    def _sweep_forever(self) -> None:
        """Delete expired jobs, but only while nothing else is working.

        Deleting a folder is disk work, and the pipeline is reading and
        writing wav files on the same disk. So a tick that finds the server
        busy does nothing and waits for the next one. A job takes minutes and
        the server is idle between jobs, so the sweep gets its turn.

        ponytail: a server that is never idle is never swept, and the disk
        fills. Give the skipped ticks a deadline if that day comes.
        """
        while not self._stopping.wait(self._sweep):
            # The same reason as the worker loop: a sweeper that dies stops
            # cleaning for ever and nothing says so.
            try:
                if self._busy():
                    continue
                self.purge_expired()
            except Exception as error:  # noqa: BLE001 - the sweeper must survive
                print(f"[sweeper] {error!r}", flush=True)

    # -- internals ---------------------------------------------------------

    def _append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.log.append(message)
        # The same line goes to stdout, so `pm2 logs dub` shows the work and
        # not only the HTTP requests. flush, because stdout under PM2 is a
        # pipe: without it Python holds the lines back until 8KB have piled up.
        print(f"[{job_id[:8]}] {message}", flush=True)

    def _set_step(self, job_id: str, name: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.step = name
        # Outside the lock: _append_log takes it too, and self._lock is a
        # plain Lock, so taking it twice in one thread would hang.
        self._append_log(job_id, name)

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
            # Nothing in here may end the loop. _run_one() already turns any
            # bug into a failed job, but this thread must survive whatever it
            # misses: if it dies, every job after it sits in the queue for
            # ever while /health still says "ok". That cost five days once.
            try:
                job = self.get(job_id)
                if job is not None:  # else it was dropped while waiting
                    self._run_one(job)
            except Exception as error:  # noqa: BLE001 - the worker must survive
                print(f"[worker] {error!r}", flush=True)

    def _run_one(self, job: Job) -> None:
        if self._is_cancelled(job.id):
            # Cancelled while it was still waiting. Never touch the GPU.
            self._cancel_and_clean(job)
            return

        with self._lock:
            job.status = RUNNING

        context = JobContext(self, job.id, job.workdir, job.params)
        try:
            result = self._run_dub(context)
        except JobCancelled:
            self._cancel_and_clean(job)
        except PipelineError as error:
            self._finish(job, FAILED, error=str(error), error_code=error.code)
        except Exception as error:  # noqa: BLE001 - any bug must land in the job
            self._finish(job, FAILED, error=str(error), error_code="internal")
        else:
            if self._is_cancelled(job.id):
                # The pipeline ignored the flag and ran to the end. The user
                # asked to stop, so we still call it cancelled.
                self._cancel_and_clean(job)
            else:
                self._finish(job, DONE, result_path=Path(result))

    def _cancel_and_clean(self, job: Job) -> None:
        """Delete the files first, then mark the job cancelled.

        The other way round is a lie the client can see: it polls, reads
        "cancelled", and the folder is still on disk for a moment.
        """
        _remove(job.workdir)
        self._finish(job, CANCELLED)


def _remove(path: Path) -> None:
    # Every OSError is swallowed, not only the missing folder. A file still
    # held open by a child process, or a busy network disk, leaves rubbish
    # behind; that costs disk. Letting the error out costs the worker.
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError:
        pass
