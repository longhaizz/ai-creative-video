"""A fake run_dub, so tests can drive the queue without a GPU.

This is the whole point of the JobRunner seam: the real pipeline needs a
48GB card and several minutes, but the queue, the job state and the cancel
path are plain Python and must be tested on any machine.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from server.jobs import JobContext, PipelineError


class FakePipeline:
    """Records every call, so a test can check order and overlap."""

    def __init__(self, steps: int = 2, sleep: float = 0.01):
        self.steps = steps
        self.sleep = sleep
        self.calls: list[tuple[str, float, float]] = []  # (job_id, start, end)
        self._lock = threading.Lock()

    def __call__(self, context: JobContext) -> Path:
        start = time.monotonic()
        for number in range(1, self.steps + 1):
            context.check_cancel()
            context.step(f"step {number}/{self.steps}")
            time.sleep(self.sleep)
        context.check_cancel()

        result = context.workdir / "result.mp4"
        result.write_bytes(b"fake video")
        with self._lock:
            self.calls.append((context.job_id, start, time.monotonic()))
        return result

    def ran(self) -> list[str]:
        with self._lock:
            return [job_id for job_id, _, _ in self.calls]

    def overlapped(self) -> bool:
        """True if any two jobs ran at the same time. Must always be False."""
        with self._lock:
            windows = sorted((start, end) for _, start, end in self.calls)
        return any(
            windows[i][1] > windows[i + 1][0] for i in range(len(windows) - 1)
        )


def failing_pipeline(message: str = "boom", code: str = "internal"):
    def run(context: JobContext) -> Path:
        context.step("about to fail")
        raise PipelineError(message, code=code)

    return run


def crashing_pipeline(message: str = "unexpected"):
    """A plain bug, not a PipelineError. It must not kill the worker."""

    def run(context: JobContext) -> Path:
        raise ValueError(message)

    return run
