import sys
import time
from pathlib import Path

import pytest

from server.jobs import JobRunner

# The tests import `server.tests.fake_pipeline`, so the repo root must be on
# the path when pytest is started from somewhere else.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.fixture
def make_runner(tmp_path):
    """Build a JobRunner on a temp folder and always stop it afterwards."""
    started: list[JobRunner] = []

    def build(run_dub, ttl_seconds: int = 3600,
              sweep_seconds: float = 0.01) -> JobRunner:
        # The sweep is tiny here: a test must not wait a real minute for the
        # clean-up thread to take its turn.
        runner = JobRunner(run_dub, jobs_dir=tmp_path / "jobs",
                           ttl_seconds=ttl_seconds, sweep_seconds=sweep_seconds)
        runner.start()
        started.append(runner)
        return runner

    yield build
    for runner in started:
        runner.stop()


def wait_until(check, timeout: float = 5.0, interval: float = 0.01):
    """Poll until `check()` is true. Returns False if the time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(interval)
    return False
