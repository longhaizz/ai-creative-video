"""The one thing worth a test in run_ffmpeg: it must give up."""

import sys

import pytest

from server.jobs import PipelineError
from server.steps.audio import run_ffmpeg


def test_a_command_that_never_ends_is_stopped():
    """A stalled ffmpeg must fail the job, not hold the worker for days.

    Any hanging program does here what a stalled filter graph does in real
    life: it sleeps and never exits. No ffmpeg needed to check the rule.
    """
    with pytest.raises(PipelineError) as error:
        run_ffmpeg(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.5,
        )
    assert "longer than" in str(error.value), "the user should read why"
