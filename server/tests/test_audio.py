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


def test_clean_take_never_runs_one_ffmpeg(tmp_path, monkeypatch):
    """The graph deadlocks in one run, so it must stay in two.

    Reversing a take, then loudnorm, then any rate change hangs ffmpeg 6.1
    for good. The split is the fix, so a later tidy-up that joins the two
    filter lists back into one has to fail here.
    """
    from pathlib import Path

    from server.steps import audio

    seen = []

    def fake_ffmpeg(command, timeout=None):
        seen.append(command)
        # Stand in for the file the first run writes.
        Path(command[-1]).write_bytes(b"")
        return ""

    monkeypatch.setattr(audio, "run_ffmpeg", fake_ffmpeg)
    audio.clean_take(tmp_path / "take.wav", tmp_path / "take_clean.wav")

    assert len(seen) == 2, "clean_take must stay two ffmpeg runs"
    first, second = (" ".join(cmd) for cmd in seen)
    assert "areverse" in first and "loudnorm" not in first
    assert "loudnorm" in second and "areverse" not in second
    # Nothing may resample in the same run as the reversing.
    assert "aresample" not in first
    # The scratch file must not be left behind for the next take.
    assert not (tmp_path / "take_clean_cut.wav").exists()
