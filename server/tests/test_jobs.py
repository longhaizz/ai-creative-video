import time

from server.jobs import CANCELLED, DONE, FAILED, QUEUED, RUNNING
from server.tests.conftest import wait_until
from server.tests.fake_pipeline import FakePipeline, crashing_pipeline, failing_pipeline


def test_jobs_run_one_at_a_time_in_order(make_runner):
    """One GPU means one job at a time. Overlap here would mean OOM in real life."""
    pipeline = FakePipeline(sleep=0.05)
    runner = make_runner(pipeline)

    jobs = [runner.submit({"n": n}) for n in range(3)]

    assert wait_until(lambda: len(pipeline.ran()) == 3), "all three should finish"
    assert pipeline.ran() == [job.id for job in jobs], "order must follow submit order"
    assert not pipeline.overlapped(), "two jobs must never run together"
    for job in jobs:
        assert runner.get(job.id).status == DONE


def test_queue_position_counts_jobs_in_front(make_runner):
    pipeline = FakePipeline(sleep=0.2)
    runner = make_runner(pipeline)

    first, second, third = (runner.submit({}) for _ in range(3))

    # The first job is already running, so it has no position in the line.
    assert wait_until(lambda: runner.get(first.id).status == RUNNING)
    assert runner.queue_position(first.id) is None
    assert runner.queue_position(second.id) == 0, "second is next"
    assert runner.queue_position(third.id) == 1, "one job in front of the third"

    assert wait_until(lambda: runner.get(third.id).status == DONE, timeout=10)
    assert runner.queue_position(third.id) is None


def test_result_file_is_kept_until_it_is_dropped(make_runner):
    runner = make_runner(FakePipeline())
    job = runner.submit({})

    assert wait_until(lambda: runner.get(job.id).status == DONE)
    result = runner.get(job.id).result_path
    assert result.is_file(), "the client has not downloaded it yet"

    runner.drop(job.id)
    assert runner.get(job.id) is None
    assert not result.exists(), "drop must delete the files as well"


def test_log_and_step_are_recorded(make_runner):
    runner = make_runner(FakePipeline(steps=3))
    job = runner.submit({})

    assert wait_until(lambda: runner.get(job.id).status == DONE)
    state = runner.snapshot(job.id)
    assert state["step"] == "step 3/3"
    assert state["log"] == ["step 1/3", "step 2/3", "step 3/3"]
    assert state["log_offset"] == 3


def test_snapshot_since_returns_only_new_lines(make_runner):
    runner = make_runner(FakePipeline(steps=3))
    job = runner.submit({})

    assert wait_until(lambda: runner.get(job.id).status == DONE)
    assert runner.snapshot(job.id, since=2)["log"] == ["step 3/3"]
    assert runner.snapshot(job.id, since=3)["log"] == []


def test_pipeline_error_keeps_its_code(make_runner):
    runner = make_runner(failing_pipeline("no face here", code="no_face"))
    job = runner.submit({})

    assert wait_until(lambda: runner.get(job.id).status == FAILED)
    state = runner.snapshot(job.id)
    assert state["error_code"] == "no_face"
    assert "no face here" in state["error"]


def test_a_plain_bug_does_not_kill_the_worker(make_runner):
    """A crash must fail one job, not stop the server for everyone."""
    runner = make_runner(crashing_pipeline("bad index"))

    first = runner.submit({})
    assert wait_until(lambda: runner.get(first.id).status == FAILED)
    assert runner.snapshot(first.id)["error_code"] == "internal"

    # The worker is still alive, so the next job is picked up and also fails.
    second = runner.submit({})
    assert wait_until(lambda: runner.get(second.id).status == FAILED)


def test_expired_jobs_are_purged_with_their_files(make_runner, tmp_path):
    runner = make_runner(FakePipeline(), ttl_seconds=0)
    job = runner.submit({})

    assert wait_until(lambda: runner.get(job.id) is None), "TTL 0 purges at once"
    assert not (tmp_path / "jobs" / job.id).exists()


def test_running_jobs_are_never_purged(make_runner):
    runner = make_runner(FakePipeline(sleep=0.3), ttl_seconds=0)
    job = runner.submit({})

    assert wait_until(lambda: runner.get(job.id).status == RUNNING)
    runner.purge_expired()
    assert runner.get(job.id) is not None, "a running job must survive a purge"
    assert wait_until(lambda: runner.get(job.id) is None, timeout=10)


def test_starting_twice_is_refused(make_runner):
    """A second worker would put two jobs on the same GPU at once."""
    import pytest

    runner = make_runner(FakePipeline())
    with pytest.raises(RuntimeError):
        runner.start()


def test_start_removes_files_left_by_a_dead_run(tmp_path):
    from server.jobs import JobRunner

    jobs_dir = tmp_path / "jobs"
    stale = jobs_dir / "old-job-id"
    stale.mkdir(parents=True)
    (stale / "half-written.mp4").write_bytes(b"x")

    runner = JobRunner(FakePipeline(), jobs_dir=jobs_dir)
    runner.start()
    try:
        assert not stale.exists(), "nothing is in memory at boot, so it is rubbish"
    finally:
        runner.stop()
