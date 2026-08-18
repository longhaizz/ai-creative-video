"""Cancelling a job.

Cancel is cooperative: DELETE only raises a flag. A waiting job is dropped
before it ever touches the GPU. A running job stops at its next
check_cancel(), which the pipeline calls between steps.

We do not kill the thread. The worker and the models share one process, so
killing it in the middle of an inference would leave VRAM and half-written
files behind with nobody to clean them up.
"""

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.jobs import CANCELLED, DONE, QUEUED, RUNNING
from server.tests.conftest import wait_until
from server.tests.fake_pipeline import FakePipeline

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


# -- through the JobRunner --------------------------------------------------


def test_a_waiting_job_never_reaches_the_pipeline(make_runner):
    pipeline = FakePipeline(sleep=0.3)
    runner = make_runner(pipeline)

    first = runner.submit({})
    waiting = runner.submit({})
    assert runner.get(waiting.id).status == QUEUED

    assert runner.cancel(waiting.id) is True
    assert wait_until(lambda: runner.get(waiting.id).status == CANCELLED, timeout=10)
    assert pipeline.ran() == [first.id], "the cancelled job must never run"


def test_a_running_job_stops_part_way(make_runner):
    """The job must stop early, not run all ten steps to the end."""
    pipeline = FakePipeline(steps=10, sleep=0.05)
    runner = make_runner(pipeline)

    job = runner.submit({})
    assert pipeline.started.wait(timeout=5), "the pipeline should be running"

    assert runner.cancel(job.id) is True
    assert wait_until(lambda: runner.get(job.id).status == CANCELLED, timeout=5)
    assert pipeline.steps_done_count() < 10, "it should not have finished"
    assert pipeline.ran() == [], "a cancelled job produces no result"


def test_a_cancelled_job_leaves_no_files(make_runner, tmp_path):
    pipeline = FakePipeline(steps=10, sleep=0.05)
    runner = make_runner(pipeline)

    job = runner.submit({})
    assert pipeline.started.wait(timeout=5)
    runner.cancel(job.id)

    assert wait_until(lambda: runner.get(job.id).status == CANCELLED, timeout=5)
    assert not (tmp_path / "jobs" / job.id).exists()


def test_cancelling_does_not_stop_the_worker(make_runner):
    """The queue must keep moving after a cancel."""
    pipeline = FakePipeline(steps=10, sleep=0.05)
    runner = make_runner(pipeline)

    first = runner.submit({})
    second = runner.submit({})
    assert pipeline.started.wait(timeout=5)
    runner.cancel(first.id)

    assert wait_until(lambda: runner.get(second.id).status == DONE, timeout=10)


def test_cancel_after_the_job_ended_changes_nothing(make_runner):
    runner = make_runner(FakePipeline())
    job = runner.submit({})

    assert wait_until(lambda: runner.get(job.id).status == DONE)
    assert runner.cancel(job.id) is False
    assert runner.get(job.id).status == DONE


def test_cancel_an_unknown_job_is_false(make_runner):
    runner = make_runner(FakePipeline())
    assert runner.cancel("no-such-job") is False


# -- through HTTP -----------------------------------------------------------


@pytest.fixture
def http(monkeypatch, tmp_path):
    def build(run_dub):
        monkeypatch.setattr("server.config.API_KEY", KEY)
        monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
        return TestClient(create_app(run_dub))

    return build


def post_dub(client):
    return client.post(
        "/dub",
        headers=AUTH,
        files={"video": ("clip.mp4", b"xxxx", "video/mp4")},
    ).json()["job_id"]


def status_of(client, job_id: str) -> str:
    return client.get(f"/jobs/{job_id}", headers=AUTH).json()["status"]


def test_delete_stops_a_running_job(http):
    pipeline = FakePipeline(steps=10, sleep=0.05)
    with http(pipeline) as client:
        job_id = post_dub(client)
        assert pipeline.started.wait(timeout=5)

        response = client.delete(f"/jobs/{job_id}", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["cancelling"] is True

        assert wait_until(lambda: status_of(client, job_id) == "cancelled", timeout=5)
        assert pipeline.steps_done_count() < 10


def test_the_result_of_a_cancelled_job_is_refused(http):
    pipeline = FakePipeline(steps=10, sleep=0.05)
    with http(pipeline) as client:
        job_id = post_dub(client)
        assert pipeline.started.wait(timeout=5)
        client.delete(f"/jobs/{job_id}", headers=AUTH)
        assert wait_until(lambda: status_of(client, job_id) == "cancelled", timeout=5)

        response = client.get(f"/jobs/{job_id}/result", headers=AUTH)
        assert response.status_code == 409


def test_deleting_twice_is_refused(http):
    pipeline = FakePipeline(steps=10, sleep=0.05)
    with http(pipeline) as client:
        job_id = post_dub(client)
        assert pipeline.started.wait(timeout=5)

        assert client.delete(f"/jobs/{job_id}", headers=AUTH).status_code == 200
        assert wait_until(lambda: status_of(client, job_id) == "cancelled", timeout=5)
        assert client.delete(f"/jobs/{job_id}", headers=AUTH).status_code == 409
