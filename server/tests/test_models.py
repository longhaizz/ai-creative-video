"""Start-up model loading, checked without a GPU.

The real LipsyncModel needs torch, a 1.3 GB checkpoint and a card. What the
app has to get right is smaller than that: load every model once, before the
first request, and report the names on /health.
"""

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.tests.fake_pipeline import FakePipeline

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeModel:
    def __init__(self, name: str, fails: bool = False):
        self.name = name
        self.fails = fails
        self.load_count = 0

    def load(self):
        self.load_count += 1
        if self.fails:
            raise RuntimeError(f"{self.name} could not be loaded")


@pytest.fixture
def build(monkeypatch, tmp_path):
    def make(models):
        monkeypatch.setattr("server.config.API_KEY", KEY)
        monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
        return TestClient(create_app(FakePipeline(), models=models))

    return make


def test_as_list_skips_lipsync_when_none():
    from server.pipeline import Models

    models = Models(voice=FakeModel("voxcpm"), lipsync=None)
    assert [m.name for m in models.as_list()] == ["voxcpm"]


def test_as_list_keeps_lipsync_when_set():
    from server.pipeline import Models

    models = Models(
        voice=FakeModel("voxcpm"),
        lipsync=FakeModel("latentsync"),
    )
    assert [m.name for m in models.as_list()] == ["voxcpm", "latentsync"]


def test_as_list_includes_whisper():
    from server.pipeline import Models

    models = Models(
        voice=FakeModel("voxcpm"),
        lipsync=FakeModel("latentsync"),
        whisper=FakeModel("whisper"),
    )
    assert [m.name for m in models.as_list()] == ["voxcpm", "whisper", "latentsync"]


def test_no_models_is_fine(build):
    with build([]) as client:
        assert client.get("/health", headers=AUTH).json()["models_loaded"] == []


def test_models_are_named_on_health(build):
    with build([FakeModel("latentsync"), FakeModel("voxcpm")]) as client:
        body = client.get("/health", headers=AUTH).json()
        assert body["models_loaded"] == ["latentsync", "voxcpm"]


def test_each_model_is_loaded_once_for_many_requests(build):
    """Loading per request is the 30-60 second cost we are removing."""
    model = FakeModel("latentsync")
    with build([model]) as client:
        for _ in range(5):
            client.get("/health", headers=AUTH)
    assert model.load_count == 1


def test_a_model_that_fails_stops_the_server(build):
    """Better to die at boot than to accept jobs we cannot run."""
    with pytest.raises(RuntimeError):
        with build([FakeModel("latentsync", fails=True)]):
            pass


def test_a_broken_model_stops_the_ones_after_it(build):
    later = FakeModel("voxcpm")
    with pytest.raises(RuntimeError):
        with build([FakeModel("latentsync", fails=True), later]):
            pass
    assert later.load_count == 0
