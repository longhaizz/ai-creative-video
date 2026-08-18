import pytest
from fastapi.testclient import TestClient

from server.app import app


def test_health_returns_ok(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.API_KEY", "test-key")
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    assert (tmp_path / "jobs").is_dir(), "lifespan must create JOBS_DIR"


def test_no_api_key_means_no_start(monkeypatch, tmp_path):
    """Better to fail at boot than to run an API with no lock."""
    monkeypatch.setattr("server.config.API_KEY", "")
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_schema_is_not_published(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.API_KEY", "test-key")
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    with TestClient(app) as client:
        for path in ("/openapi.json", "/docs", "/redoc"):
            assert client.get(path).status_code == 404, path
