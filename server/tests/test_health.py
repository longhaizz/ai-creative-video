import pytest
from fastapi.testclient import TestClient

from server.app import app


def test_health_ok(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.API_KEY", "test-key")
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    assert (tmp_path / "jobs").is_dir(), "lifespan phải tạo JOBS_DIR"


def test_khong_co_api_key_thi_khong_khoi_dong(monkeypatch, tmp_path):
    """Thà chết lúc boot còn hơn chạy một API không khoá cửa."""
    monkeypatch.setattr("server.config.API_KEY", "")
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_khong_phoi_openapi(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.API_KEY", "test-key")
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    with TestClient(app) as client:
        for path in ("/openapi.json", "/docs", "/redoc"):
            assert client.get(path).status_code == 404, path
