import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.tests.conftest import wait_until
from server.tests.fake_pipeline import FakePipeline, failing_pipeline

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A running app whose pipeline is fake, so no GPU is needed."""

    def build(run_dub=None, **settings):
        monkeypatch.setattr("server.config.API_KEY", KEY)
        monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
        for name, value in settings.items():
            monkeypatch.setattr(f"server.config.{name}", value)
        return TestClient(create_app(run_dub or FakePipeline()))

    return build


def a_video(name: str = "clip.mp4", size: int = 32):
    return {"video": (name, b"x" * size, "video/mp4")}


def post_dub(http, files=None, **fields):
    return http.post("/dub", headers=AUTH, files=files or a_video(), data=fields)


def wait_for_status(http, job_id: str, status: str, timeout: float = 5.0):
    def check():
        body = http.get(f"/jobs/{job_id}", headers=AUTH).json()
        return body["status"] == status

    return wait_until(check, timeout=timeout)


# -- auth -------------------------------------------------------------------


def test_no_token_is_rejected(client):
    with client() as http:
        assert http.get("/health").status_code == 401


def test_wrong_token_is_rejected(client):
    with client() as http:
        wrong = {"Authorization": "Bearer not-the-key"}
        assert http.get("/health", headers=wrong).status_code == 401


def test_right_token_passes(client):
    with client() as http:
        assert http.get("/health", headers=AUTH).status_code == 200


def test_no_api_key_means_no_start(monkeypatch, tmp_path):
    """Better to fail at boot than to run an API with no lock."""
    monkeypatch.setattr("server.config.API_KEY", "")
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    with pytest.raises(RuntimeError):
        with TestClient(create_app(FakePipeline())):
            pass


def test_schema_is_not_published(client):
    with client() as http:
        for path in ("/openapi.json", "/docs", "/redoc"):
            assert http.get(path, headers=AUTH).status_code == 404, path


# -- submit -----------------------------------------------------------------


def test_dub_returns_202_and_a_job_id(client):
    with client() as http:
        response = post_dub(http)
        assert response.status_code == 202
        assert response.json()["job_id"]


def test_job_runs_and_the_result_can_be_downloaded(client):
    with client() as http:
        job_id = post_dub(http).json()["job_id"]
        assert wait_for_status(http, job_id, "done")

        result = http.get(f"/jobs/{job_id}/result", headers=AUTH)
        assert result.status_code == 200
        assert result.content == b"fake video"

        # The job is forgotten once the file has been sent.
        assert wait_until(
            lambda: http.get(f"/jobs/{job_id}", headers=AUTH).status_code == 404
        )


def test_result_before_the_job_is_done_is_refused(client):
    with client(FakePipeline(sleep=0.3)) as http:
        job_id = post_dub(http).json()["job_id"]
        assert http.get(f"/jobs/{job_id}/result", headers=AUTH).status_code == 409


def test_unknown_job_is_404(client):
    with client() as http:
        assert http.get("/jobs/does-not-exist", headers=AUTH).status_code == 404
        assert http.delete("/jobs/does-not-exist", headers=AUTH).status_code == 404


# -- uploads ----------------------------------------------------------------


def test_a_file_type_we_cannot_read_is_refused(client):
    with client() as http:
        response = post_dub(http, files=a_video("clip.txt"))
        assert response.status_code == 415


def test_a_file_over_the_limit_is_refused(client):
    with client(MAX_VIDEO_BYTES=10) as http:
        response = post_dub(http, files=a_video(size=50))
        assert response.status_code == 413


def test_a_refused_upload_leaves_no_job_behind(client):
    with client(MAX_VIDEO_BYTES=10) as http:
        post_dub(http, files=a_video(size=50))
        # A second, good job must be the one that runs, with nothing stuck
        # in front of it.
        job_id = post_dub(http, files=a_video(size=5)).json()["job_id"]
        assert wait_for_status(http, job_id, "done")


# -- parameters -------------------------------------------------------------


def test_defaults_are_enough(client):
    with client() as http:
        assert post_dub(http).status_code == 202


def test_a_value_out_of_range_is_refused(client):
    with client() as http:
        assert post_dub(http, cfg_value=9.0).status_code == 422


def test_a_choice_outside_the_list_is_refused(client):
    with client() as http:
        assert post_dub(http, whisper_model="enormous").status_code == 422
        assert post_dub(http, vsr_mode="magic").status_code == 422


def test_a_backwards_scan_area_is_refused(client):
    with client() as http:
        assert post_dub(http, vsr_top=0.9, vsr_bottom=0.2).status_code == 422


# -- polling ----------------------------------------------------------------


def test_since_returns_only_new_log_lines(client):
    with client(FakePipeline(steps=3)) as http:
        job_id = post_dub(http).json()["job_id"]
        assert wait_for_status(http, job_id, "done")

        body = http.get(f"/jobs/{job_id}", headers=AUTH).json()
        assert body["log"] == ["step 1/3", "step 2/3", "step 3/3"]
        assert body["log_offset"] == 3

        later = http.get(f"/jobs/{job_id}?since=2", headers=AUTH).json()
        assert later["log"] == ["step 3/3"]


def test_queue_position_is_reported(client):
    with client(FakePipeline(sleep=0.3)) as http:
        first = post_dub(http).json()["job_id"]
        second = post_dub(http).json()["job_id"]

        assert wait_until(
            lambda: http.get(f"/jobs/{first}", headers=AUTH).json()["status"]
            == "running"
        )
        body = http.get(f"/jobs/{second}", headers=AUTH).json()
        assert body["status"] == "queued"
        assert body["queue_position"] == 0


def test_a_failed_job_reports_its_code(client):
    with client(failing_pipeline("no face in this clip", code="no_face")) as http:
        job_id = post_dub(http).json()["job_id"]
        assert wait_for_status(http, job_id, "failed")

        body = http.get(f"/jobs/{job_id}", headers=AUTH).json()
        assert body["error_code"] == "no_face"
        assert "no face" in body["error"]


# -- speak (text + reference audio → wav) -----------------------------------


def an_audio(name: str = "voice.wav", size: int = 32):
    return {"audio": (name, b"x" * size, "audio/wav")}


def post_speak(http, files=None, **fields):
    data = {"text": fields.pop("text", "xin chao"), **fields}
    return http.post("/speak", headers=AUTH, files=files or an_audio(), data=data)


def test_speak_returns_202_and_a_job_id(client):
    with client() as http:
        response = post_speak(http)
        assert response.status_code == 202
        assert response.json()["job_id"]


def test_speak_result_is_wav_and_the_job_is_forgotten(client):
    with client() as http:
        job_id = post_speak(http).json()["job_id"]
        assert wait_for_status(http, job_id, "done")

        result = http.get(f"/jobs/{job_id}/result", headers=AUTH)
        assert result.status_code == 200
        assert result.content == b"fake audio"
        assert result.headers["content-type"].startswith("audio/wav")

        assert wait_until(
            lambda: http.get(f"/jobs/{job_id}", headers=AUTH).status_code == 404
        )


def test_speak_without_text_is_refused(client):
    with client() as http:
        response = http.post(
            "/speak", headers=AUTH, files=an_audio(), data={"text": "  "},
        )
        assert response.status_code == 422


def test_speak_without_audio_is_refused(client):
    with client() as http:
        response = http.post("/speak", headers=AUTH, data={"text": "hello"})
        assert response.status_code == 422


def test_speak_rejects_a_video_as_audio(client):
    with client() as http:
        response = post_speak(http, files={"audio": ("clip.mp4", b"x" * 32, "video/mp4")})
        assert response.status_code == 415

# Cancelling has its own file: server/tests/test_cancel.py
