import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from server.app import create_app
from server.jobs import JobContext
from server.tests.conftest import wait_until

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeClonePipeline:
    """A clone-only fake pipeline: always writes one WAV result."""

    def __call__(self, context: JobContext) -> Path:
        result = context.workdir / "result.wav"
        result.write_bytes(b"fake audio")
        return result


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.API_KEY", KEY)
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    return TestClient(create_app(FakeClonePipeline()))


def post_speak(http, files=None, **fields):
    return http.post(
        "/speak",
        headers=AUTH,
        files=files or {
            "audio": ("ref.mp3", b"x" * 10, "audio/mpeg"),
        },
        data=fields,
    )


def wait_for_status(http, job_id: str, status: str, timeout: float = 5.0):
    def check():
        body = http.get(f"/jobs/{job_id}", headers=AUTH).json()
        return body["status"] == status

    return wait_until(check, timeout=timeout)


def test_speak_returns_202_and_a_job_id(client):
    with client as http:
        response = post_speak(http, text="Xin chào")
        assert response.status_code == 202
        assert response.json()["job_id"]


def test_speak_job_runs_and_result_can_be_downloaded(client):
    with client as http:
        job_id = post_speak(http, text="Xin chào").json()["job_id"]
        assert wait_for_status(http, job_id, "done")

        result = http.get(f"/jobs/{job_id}/result", headers=AUTH)
        assert result.status_code == 200
        assert result.content == b"fake audio"
        assert result.headers["content-type"].startswith("audio/")

        # The job is forgotten once the file has been sent.
        assert wait_until(
            lambda: http.get(f"/jobs/{job_id}", headers=AUTH).status_code == 404
        )


def test_speak_refuses_unreadable_file_type(client):
    with client as http:
        response = post_speak(
            http,
            text="Xin chào",
            files={"audio": ("ref.txt", b"x" * 10, "text/plain")},
        )
        assert response.status_code == 415


def test_speak_refuses_blank_text(client):
    with client as http:
        response = post_speak(
            http,
            text="   ",
            files={"audio": ("ref.mp3", b"x" * 10, "audio/mpeg")},
        )
        assert response.status_code == 422


def test_speak_accepts_mp4_container_audio(client):
    with client as http:
        response = post_speak(
            http,
            text="Xin chào",
            files={"audio": ("ref.mp4", b"x" * 10, "video/mp4")},
        )
        assert response.status_code == 202


def test_clone_does_not_convert_wav_onto_itself(monkeypatch, tmp_path):
    """A .wav upload must not be ffmpeg's input and output at once."""
    from server import pipeline
    from server.pipeline import Models
    from server.schemas import CloneParams

    (tmp_path / "reference_audio.wav").write_bytes(b"u")
    seen = []

    def fake_extract(src, out):
        seen.append((Path(src).resolve(), Path(out).resolve()))
        Path(out).write_bytes(b"pcm")
        return Path(out)

    monkeypatch.setattr(pipeline.audio, "extract_audio", fake_extract)

    class Voice:
        def speak(self, text, out_wav, *a, **kw):
            Path(out_wav).write_bytes(b"wav")
            return Path(out_wav)

    class Ctx:
        params = CloneParams(text="hi")
        workdir = tmp_path

        def step(self, *a, **kw):
            pass

        def log(self, *a, **kw):
            pass

        def check_cancel(self):
            pass

    pipeline._clone(Ctx(), Models(voice=Voice(), lipsync=None))
    assert seen and seen[0][0] != seen[0][1]
