"""The cap on how big a request may be.

save_upload() also counts bytes, but that check happens after Starlette has
already read and spooled the whole body, and after the token is checked.
Measured on a running server: an unauthenticated 60 MB upload sent all
60 MB, and the temp folder grew by all of it before the 401 came back. So
anyone who could reach the port could fill the disk with no token at all.

These tests are about the earlier check, the one that runs first.
"""

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from server.tests.fake_pipeline import FakePipeline

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
LIMIT = 4096


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.API_KEY", KEY)
    monkeypatch.setattr("server.config.JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr("server.config.MAX_REQUEST_BYTES", LIMIT)
    with TestClient(create_app(FakePipeline())) as http:
        yield http


def upload(http, size: int, headers=AUTH):
    return http.post(
        "/dub",
        headers=headers,
        files={"video": ("clip.mp4", b"x" * size, "video/mp4")},
    )


def test_a_request_within_the_limit_is_accepted(client):
    assert upload(client, 100).status_code == 202


def test_a_request_over_the_limit_is_refused(client):
    response = upload(client, LIMIT * 2)
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_the_limit_applies_without_a_token(client):
    """This is the point of the whole file.

    A caller with no token must not be able to write to our disk. The token
    check happens after the body is read, so the size check has to happen
    before both.
    """
    response = upload(client, LIMIT * 2, headers={})
    assert response.status_code == 413, "size first, then the token"


def test_a_wrong_token_is_also_stopped_early(client):
    response = upload(client, LIMIT * 2, headers={"Authorization": "Bearer nope"})
    assert response.status_code == 413


def test_no_job_is_left_behind_by_a_refused_request(client):
    upload(client, LIMIT * 2)
    # A good request afterwards must run, with nothing stuck in front of it.
    assert upload(client, 100).status_code == 202


def test_small_requests_are_untouched(client):
    assert client.get("/health", headers=AUTH).status_code == 200
    assert client.get("/jobs/nope", headers=AUTH).status_code == 404


def test_a_lying_content_length_does_not_get_through(client):
    """The header is written by the caller, so it cannot be the only check.

    Sending more bytes than the header claims must still be stopped, by the
    counter that watches the body as it streams.
    """
    body = b"x" * (LIMIT * 2)
    response = client.post(
        "/dub",
        headers={**AUTH, "Content-Length": "10", "Content-Type": "video/mp4"},
        content=body,
    )
    assert response.status_code >= 400
    assert response.status_code < 500
