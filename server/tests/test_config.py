"""Paths in config must not move when lip sync changes the directory."""

import importlib

import server.config


def test_paths_stay_put_after_chdir(tmp_path, monkeypatch):
    monkeypatch.delenv("JOBS_DIR", raising=False)
    config = importlib.reload(server.config)
    before = config.JOBS_DIR

    # The same thing lipsync._inside() does while an upload comes in.
    monkeypatch.chdir(tmp_path)

    for path in (config.JOBS_DIR, config.DURATION_DATA,
                 config.VSR_DIR, config.LATENTSYNC_DIR):
        assert path.is_absolute(), path
    assert config.JOBS_DIR.resolve() == before
