"""The subtitle-only branch: no new voice, only the picture.

Only the part that can go wrong quietly is tested here. Removing the old
subtitles, reading the speech and burning the new lines each have their own
test file; what is new is the decision to keep a job alive when Whisper
heard nobody speak.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from server import pipeline


class FakeContext:
    """Just enough of JobContext for the branch under test."""

    def __init__(self, workdir: Path, params):
        self.job_id = "test"
        self.workdir = workdir
        self.params = params
        self.logs: list[str] = []

    def log(self, message: str) -> None:
        self.logs.append(message)

    def step(self, name: str) -> None:
        self.logs.append(name)

    def check_cancel(self) -> None:
        pass


def params(**changes):
    base = dict(
        dub=False,
        remove_subtitle=False,
        vsr_mode="sttn-det",
        vsr_top=0.6, vsr_bottom=0.96, vsr_left=0.03, vsr_right=0.97,
        burn_subtitle=True,
        whisper_model="medium",
        subtitle_font="Noto Sans",
        subtitle_size=None,
        subtitle_position=None,
        hook_text="",
    )
    base.update(changes)
    return SimpleNamespace(**base)


def stub_reading(monkeypatch, cues):
    """Let the branch run without ffmpeg or a GPU."""
    monkeypatch.setattr(
        pipeline.audio, "extract_audio", lambda video, out: out)
    monkeypatch.setattr(pipeline.audio, "video_size", lambda video: (1080, 1920))
    monkeypatch.setattr(
        pipeline.transcribe, "transcribe",
        lambda models, wav, size, ctx=None: (cues, {}),
    )


def test_a_video_nobody_speaks_in_still_comes_back(monkeypatch, tmp_path):
    """Whisper heard nothing, so there is no text -- but the job is fine.

    Nothing could have known this before the expensive work ran, so the
    picture that was already made must not be thrown away.
    """
    stub_reading(monkeypatch, [])
    burned = []
    monkeypatch.setattr(
        pipeline.subtitle, "burn",
        lambda *a, **kw: burned.append(a) or Path("never"),
    )
    source = tmp_path / "video.mp4"
    source.write_bytes(b"v")

    ctx = FakeContext(tmp_path, params())
    result = pipeline._subtitle_only(ctx, pipeline.Models(None, None, object()))

    assert result == source
    assert not burned, "no lines means burning must be skipped, not attempted"
    assert any("nothing to burn" in line for line in ctx.logs), ctx.logs


def _stub_vsr(monkeypatch):
    """Record what the subtitle remover was given, and paint nothing."""
    seen = {}

    def fake_remove(video, out_path, *args, ctx=None, speech_cues=None):
        seen["speech_cues"] = speech_cues
        return video, None

    monkeypatch.setattr(pipeline.vsr, "remove_subtitles", fake_remove)
    return seen


def test_removing_only_still_hands_the_speech_to_the_remover(monkeypatch, tmp_path):
    """No burn asked for, but the remover still needs to know what was said."""
    stub_reading(monkeypatch, [{"start": 0.0, "end": 1.0, "text": "xin chào"}])
    seen = _stub_vsr(monkeypatch)
    (tmp_path / "video.mp4").write_bytes(b"v")

    ctx = FakeContext(tmp_path, params(remove_subtitle=True, burn_subtitle=False))
    pipeline._subtitle_only(ctx, pipeline.Models(None, None, object()))

    written = json.loads(seen["speech_cues"].read_text(encoding="utf-8"))
    assert written["cues"] == [{"start": 0.0, "end": 1.0, "text": "xin chào"}]


def test_removing_without_whisper_works_by_position_alone(monkeypatch, tmp_path):
    """A server with no Whisper must not fail a job that burns nothing."""
    stub_reading(monkeypatch, [])
    seen = _stub_vsr(monkeypatch)
    (tmp_path / "video.mp4").write_bytes(b"v")

    ctx = FakeContext(tmp_path, params(remove_subtitle=True, burn_subtitle=False))
    pipeline._subtitle_only(ctx, pipeline.Models(None, None, None))

    assert seen["speech_cues"] is None


def test_the_heard_lines_are_the_ones_burned(monkeypatch, tmp_path):
    stub_reading(monkeypatch, [
        {"start": 0.0, "end": 1.0, "text": "hello"},
        {"start": 1.0, "end": 2.0, "text": "   "},
    ])
    seen = {}

    def fake_burn(video, cues, out_path, width, height, **kw):
        seen["cues"] = cues
        out_path.write_bytes(b"subbed")
        return out_path

    monkeypatch.setattr(pipeline.subtitle, "burn", fake_burn)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"v")

    ctx = FakeContext(tmp_path, params())
    result = pipeline._subtitle_only(ctx, pipeline.Models(None, None, object()))

    assert result.name == "result_subbed.mp4"
    assert seen["cues"] == [{"start": 0.0, "end": 1.0, "text": "hello"}]
