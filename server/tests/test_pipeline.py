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
        translate_screen_text=False,
        target_lang="same",
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

    def fake_remove(video, out_path, *args, ctx=None, speech_cues=None,
                    screen_text=None, detect_only=False):
        seen["speech_cues"] = speech_cues
        seen["screen_text"] = screen_text
        seen["detect_only"] = detect_only
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
    assert "Heard () 0.00-1.00s: xin chào" in ctx.logs, ctx.logs


def test_nothing_heard_still_tells_the_remover_to_filter(monkeypatch, tmp_path):
    """An empty file, not no file: the remover removes nothing, not everything."""
    stub_reading(monkeypatch, [])
    seen = _stub_vsr(monkeypatch)
    (tmp_path / "video.mp4").write_bytes(b"v")

    ctx = FakeContext(tmp_path, params(remove_subtitle=True, burn_subtitle=False))
    pipeline._subtitle_only(ctx, pipeline.Models(None, None, object()))

    written = json.loads(seen["speech_cues"].read_text(encoding="utf-8"))
    assert written["cues"] == []


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


# -- translating the text printed on the picture ----------------------------


def _fake_translate(monkeypatch, out=None):
    """Answer the translate step without an OpenAI key."""
    seen = {}

    def fake_labels(texts, target_lang, api_key, asr_meta=None, ctx=None, **kw):
        seen["texts"] = list(texts)
        seen["target_lang"] = target_lang
        return out if out is not None else [f"[{t}]" for t in texts]

    import server.steps.translate as translate
    monkeypatch.setattr(translate, "translate_labels", fake_labels)
    return seen


def _write_screen(path, pieces):
    path.write_text(json.dumps(pieces), encoding="utf-8")


def test_the_remover_is_told_where_to_write_the_screen_text(monkeypatch, tmp_path):
    stub_reading(monkeypatch, [{"start": 0.0, "end": 1.0, "text": "xin chào"}])
    seen = _stub_vsr(monkeypatch)
    (tmp_path / "video.mp4").write_bytes(b"v")

    ctx = FakeContext(tmp_path, params(
        remove_subtitle=True, burn_subtitle=False,
        translate_screen_text=True, target_lang="VI"))
    pipeline._subtitle_only(ctx, pipeline.Models(None, None, object()))

    assert seen["screen_text"] == tmp_path / "screen_text.json"
    assert seen["detect_only"] is False


def test_without_removing_the_subtitles_the_video_is_only_read(monkeypatch, tmp_path):
    """The old subtitles must stay exactly where they are."""
    stub_reading(monkeypatch, [{"start": 0.0, "end": 1.0, "text": "xin chào"}])
    seen = _stub_vsr(monkeypatch)
    (tmp_path / "video.mp4").write_bytes(b"v")

    ctx = FakeContext(tmp_path, params(
        remove_subtitle=False, burn_subtitle=False,
        translate_screen_text=True, target_lang="VI"))
    pipeline._subtitle_only(ctx, pipeline.Models(None, None, object()))

    assert seen["detect_only"] is True
    assert seen["screen_text"] == tmp_path / "screen_text.json"


def test_the_same_language_asks_for_no_reading_at_all(monkeypatch, tmp_path):
    """Writing the same words back would only print the OCR mistakes."""
    stub_reading(monkeypatch, [{"start": 0.0, "end": 1.0, "text": "xin chào"}])
    seen = _stub_vsr(monkeypatch)
    (tmp_path / "video.mp4").write_bytes(b"v")

    ctx = FakeContext(tmp_path, params(
        remove_subtitle=True, burn_subtitle=False,
        translate_screen_text=True, target_lang="same"))
    pipeline._subtitle_only(ctx, pipeline.Models(None, None, object()))

    assert seen["screen_text"] is None


def test_the_found_text_reaches_the_burner_translated(monkeypatch, tmp_path):
    seen_translate = _fake_translate(monkeypatch)
    _write_screen(tmp_path / "screen_text.json",
                  [{"text": "SALE 50%", "box": [100, 300, 200, 260],
                    "start": 1.0, "end": 3.0}])

    ctx = FakeContext(tmp_path, params(
        translate_screen_text=True, target_lang="VI"))
    screen = pipeline._translated_screen_text(
        tmp_path / "screen_text.json", ctx.params, {}, 1080, 1920, ctx)

    assert seen_translate["texts"] == ["SALE 50%"]
    assert [p["text"] for p in screen] == ["[SALE 50%]"]
    assert screen[0]["box"] == [100, 300, 200, 260]


def test_text_in_the_hook_box_is_left_to_the_hook(monkeypatch, tmp_path):
    """The client typed what that box says; a translation must not fight it."""
    _fake_translate(monkeypatch)
    _write_screen(tmp_path / "screen_text.json", [
        {"text": "OLD HOOK", "box": [200, 800, 200, 300], "start": 0.0, "end": 5.0},
        {"text": "SALE 50%", "box": [100, 300, 1200, 1260], "start": 1.0, "end": 3.0},
    ])

    ctx = FakeContext(tmp_path, params(
        translate_screen_text=True, target_lang="VI",
        hook_text="Mua ngay", hook_top=0.05, hook_bottom=0.20,
        hook_left=0.10, hook_right=0.90))
    screen = pipeline._translated_screen_text(
        tmp_path / "screen_text.json", ctx.params, {}, 1080, 1920, ctx)

    assert [p["text"] for p in screen] == ["[SALE 50%]"]


def test_a_translation_failure_costs_the_text_and_not_the_job(monkeypatch, tmp_path):
    import server.steps.translate as translate

    def boom(*a, **kw):
        raise pipeline.PipelineError("OpenAI is down")

    monkeypatch.setattr(translate, "translate_labels", boom)
    _write_screen(tmp_path / "screen_text.json",
                  [{"text": "SALE", "box": [1, 2, 3, 4], "start": 0.0, "end": 1.0}])

    ctx = FakeContext(tmp_path, params(translate_screen_text=True, target_lang="VI"))
    assert pipeline._translated_screen_text(
        tmp_path / "screen_text.json", ctx.params, {}, 1080, 1920, ctx) == []


def test_a_missing_dump_is_not_a_failure(tmp_path):
    ctx = FakeContext(tmp_path, params(translate_screen_text=True, target_lang="VI"))
    assert pipeline._translated_screen_text(
        tmp_path / "never_written.json", ctx.params, {}, 1080, 1920, ctx) == []
