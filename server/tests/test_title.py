"""A ticked box names the download from the translated speech."""

import json
from types import SimpleNamespace

from server import pipeline
from server.steps import llm, title

GOOD = "Kem chống nắng giúp da hết thâm chỉ sau bảy ngày dùng"


def _answers(*texts):
    pending = list(texts)

    def fake_ask(system, user, api_key, **_kw):
        fake_ask.calls.append({"system": system, "user": user})
        return pending.pop(0)

    fake_ask.calls = []
    return fake_ask


def test_a_title_in_range_is_kept_with_its_diacritics(monkeypatch):
    monkeypatch.setattr(title.llm, "ask", _answers(GOOD))
    assert title.content_title("Mua kem này ngay hôm nay", "VI", "sk") == GOOD


def test_illegal_characters_and_an_extension_are_removed(monkeypatch):
    raw = 'Kem chống nắng: giúp da "hết thâm"\nchỉ sau bảy ngày.mp4'
    monkeypatch.setattr(title.llm, "ask", _answers(raw))
    assert title.content_title("script", "vi", "sk") == (
        "Kem chống nắng giúp da hết thâm chỉ sau bảy ngày")


def test_forty_and_eighty_characters_are_kept(monkeypatch):
    for count in (40, 80):
        text = "a" * count
        monkeypatch.setattr(title.llm, "ask", _answers(text))
        assert title.content_title("script", "vi", "sk") == text


def test_a_title_outside_the_range_is_asked_once_more(monkeypatch):
    ask = _answers("qua ngan", GOOD)
    monkeypatch.setattr(title.llm, "ask", ask)
    logs = []
    assert title.content_title(
        "Mua kem này", "vi", "sk", log=logs.append) == GOOD
    assert len(ask.calls) == 2
    assert "40" in ask.calls[1]["user"] and "80" in ask.calls[1]["user"]
    assert any("asking once more" in line for line in logs)


def test_a_second_miss_keeps_the_original_name(monkeypatch):
    ask = _answers("n", "n")
    monkeypatch.setattr(title.llm, "ask", ask)
    logs = []
    assert title.content_title("script", "vi", "sk", log=logs.append) is None
    assert len(ask.calls) == 2
    assert any("keeping the original file name" in line for line in logs)


def test_an_api_failure_is_not_retried(monkeypatch):
    calls = {"n": 0}

    def fake_ask(*_a, **_k):
        calls["n"] += 1
        raise llm.LLMError("OpenAI HTTP 429: slow")

    monkeypatch.setattr(title.llm, "ask", fake_ask)
    logs = []
    assert title.content_title("script", "vi", "sk", log=logs.append) is None
    assert calls["n"] == 1
    assert any("429" in line for line in logs)


def test_empty_speech_is_not_sent_to_the_model(monkeypatch):
    def fake_ask(*_a, **_k):
        raise AssertionError("the model was asked")

    monkeypatch.setattr(title.llm, "ask", fake_ask)
    assert title.content_title("   ", "vi", "sk") is None


def test_same_language_uses_the_language_whisper_heard(monkeypatch):
    ask = _answers(GOOD)
    monkeypatch.setattr(title.llm, "ask", ask)
    title.content_title(
        "buy this cream", "same", "sk", asr_meta={"language": "en"})
    assert "English" in ask.calls[0]["system"]
    assert "buy this cream" in ask.calls[0]["user"]


class _Ctx:
    def __init__(self, **params):
        self.params = SimpleNamespace(**params)
        self.logs: list[str] = []
        self.output_name = None

    def log(self, message: str) -> None:
        self.logs.append(message)

    def set_output_name(self, name: str) -> None:
        self.output_name = name


def test_the_title_comes_from_the_lines_the_viewer_hears(monkeypatch, tmp_path):
    seen = {}

    def fake(speech, target_lang, api_key, asr_meta=None, log=None):
        seen["speech"] = speech
        seen["lang"] = target_lang
        seen["meta"] = asr_meta
        return GOOD

    monkeypatch.setattr(pipeline, "content_title", fake)
    (tmp_path / "spoken_cues.json").write_text(json.dumps([
        {"text": "Kem chống nắng này"},
        {"text": "   "},
        {"text": "giúp da hết thâm."},
    ]), encoding="utf-8")
    ctx = _Ctx(name_from_content=True, dub=True, target_lang="vi")
    pipeline._maybe_name_output(ctx, tmp_path, {"language": "zh"})
    assert seen["speech"] == "Kem chống nắng này giúp da hết thâm."
    assert seen["lang"] == "vi"
    assert seen["meta"]["language"] == "zh"
    assert ctx.output_name == GOOD
    assert ctx.logs[-1] == f"Output file name: {GOOD}"


def test_the_box_left_unticked_does_not_rename(monkeypatch, tmp_path):
    def fake(*_a, **_k):
        raise AssertionError("the model was asked")

    monkeypatch.setattr(pipeline, "content_title", fake)
    ctx = _Ctx(name_from_content=False, dub=True, target_lang="vi")
    pipeline._maybe_name_output(ctx, tmp_path, {})
    assert ctx.output_name is None
    assert ctx.logs == []


def test_no_translated_speech_keeps_the_original_name(tmp_path):
    ctx = _Ctx(name_from_content=True, dub=True, target_lang="vi")
    pipeline._maybe_name_output(ctx, tmp_path, {})
    assert ctx.output_name is None
    assert "no translated speech" in ctx.logs[0]


def test_a_job_that_does_not_dub_keeps_the_original_name(tmp_path):
    ctx = _Ctx(name_from_content=True, dub=False, target_lang="vi")
    pipeline._maybe_name_output(ctx, tmp_path, {})
    assert ctx.output_name is None
    assert "does not dub" in ctx.logs[0]


def test_a_naming_bug_does_not_fail_the_job(monkeypatch, tmp_path):
    def boom(*_a, **_k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(pipeline, "content_title", boom)
    (tmp_path / "spoken_cues.json").write_text(
        json.dumps([{"text": "xin chào mua ngay hôm nay kẻo hết hàng"}]),
        encoding="utf-8")
    ctx = _Ctx(name_from_content=True, dub=True, target_lang="vi")
    pipeline._maybe_name_output(ctx, tmp_path, {})
    assert ctx.output_name is None
    assert any("keeping the original file name" in line for line in ctx.logs)

