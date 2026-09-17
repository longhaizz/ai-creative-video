"""The one place a language model is asked, for either provider."""

import base64

import pytest

from server import config
from server.steps import llm


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400
        self.text = str(body)

    def json(self):
        return self._body


def _record(monkeypatch, *responses):
    sent = []
    answers = list(responses)

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.append({"url": url, "json": json, "headers": headers})
        return answers.pop(0)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return sent


SCHEMA = {"name": "x", "strict": True, "schema": {"type": "object"}}
OPENAI_OK = FakeResponse(200, {"choices": [{"message": {"content": " ok "}}]})
GEMINI_OK = FakeResponse(200, {"candidates": [{"content": {"parts": [
    {"text": "o"}, {"text": "k"}]}}]})


def test_openai_is_the_default(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    sent = _record(monkeypatch, OPENAI_OK)

    assert llm.ask("sys", "user", "sk-1", schema=SCHEMA,
                   images=[("picture of piece 0", b"png")]) == "ok"
    body = sent[0]["json"]
    assert sent[0]["url"] == llm.OPENAI_URL
    assert sent[0]["headers"]["Authorization"] == "Bearer sk-1"
    assert body["model"] == "gpt-4o-mini"
    assert body["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
    content = body["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "user"}
    assert content[1] == {"type": "text", "text": "picture of piece 0"}
    assert content[2]["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"png").decode())


def test_openai_without_pictures_sends_plain_text(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    sent = _record(monkeypatch, OPENAI_OK)
    llm.ask("sys", "user", "sk-1", model="gpt-x", json_mode=True)
    body = sent[0]["json"]
    assert body["messages"][1]["content"] == "user"
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "gpt-x"


def test_gemini_gets_the_same_ask_in_its_own_shape(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "LLM_MODEL", "")
    sent = _record(monkeypatch, GEMINI_OK)

    assert llm.ask("sys", "user", "AIza", schema=SCHEMA,
                   images=[("picture of piece 0", b"png")]) == "ok"
    body = sent[0]["json"]
    assert "gemini-2.5-flash:generateContent" in sent[0]["url"]
    assert sent[0]["headers"]["x-goog-api-key"] == "AIza"
    assert body["system_instruction"] == {"parts": [{"text": "sys"}]}
    parts = body["contents"][0]["parts"]
    assert parts[:2] == [{"text": "user"}, {"text": "picture of piece 0"}]
    assert parts[2]["inline_data"]["data"] == base64.b64encode(b"png").decode()
    config_ = body["generationConfig"]
    assert config_["responseMimeType"] == "application/json"
    assert config_["responseJsonSchema"] == {"type": "object"}


def test_gemini_is_asked_again_without_a_schema_it_refused(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    sent = _record(monkeypatch, FakeResponse(400, {"error": "schema"}), GEMINI_OK)

    assert llm.ask("sys", "user", "AIza", schema=SCHEMA) == "ok"
    assert "responseJsonSchema" not in sent[1]["json"]["generationConfig"]
    assert sent[1]["json"]["generationConfig"]["responseMimeType"] == "application/json"


def test_a_refusal_names_the_provider_and_status(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    _record(monkeypatch, FakeResponse(429, {"error": "slow"}))
    with pytest.raises(llm.LLMError, match="Gemini HTTP 429"):
        llm.ask("sys", "user", "AIza")


def test_no_key_for_the_chosen_provider_says_which(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.ask("sys", "user", "")


def test_an_unknown_provider_is_refused(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "claude")
    with pytest.raises(llm.LLMError, match="LLM_PROVIDER"):
        llm.ask("sys", "user", "key")
