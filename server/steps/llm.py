"""One way to ask a language model, whichever company answers.

LLM_PROVIDER picks it: "openai" (the default) or "gemini". Both are called
over plain HTTP, so no SDK is needed. ask() is the only thing the rest of
the server calls; retrying is left to the caller (translate._chat).

A picture goes along as PNG bytes with a short caption, so the model knows
which piece of the request it belongs to.
"""

from __future__ import annotations

import base64

import requests

from server import config
from server.jobs import PipelineError

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "gemini": "gemini-2.5-flash"}
TIMEOUT = 120
TEMPERATURE = 0.1


class LLMError(PipelineError):
    """A failure the user should see, with the standard error code.

    The message starts with "<Provider> HTTP <status>" when the server said
    no, so the caller can tell a rate limit from a bad request.
    """

    def __init__(self, message: str):
        super().__init__(message, code="internal")


def ask(system: str, user: str, api_key: str, model: str = "",
        json_mode: bool = False, schema: dict | None = None,
        images: list[tuple[str, bytes]] = ()) -> str:
    """Ask the chosen model once and return its answer as text.

    schema is the OpenAI json_schema shape: {"name", "strict", "schema"}.
    images are (caption, png bytes) pairs, sent after the user text.
    """
    provider = config.LLM_PROVIDER
    if provider not in DEFAULT_MODELS:
        raise LLMError(f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, "
                       f"not {provider!r}")
    key = (api_key or "").strip()
    if not key:
        raise LLMError(f"Chưa có API key cho {provider} "
                       f"({provider.upper()}_API_KEY)")
    model = model or config.LLM_MODEL or DEFAULT_MODELS[provider]
    send = _openai if provider == "openai" else _gemini
    out = send(system, user, key, model, json_mode, schema, list(images))
    out = (out or "").strip().strip('"').strip("'")
    if not out:
        raise LLMError(f"{provider} trả text rỗng")
    return out


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _openai(system, user, key, model, json_mode, schema, images) -> str:
    content = user
    if images:
        content = [{"type": "text", "text": user}]
        for caption, png in images:
            content.append({"type": "text", "text": caption})
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{_b64(png)}"}})
    body = {
        "model": model,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": schema}
    elif json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(OPENAI_URL, json=body, timeout=TIMEOUT, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    if not r.ok:
        raise LLMError(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
    try:
        return r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise LLMError(f"OpenAI response lạ: {r.text[:300]}") from e


def _gemini(system, user, key, model, json_mode, schema, images) -> str:
    parts = [{"text": user}]
    for caption, png in images:
        parts.append({"text": caption})
        parts.append({"inline_data": {"mime_type": "image/png", "data": _b64(png)}})
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": TEMPERATURE},
    }
    if json_mode or schema is not None:
        body["generationConfig"]["responseMimeType"] = "application/json"
    if schema is not None:
        body["generationConfig"]["responseJsonSchema"] = schema["schema"]
    r = _gemini_post(model, key, body)
    if r.status_code == 400 and schema is not None:
        # ponytail: a schema Gemini will not take is dropped, not translated;
        # the prompt still names the keys and the caller checks the JSON
        del body["generationConfig"]["responseJsonSchema"]
        r = _gemini_post(model, key, body)
    if not r.ok:
        raise LLMError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    try:
        found = r.json()["candidates"][0]["content"]["parts"]
        return "".join(part.get("text", "") for part in found)
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise LLMError(f"Gemini response lạ: {r.text[:300]}") from e


def _gemini_post(model, key, body):
    return requests.post(
        GEMINI_URL.format(model=model), json=body, timeout=TIMEOUT,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
    )
