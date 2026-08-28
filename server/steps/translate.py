"""Rewrite and translate the transcript before the voice is made.

Copied almost unchanged from spy-ads openai_translate_api.py. It is worth
keeping as it is: the prompts and the repair passes here were tuned against
real ad transcripts, and rewriting them would quietly lose that work.

Only two things changed:
  * OpenAIError now sits under PipelineError, so a failure carries an
    error_code back to the client like every other step;
  * the key comes from the server environment, not from the desktop app, so
    it never ships inside a .exe.

The comments below are still the original Vietnamese.
"""

from __future__ import annotations

import json
import re

import requests

from server.jobs import PipelineError

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"

# Mã UI → tên ngôn ngữ cho prompt
LANG_NAMES = {
    "AR": "Arabic",
    "DA": "Danish",
    "NL": "Dutch",
    "EN": "English",
    "FI": "Finnish",
    "FR": "French",
    "DE": "German",
    "EL": "Greek",
    "HE": "Hebrew",
    "HI": "Hindi",
    "ID": "Indonesian",
    "IT": "Italian",
    "JA": "Japanese",
    "KO": "Korean",
    "NB": "Norwegian",
    "PL": "Polish",
    "PT": "Portuguese",
    "RU": "Russian",
    "ES": "Spanish",
    "SV": "Swedish",
    "TH": "Thai",
    "TL": "Filipino (Tagalog)",
    "TR": "Turkish",
    "VI": "Vietnamese",
    "ZH": "Chinese",
}

# The three lengths every block comes back in. The caller picks the one
# that fits the room it has; there is no asking again for a shorter line.
VARIANTS = ("short", "normal", "long")


class OpenAIError(PipelineError):
    """A failure the user should see, with the standard error code."""

    def __init__(self, message: str):
        super().__init__(message, code="internal")


def word_count(text: str) -> int:
    return len((text or "").split())


def _chat(system: str, user: str, api_key: str, model: str) -> str:
    key = (api_key or "").strip()
    if not key:
        raise OpenAIError("Chưa có OpenAI API key")
    r = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=120,
    )
    if not r.ok:
        raise OpenAIError(f"OpenAI HTTP {r.status_code}: {r.text[:300]}")
    try:
        out = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise OpenAIError(f"OpenAI response lạ: {r.text[:300]}") from e
    out = (out or "").strip().strip('"').strip("'")
    if not out:
        raise OpenAIError("OpenAI trả text rỗng")
    return out


# Chữ có dấu thanh Việt (heuristic phát hiện drift VI ↔ Latin khác).
_VI_MARKS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)


def _vi_mark_ratio(text: str) -> float:
    """Tỉ lệ chữ cái mang dấu Việt; text quá ngắn → 0."""
    letters = [c for c in (text or "") if c.isalpha()]
    if len(letters) < 8:
        return 0.0
    return sum(1 for c in letters if c in _VI_MARKS) / len(letters)


def _normalize_lang_code(code: str) -> str:
    c = (code or "").strip().lower()
    if c in ("vi", "vietnamese"):
        return "vi"
    if c in ("id", "in", "indonesian"):
        return "id"
    return c


def _resolve_output_lang(target_lang: str, asr_meta=None) -> tuple:
    """Trả (expected_code, lang_name, same_mode).

    same → map Whisper language (vd. id) sang LANG_NAMES.
    """
    asr_meta = asr_meta or {}
    raw = (target_lang or "same").strip().lower()
    if raw == "same":
        det = (asr_meta.get("language") or "").strip().lower()
        code = _normalize_lang_code(det) or "unknown"
        key = code.upper()
        lang_name = LANG_NAMES.get(key, det or "the detected source language")
        return code, lang_name, True
    code = _normalize_lang_code(raw)
    key = code.upper()
    lang_name = LANG_NAMES.get(key, key)
    return code, lang_name, False


def _lines_wrong_language(cues, expected_code: str) -> list:
    """Indices cue lệch ngôn ngữ so với expected (heuristic dấu Việt).

    - expected vi: dòng Latin dài gần như không dấu → nghi không phải VI
    - expected khác vi (id/en/...): mật độ dấu Việt cao → nghi nhảy sang VI
    """
    expected = _normalize_lang_code(expected_code)
    bad = []
    for i, t in enumerate(cues or []):
        text = (t or "").strip()
        if not text:
            continue
        letters = [c for c in text if c.isalpha()]
        if len(letters) < 10:
            continue
        ratio = _vi_mark_ratio(text)
        if expected == "vi":
            if ratio < 0.04:
                bad.append(i)
        else:
            # Non-VI output: Vietnamese diacritics strongly suggest drift
            if ratio >= 0.08:
                bad.append(i)
    return bad


def _extract_json(raw: str) -> dict:
    """Lấy object JSON đầu tiên từ response (có thể có ```fence)."""
    import json
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise OpenAIError(f"OpenAI không trả JSON: {text[:200]}")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise OpenAIError(f"OpenAI JSON lỗi: {e}") from e


def _blocks_system_prompt(*, n: int, lang_name: str, expected_code: str,
                          lang_det: str, lang_p: float, task: str) -> str:
    """Prompt for block translation: three lengths, one line per block."""
    return (
        f"You write spoken dubbing lines in {lang_name} (code={expected_code}). "
        f"The input is an ASR transcript cut into {n} blocks. A block is one "
        f"run of speech between two real pauses, so it is what a person says "
        f"in one breath. Detected source language: {lang_det} "
        f"(p={lang_p:.2f}). "
        f"The ASR may contain phonetic errors, duplicated words and mixed "
        f"languages. First infer the intended message from the WHOLE "
        f"transcript, then write each block. "
        f"{task} "
        f"HARD RULE: return exactly {n} entries, one per block, in order. "
        f"Entry i is spoken while block i plays. Never merge two blocks into "
        f"one entry, never move an entry to another index, never drop one. "
        f"EVERY block was built from real speech, so EVERY block must have "
        f"words. An empty string is never a valid answer: when the ASR of a "
        f"block is unclear, write what the speaker must have been saying "
        f"there, from the surrounding blocks. "
        f"THREE LENGTHS: each entry has \"short\", \"normal\" and \"long\". "
        f"All three say the same thing for that block; only the wording is "
        f"tighter or fuller. The block header gives a word count for "
        f"\"normal\": aim about 60% of it for \"short\" and about 130% for "
        f"\"long\". \"long\" fills the extra room by restating, adding a "
        f"natural connector, or naming again what is being talked about — "
        f"and it may lean on something ALREADY SAID in an EARLIER block. "
        f"It must never use an idea from a LATER block, and never repeat an "
        f"earlier line word for word. "
        f"Do not invent products, prices, numbers, names or calls to action "
        f"that are not in the transcript — not even in \"long\". "
        f"Everything must be in {lang_name} only, never mixed. "
        f"Return ONLY valid JSON (no markdown) with keys: master_meaning "
        f"(one sentence, in English), master_translation (the full spoken "
        f"script in {lang_name}, using the \"normal\" lines), lines (array "
        f"of exactly {n} objects with keys short, normal, long)."
    )


def translate_blocks(blocks, target_lang: str, api_key: str,
                     asr_meta=None, model: str = DEFAULT_MODEL) -> dict:
    """Translate whole blocks, three lengths each, one line per block.

    Blocks are built from word timestamps by the caller, so the mapping
    from text to time is decided by code, not by the model. The model keeps
    the order and the count, and both are checked here. It offers three
    lengths; the caller picks the one that fits the room it has, which is
    what replaces asking again for a shorter line.
    """
    blocks = list(blocks or [])
    if not blocks:
        raise OpenAIError("Khong co block nao de dich")
    asr_meta = asr_meta or {}
    expected_code, lang_name, same_mode = _resolve_output_lang(
        target_lang, asr_meta)
    n = len(blocks)

    body = "\n\n".join(
        f"[{i}] {float(b['start']):.2f}-{float(b['end']):.2f} "
        f"({float(b['end']) - float(b['start']):.1f}s, ~{int(b['words'])} words)\n"
        f"{(b.get('text') or '').strip()}"
        for i, b in enumerate(blocks)
    )
    if same_mode:
        task = (
            "Lightly repair the ASR errors and keep the original wording "
            "where a block is already clear. Do NOT translate into another "
            "language."
        )
    else:
        task = f"Write a natural spoken translation into {lang_name}."

    raw = _chat(
        _blocks_system_prompt(
            n=n, lang_name=lang_name, expected_code=expected_code,
            lang_det=asr_meta.get("language") or "unknown",
            lang_p=float(asr_meta.get("language_probability") or 0.0),
            task=task,
        ),
        body, api_key, model,
    )
    data = _extract_json(raw)
    lines = _block_variants(data, n, body, api_key, model)
    lines = _repair_block_languages(
        lines, expected_code, lang_name, api_key, model)

    master = (data.get("master_translation") or "").strip()
    if not master:
        master = " ".join(entry["normal"] for entry in lines if entry["normal"])
    meaning = (data.get("master_meaning") or "").strip() or master[:240]
    return {
        "lines": lines,
        "master_meaning": meaning,
        "master_translation": master,
        "output_lang_code": expected_code,
        "output_lang_name": lang_name,
    }


def _one_entry(item) -> dict | None:
    """Normalise one entry to three non-empty lengths, or None.

    A model that answers with a bare string is not wrong about the words,
    only about the shape, so that is accepted and the one line is used at
    all three lengths.
    """
    if isinstance(item, str):
        text = item.strip()
        return {"short": text, "normal": text, "long": text} if text else None
    if not isinstance(item, dict):
        return None
    out = {}
    for key in VARIANTS:
        value = str(item.get(key) or "").strip()
        out[key] = value
    if not out["normal"]:
        out["normal"] = out["long"] or out["short"]
    if not out["normal"]:
        return None
    for key in ("short", "long"):
        if not out[key]:
            out[key] = out["normal"]
    return out


def _block_variants(data: dict, n: int, body: str, api_key: str,
                    model: str) -> list:
    """Exactly n entries of three lengths, or one repair call, or a failure.

    There is no guessing here on purpose. The old code padded a short list
    with empty strings at a place it picked by word overlap, which silently
    shifted every later line by one block.
    """
    entries = _entries_or_none(data.get("lines"), n)
    if entries is not None:
        return entries

    given = data.get("lines")
    got = len(given) if isinstance(given, list) else type(given).__name__
    out = _chat(
        (
            f"You returned {got} usable entries; exactly {n} are needed, one "
            f"per block, in the same order, and none of them may be empty. "
            f"Split any entry you merged back onto the blocks it came from. "
            f"Every block has speech, so write words for every one, using "
            f"the neighbouring blocks when the transcript is unclear. "
            f"Return ONLY JSON: {{\"lines\": [exactly {n} objects with keys "
            f"short, normal, long]}}."
        ),
        body + "\n\nYour lines:\n" + json.dumps(given, ensure_ascii=False),
        api_key, model,
    )
    entries = _entries_or_none(_extract_json(out).get("lines"), n)
    if entries is None:
        raise OpenAIError(
            f"Ban dich phai co dung {n} dong day du cho {n} block")
    return entries


def _entries_or_none(raw, n: int) -> list | None:
    if not isinstance(raw, list) or len(raw) != n:
        return None
    entries = [_one_entry(item) for item in raw]
    return None if any(entry is None for entry in entries) else entries


def _repair_block_languages(lines, expected_code: str, lang_name: str,
                            api_key: str, model: str) -> list:
    """Redo the variants that came back in the wrong language. One call."""
    wrong = _wrong_language_keys(lines, expected_code)
    if not wrong:
        return lines
    out = _chat(
        (
            f"Rewrite each line into {lang_name} only, keeping its meaning "
            f"and roughly its length. Return ONLY JSON: "
            f"{{\"lines\": {{\"<index>.<short|normal|long>\": \"<line>\"}}}}."
        ),
        json.dumps({f"{i}.{key}": lines[i][key] for i, key in wrong},
                   ensure_ascii=False),
        api_key, model,
    )
    fixed = _extract_json(out).get("lines") or {}
    repaired = [dict(entry) for entry in lines]
    for tag, value in fixed.items():
        index, _, key = str(tag).partition(".")
        text = str(value or "").strip()
        try:
            index = int(index)
        except (TypeError, ValueError):
            continue
        if key in VARIANTS and 0 <= index < len(repaired) and text:
            repaired[index][key] = text
    still = _wrong_language_keys(repaired, expected_code)
    if still:
        raise OpenAIError(
            f"Van con dong sai ngon ngu sau khi sua (mong {expected_code}): "
            + " | ".join(f"[{i}.{key}] {repaired[i][key][:60]}"
                         for i, key in still))
    return repaired


def _wrong_language_keys(lines, expected_code: str) -> list:
    """(index, variant) pairs whose text is not in the output language."""
    out = []
    for index, entry in enumerate(lines):
        for key in VARIANTS:
            if _lines_wrong_language([entry[key]], expected_code):
                out.append((index, key))
    return out


def _selfcheck():
    """The block path, with the network replaced by canned answers."""
    replies = []

    def fake_chat(system, user, api_key, model):
        return replies.pop(0)

    global _chat
    real_chat, _chat = _chat, fake_chat
    try:
        blocks = [
            {"start": 0.0, "end": 2.0, "text": "hello there", "words": 9},
            {"start": 3.0, "end": 5.0, "text": "buy it now", "words": 9},
        ]

        def entry(text):
            return {"short": text, "normal": text, "long": text + " nhé"}

        replies.append(json.dumps({
            "master_meaning": "an ad",
            "master_translation": "xin chào. mua ngay.",
            "lines": [entry("xin chào các bạn nhé"), entry("mua ngay hôm nay đi")],
        }))
        out = translate_blocks(blocks, "vi", "key")
        assert out["lines"][0]["normal"] == "xin chào các bạn nhé", out
        assert out["lines"][1]["long"].endswith("nhé"), out
        assert out["output_lang_code"] == "vi"

        # An entry with an empty length is filled from the one that is there.
        replies.append(json.dumps({
            "lines": [
                {"short": "", "normal": "một dòng đầy đủ đây", "long": ""},
                entry("mua ngay hôm nay đi"),
            ],
        }))
        out = translate_blocks(blocks, "vi", "key")
        assert out["lines"][0]["short"] == "một dòng đầy đủ đây", out

        # A short list is repaired by asking again, never padded by guessing.
        replies.append(json.dumps({"lines": [entry("chỉ một dòng thôi bạn")]}))
        replies.append(json.dumps({
            "lines": [entry("dòng một ở đây"), entry("dòng hai ở đây")]}))
        out = translate_blocks(blocks, "vi", "key")
        assert [e["normal"] for e in out["lines"]] == [
            "dòng một ở đây", "dòng hai ở đây"], out

        # A block left empty is a failure, not something to paper over.
        replies.append(json.dumps({"lines": [entry("dòng một ở đây"), ""]}))
        replies.append(json.dumps({"lines": [entry("dòng một ở đây"), ""]}))
        try:
            translate_blocks(blocks, "vi", "key")
        except OpenAIError:
            pass
        else:
            raise AssertionError("block rong phai bao loi")

        # A variant that came back in English is sent back once.
        replies.append(json.dumps({
            "master_translation": "x",
            "lines": [
                entry("xin chào các bạn nhé"),
                {"short": "mua ngay đi bạn ơi", "normal": "mua ngay hôm nay đi",
                 "long": "buy it now today please my friend"},
            ],
        }))
        replies.append(json.dumps(
            {"lines": {"1.long": "mua ngay hôm nay đi bạn ơi"}}))
        out = translate_blocks(blocks, "vi", "key")
        assert out["lines"][1]["long"] == "mua ngay hôm nay đi bạn ơi", out
    finally:
        _chat = real_chat

    assert word_count("một hai ba") == 3
    assert _resolve_output_lang("vi", {})[0] == "vi"
    assert _lines_wrong_language(["hello there friend"], "vi") == [0]
    assert _lines_wrong_language(["xin chào các bạn ơi"], "vi") == []
    print("translate.py self-check OK")


if __name__ == "__main__":
    _selfcheck()
