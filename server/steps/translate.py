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

# Tốc độ đọc voice-over tự nhiên → ước số từ; sai số còn lại do fit_length + atempo.
WORDS_PER_SECOND = 2.4
# Band chấp nhận sau rewrite; ngoài band → 1 lần fit_length.


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


CANNOT_FIT = "CANNOT_FIT"

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
    """Prompt for block translation — one line per block, no reshuffling."""
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
        f"HARD RULE: return exactly {n} lines, one per block, in order. "
        f"lines[i] is spoken while block i plays. Never merge two blocks "
        f"into one line, never move a line to another index, never drop one. "
        f"Use \"\" only when a block truly has nothing to dub. "
        f"Each block header says how long it is and how many words fit in "
        f"that time. Stay near that word count: a line far over it has to be "
        f"rushed, a line far under it leaves the speaker silent on screen. "
        f"Everything must be in {lang_name} only — never mix languages. "
        f"Do not invent products, prices or claims that are not in the "
        f"transcript. "
        f"Return ONLY valid JSON (no markdown) with keys: master_meaning "
        f"(one sentence, in English), master_translation (the full spoken "
        f"script in {lang_name}), lines (array of exactly {n} strings)."
    )


def translate_blocks(blocks, target_lang: str, api_key: str,
                     asr_meta=None, model: str = DEFAULT_MODEL) -> dict:
    """Translate whole blocks of speech, one line per block.

    Blocks are built from word timestamps by the caller, so the mapping from
    text to time is decided by code, not by the model. The model only has to
    keep the order and the count, and both are checked here.
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
    lines = _block_lines(data, n, body, api_key, model)
    lines = _repair_block_languages(
        lines, expected_code, lang_name, api_key, model)

    master = (data.get("master_translation") or "").strip()
    if not master:
        master = " ".join(line for line in lines if line)
    meaning = (data.get("master_meaning") or "").strip() or master[:240]
    return {
        "lines": lines,
        "master_meaning": meaning,
        "master_translation": master,
        "output_lang_code": expected_code,
        "output_lang_name": lang_name,
    }


def _block_lines(data: dict, n: int, body: str, api_key: str,
                 model: str) -> list:
    """Exactly n lines, or one repair call, or a loud failure.

    There is no guessing here on purpose. The old code padded a short list
    with empty strings at a place it picked by word overlap, which silently
    shifted every later line by one block.
    """
    lines = data.get("lines")
    if isinstance(lines, list) and len(lines) == n:
        return [str(line or "").strip() for line in lines]

    got = len(lines) if isinstance(lines, list) else type(lines).__name__
    out = _chat(
        (
            f"You returned {got} lines; exactly {n} are needed, one per "
            f"block, in the same order. Split any line you merged back onto "
            f"the blocks it came from, and use \"\" for a block with nothing "
            f"to dub. Return ONLY JSON: {{\"lines\": [exactly {n} strings]}}."
        ),
        body + "\n\nYour lines:\n" + json.dumps(lines, ensure_ascii=False),
        api_key, model,
    )
    fixed = _extract_json(out).get("lines")
    if not isinstance(fixed, list) or len(fixed) != n:
        raise OpenAIError(
            f"Ban dich phai co dung {n} dong cho {n} block, nhan "
            f"{len(fixed) if isinstance(fixed, list) else type(fixed)}")
    return [str(line or "").strip() for line in fixed]


def _repair_block_languages(lines, expected_code: str, lang_name: str,
                            api_key: str, model: str) -> list:
    """Redo the lines that came back in the wrong language. One call."""
    wrong = _lines_wrong_language(lines, expected_code)
    if not wrong:
        return lines
    out = _chat(
        (
            f"Rewrite each line into {lang_name} only, keeping its meaning "
            f"and roughly its length. Return ONLY JSON: "
            f"{{\"lines\": {{\"<index>\": \"<line>\"}}}}."
        ),
        json.dumps({str(i): lines[i] for i in wrong}, ensure_ascii=False),
        api_key, model,
    )
    fixed = _extract_json(out).get("lines") or {}
    repaired = list(lines)
    for key, value in fixed.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(repaired) and str(value or "").strip():
            repaired[index] = str(value).strip()
    still = _lines_wrong_language(repaired, expected_code)
    if still:
        raise OpenAIError(
            f"Van con dong sai ngon ngu sau khi sua (mong {expected_code}): "
            + " | ".join(f"[{i}] {repaired[i][:80]}" for i in still))
    return repaired


def rephrase_for_duration(text: str, seconds: float, api_key: str, *,
                          master_meaning: str, prev_text: str = "",
                          next_text: str = "", shorter: bool = True,
                          target_lang: str = "", lang_name: str = "",
                          target_words_n: int,
                          model: str = DEFAULT_MODEL) -> str:
    """Chỉ đổi phrasing để vừa ~seconds. Khóa intent dòng — có thể CANNOT_FIT."""
    text = (text or "").strip()
    if not text:
        raise OpenAIError("Không có text để rephrase")
    sec = max(float(seconds), 0.4)
    # The word budget is measured from the real voice by the caller. There is
    # no constant here on purpose: the old one was tuned for English and made
    # every Vietnamese rewrite come out half as long as the slot.
    tw = max(int(target_words_n), 2)
    system = _rephrase_system_prompt(
        seconds=sec, target_words_n=tw, shorter=shorter,
        master_meaning=master_meaning, prev_text=prev_text,
        next_text=next_text, lang_name=lang_name)
    out = _chat(system, text, api_key, model)
    cleaned = (out or "").strip().strip('"').strip("'")
    compact = re.sub(r"[\s_]+", "", cleaned.upper())
    if compact == "CANNOTFIT" or compact.startswith("CANNOTFIT"):
        return CANNOT_FIT
    return cleaned


def _rephrase_system_prompt(*, seconds: float, target_words_n: int,
                            shorter: bool, master_meaning: str,
                            prev_text: str, next_text: str,
                            lang_name: str) -> str:
    """Prompt rephrase — tách ra để self-check substring rules."""
    direction = "slightly SHORTER" if shorter else "slightly LONGER"
    if lang_name:
        lang_rule = (
            f"Keep the output entirely in {lang_name} "
            f"(same language as the input line)."
        )
    else:
        lang_rule = "Keep the SAME language as the input line."
    intent_hard = (
        "HARD: Keep the SAME speech-act / same intent as the INPUT line "
        "(question stays a question; CTA stays a CTA; app pitch stays an "
        "app pitch). Do not borrow ideas from master_meaning, previous, or "
        "next to replace this line's topic. Context lines are only to avoid "
        "repeating or stealing their info — not to import their content."
    )
    if shorter:
        length_rules = (
            f"{lang_rule} {intent_hard} "
            "Aggressive condense ONLY this line to its core speech-act "
            "(especially CTAs: 'click the button below now' → "
            "'click below' / 'click the button'). "
            "Drop adverbs and filler; do not substitute a different sentence "
            "from the ad. Do not add padding. "
            f"Stay near the ~{seconds:.1f}s target band — do NOT overshoot "
            f"into a line that would be far too short. "
            f"Return exactly {CANNOT_FIT} only if even an ultra-short form "
            f"cannot keep the same speech-act in ~{seconds:.1f}s."
        )
    else:
        length_rules = (
            f"{lang_rule} {intent_hard} "
            "Allow light expansion of ONLY this line: natural synonyms and "
            "at most 1–2 short relative clauses that keep the same topic. "
            f"Aim close to ~{seconds:.1f}s spoken (~{target_words_n} words). "
            "Do not import ideas from master/prev/next. "
            "Do not add empty filler particles. "
            f"Return exactly {CANNOT_FIT} only if you truly cannot lengthen "
            f"without changing the topic in ~{seconds:.1f}s."
        )
    return (
        f"You rephrase ONE dubbing line for timing only. "
        f"Target ~{seconds:.1f}s spoken (~{target_words_n} words), "
        f"make it {direction}. "
        f"MASTER MEANING (context only — do not borrow to change this "
        f"line's intent):\n{master_meaning}\n\n"
        f"Previous line (do not repeat its info): {prev_text or '(none)'}\n"
        f"Next line (do not steal its info): {next_text or '(none)'}\n\n"
        f"Rules: {length_rules} "
        f"Otherwise return only the rephrased line."
    )


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
        replies.append(json.dumps({
            "master_meaning": "an ad",
            "master_translation": "xin chào. mua ngay.",
            "lines": ["xin chào các bạn nhé", "mua ngay hôm nay đi"],
        }))
        out = translate_blocks(blocks, "vi", "key")
        assert out["lines"] == ["xin chào các bạn nhé", "mua ngay hôm nay đi"], out
        assert out["output_lang_code"] == "vi"

        # A short list is repaired by asking again, never padded by guessing.
        replies.append(json.dumps({"lines": ["một dòng duy nhất thôi"]}))
        replies.append(json.dumps({"lines": ["dòng một ở đây", "dòng hai ở đây"]}))
        out = translate_blocks(blocks, "vi", "key")
        assert out["lines"] == ["dòng một ở đây", "dòng hai ở đây"], out

        # Still the wrong count after the repair: fail loudly.
        replies.append(json.dumps({"lines": ["a"]}))
        replies.append(json.dumps({"lines": ["a"]}))
        try:
            translate_blocks(blocks, "vi", "key")
        except OpenAIError:
            pass
        else:
            raise AssertionError("sai so dong phai bao loi")

        # A line that came back in English is sent back once.
        replies.append(json.dumps({
            "master_translation": "x",
            "lines": ["xin chào các bạn nhé", "buy it now today please"],
        }))
        replies.append(json.dumps({"lines": {"1": "mua ngay hôm nay đi"}}))
        out = translate_blocks(blocks, "vi", "key")
        assert out["lines"][1] == "mua ngay hôm nay đi", out

        # rephrase passes the measured budget through to the prompt.
        replies.append("câu ngắn hơn nhiều")
        text = rephrase_for_duration(
            "một câu rất dài", 2.0, "key",
            master_meaning="", target_words_n=9, lang_name="Vietnamese")
        assert text == "câu ngắn hơn nhiều", text
    finally:
        _chat = real_chat

    prompt = _rephrase_system_prompt(
        seconds=2.0, target_words_n=9, shorter=True, master_meaning="m",
        prev_text="", next_text="", lang_name="Vietnamese")
    assert "9" in prompt, "the word budget must reach the model"

    assert word_count("một hai ba") == 3
    assert _resolve_output_lang("vi", {})[0] == "vi"
    assert _lines_wrong_language(["hello there friend"], "vi") == [0]
    assert _lines_wrong_language(["xin chào các bạn ơi"], "vi") == []
    print("translate.py self-check OK")


if __name__ == "__main__":
    _selfcheck()
