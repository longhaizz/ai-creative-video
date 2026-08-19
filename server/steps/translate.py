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
WORD_BAND = 0.15


class OpenAIError(PipelineError):
    """A failure the user should see, with the standard error code."""

    def __init__(self, message: str):
        super().__init__(message, code="internal")


def word_count(text: str) -> int:
    return len((text or "").split())


def target_words(seconds) -> int:
    if not seconds or seconds <= 0:
        return 3
    return max(int(float(seconds) * WORDS_PER_SECOND), 3)


def _in_band(n: int, target: int) -> bool:
    if target <= 0:
        return True
    lo = target * (1 - WORD_BAND)
    hi = target * (1 + WORD_BAND)
    return lo <= n <= hi


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


def _length_rule(seconds) -> str:
    """Duration-aware dubbing: ưu tiên câu nói vừa slot, giữ nghĩa."""
    if not seconds or seconds <= 0:
        return ""
    words = target_words(seconds)
    return (
        f"This is for video dubbing. The spoken result must fit approximately "
        f"{seconds:.1f} seconds (~{words} words at a natural pace). "
        f"Preserve the meaning, but prioritize natural spoken language and timing. "
        f"You may shorten wording, remove redundancy, or restructure the sentence. "
        f"Do not omit important facts, numbers, product names, or the call to action. "
        f"Do not invent new claims. Return only the spoken sentence. "
    )


def fit_length(text: str, target: int, api_key: str,
               model: str = DEFAULT_MODEL) -> str:
    """Viết lại cùng ngôn ngữ tới ~target từ — ưu tiên rút khi dài hơn slot."""
    text = (text or "").strip()
    if not text:
        raise OpenAIError("Không có text để chỉnh độ dài")
    target = max(int(target), 3)
    cur = word_count(text)
    if cur > target:
        direction = (
            "Shorten for dubbing timing: remove redundancy, use shorter synonyms, "
            "restructure — keep meaning, facts, numbers, product names, CTA"
        )
    else:
        direction = (
            "Slightly lengthen with a few neutral connecting words only — "
            "do NOT invent new ideas or CTAs"
        )
    return _chat(
        (
            f"You rewrite voice-over for video dubbing in the SAME language to about "
            f"{target} words (currently ~{cur}). {direction}. "
            f"Natural spoken language. Return only the rewritten text."
        ),
        text, api_key, model,
    )


def rewrite_for_slot(text: str, seconds: float, api_key: str,
                     model: str = DEFAULT_MODEL, shorter: bool = True) -> str:
    """Rewrite cùng ngôn ngữ để đọc vừa ~seconds (thường rút khi TTS quá dài)."""
    text = (text or "").strip()
    if not text:
        raise OpenAIError("Không có text để rewrite")
    sec = max(float(seconds), 0.4)
    tw = target_words(sec)
    bias = (
        "Prefer a SHORTER spoken line that still preserves meaning. "
        "You may shorten, remove redundancy, or restructure."
        if shorter
        else "Prefer a LONGER spoken line to fill the slot: add natural "
        "connecting words or light on-topic phrasing — do NOT invent false "
        "facts, prices, or new CTAs."
    )
    return _chat(
        (
            f"Rewrite this voice-over in the SAME language for video dubbing. "
            f"It must be speakable in about {sec:.1f} seconds (~{tw} words). "
            f"{bias} "
            f"Do not omit important facts/numbers/product names/CTA. "
            f"Return only the spoken sentence."
        ),
        text, api_key, model,
    )


def _refit_if_needed(text: str, seconds, api_key: str, model: str) -> str:
    if not seconds or seconds <= 0:
        return text
    tw = target_words(seconds)
    if _in_band(word_count(text), tw):
        return text
    return fit_length(text, tw, api_key, model)


def looks_garbled(text: str) -> bool:
    """Heuristic: chuỗi ASR vô nghĩa / giữ nguyên sẽ phá bước dịch."""
    text = (text or "").strip()
    if not text:
        return True
    words = [w for w in re.sub(r"[,.!?]+", " ", text).split() if w]
    if len(words) < 2:
        return False
    titled = sum(1 for w in words if len(w) > 1 and w[0].isupper() and w[1:].islower())
    if titled / len(words) >= 0.6 and not text.endswith((".", "!", "?")):
        return True
    # Nhiều token "tên riêng giả" dài, ít từ chức năng thường gặp
    longish = sum(1 for w in words if len(w) >= 8)
    if len(words) >= 3 and longish / len(words) >= 0.5:
        return True
    return False


def _mostly_same_script(src: str, dst: str) -> bool:
    """True nếu bản dịch gần như giữ nguyên chuỗi nguồn (ASR rác bị leak)."""
    def norm(t):
        return re.sub(r"[^a-z0-9]+", "", (t or "").lower())
    a, b = norm(src), norm(dst)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) >= 12 and (a in b or b in a):
        return True
    # overlap token
    aw = set(re.findall(r"[a-zA-Z]{4,}", src or ""))
    bw = set(re.findall(r"[a-zA-Z]{4,}", dst or ""))
    if aw and bw and len(aw & bw) / max(len(aw), 1) >= 0.6:
        return True
    return False


def _asr_context_rule(lang: str) -> str:
    return (
        f"CRITICAL — every output line MUST be natural spoken {lang}. "
        f"Never leave source-language text, phonetic gibberish, or ASR garbage "
        f"unchanged. If a line is marked [ASR_UNCLEAR] or looks nonsensical / "
        f"garbled / hallucinated ASR, do NOT copy it: infer the intended meaning "
        f"from surrounding lines (same ad script — often a CTA, disclaimer, or "
        f"closing), then write a fitting {lang} line for that slot. Prefer a "
        f"plausible on-topic closing/CTA consistent with the script over "
        f"preserving nonsense. Still do not invent false prices or product claims "
        f"not supported by context. "
    )


def translate(text: str, target_lang: str, api_key: str,
              model: str = DEFAULT_MODEL, seconds=None) -> str:
    """Viết lại voice-over sang target_lang; khớp ~số từ theo seconds nếu có."""
    text = (text or "").strip()
    if not text:
        raise OpenAIError("Không có text để dịch")
    code = (target_lang or "").strip().upper()
    if not code or code == "SAME":
        raise OpenAIError(f"target_lang không hợp lệ: {target_lang!r}")
    lang = LANG_NAMES.get(code, code)
    unclear = " The source may be garbled ASR — infer meaning; always output " + lang + "."

    out = _chat(
        (
            f"You write spoken dubbing lines in {lang}. "
            f"Rewrite the user's text into natural spoken {lang}. "
            f"Preserve meaning — not word-for-word. "
            f"Keep domain terms, proper nouns, product names, numbers and the "
            f"call to action accurate. "
            f"{_asr_context_rule(lang)}"
            f"{_length_rule(seconds)}"
            f"{unclear if looks_garbled(text) else ''}"
            f"Return only the spoken sentence, with no quotes or notes."
        ),
        text, api_key, model,
    )
    return _refit_if_needed(out, seconds, api_key, model)


def _parse_numbered(raw: str, n: int) -> list:
    """'1. abc' → ['abc', …]; ném nếu thiếu/thừa dòng so với n."""
    found = {}
    for ln in (raw or "").splitlines():
        m = re.match(r"\s*(\d+)\s*[.):]\s*(.+)$", ln)
        if m:
            found[int(m.group(1))] = m.group(2).strip()
    missing = [i for i in range(1, n + 1) if not found.get(i)]
    if missing or len(found) != n:
        raise OpenAIError(
            f"OpenAI trả {len(found)}/{n} dòng (thiếu {missing[:5]})")
    return [found[i] for i in range(1, n + 1)]


def _repair_line_from_context(lines, idx: int, slot, target_lang: str,
                              api_key: str, model: str) -> str:
    """Một dòng ASR rác: viết lại bằng ngôn ngữ đích nhờ ngữ cảnh quanh."""
    code = (target_lang or "").strip().upper()
    lang = LANG_NAMES.get(code, code)
    n = len(lines)
    ctx = "\n".join(f"{j + 1}. {lines[j]}" for j in range(n))
    bad = lines[idx]
    sec = float(slot) if slot else 0
    length = _length_rule(sec) if sec else ""
    return _chat(
        (
            f"You write one video-dubbing line in {lang}. "
            f"Line {idx + 1} of the script below is garbled ASR (nonsense / "
            f"wrong transcription). Using the OTHER lines as context, write what "
            f"line {idx + 1} should say in natural spoken {lang} — usually a CTA "
            f"or closing consistent with the ad. Do not copy the garbled text. "
            f"Do not invent false prices or product claims absent from context. "
            f"{length}"
            f"Return ONLY that one spoken sentence for line {idx + 1}."
        ),
        f"Full script (line {idx + 1} is bad):\n{ctx}\n\nBad line text:\n{bad}",
        api_key, model,
    )


def translate_lines(lines, target_lang: str, api_key: str,
                    model: str = DEFAULT_MODEL, slots=None) -> list:
    """Viết lại từng câu, GIỮ ĐÚNG số dòng, khớp độ dài theo slots nếu có.

    Dòng ASR vô nghĩa: suy từ ngữ cảnh → luôn ra ngôn ngữ đích (không giữ nguyên).
    """
    lines = [(t or "").strip() for t in lines]
    if not lines or not all(lines):
        raise OpenAIError("Có dòng rỗng trong danh sách cần dịch")
    code = (target_lang or "").strip().upper()
    if not code or code == "SAME":
        raise OpenAIError(f"target_lang không hợp lệ: {target_lang!r}")
    lang = LANG_NAMES.get(code, code)
    slots = list(slots or [])
    garbled = [looks_garbled(t) for t in lines]

    parts = []
    for i, t in enumerate(lines):
        mark = "[ASR_UNCLEAR] " if garbled[i] else ""
        if slots and len(slots) == len(lines):
            parts.append(
                f"{i + 1}. [{slots[i]:.1f}s ~{target_words(slots[i])}w] {mark}{t}")
        else:
            parts.append(f"{i + 1}. {mark}{t}")
    body = "\n".join(parts)

    if slots and len(slots) == len(lines):
        slot_rule = (
            f"Each line is numbered and may be prefixed with [Ts ~Nw] and "
            f"optionally [ASR_UNCLEAR]. Rewrite into natural spoken {lang} that "
            f"fits about T seconds. Preserve meaning when the source is clear; "
            f"you may shorten or restructure for timing. Do not omit important "
            f"facts/numbers/product names/CTA from clear lines. "
            f"Do not output the [Ts ~Nw] or [ASR_UNCLEAR] markers. "
        )
    else:
        slot_rule = ""

    out = _chat(
        (
            f"You write video-dubbing lines in {lang}. "
            f"Use the whole script as context, but rewrite each line on its own. "
            f"{_asr_context_rule(lang)}"
            f"{slot_rule}"
            f"Return EXACTLY {len(lines)} lines, same order, same numbering "
            f"('1. ', '2. ', …). Never merge, split, add, drop or reorder lines. "
            f"No quotes, no notes, no blank lines."
        ),
        body, api_key, model,
    )
    result = _parse_numbered(out, len(lines))

    # Hậu kiểm: dòng vẫn giữ ASR rác → repair bằng context
    for i, (src, dst) in enumerate(zip(lines, result)):
        if garbled[i] or _mostly_same_script(src, dst):
            slot = slots[i] if slots and len(slots) == len(lines) else None
            result[i] = _repair_line_from_context(
                lines, i, slot, code, api_key, model)

    if slots and len(slots) == len(result):
        for i, (line, slot) in enumerate(zip(result, slots)):
            tw = target_words(slot)
            if not _in_band(word_count(line), tw):
                result[i] = fit_length(line, tw, api_key, model)
    return result


def fit_lines(lines, slots, api_key: str, model: str = DEFAULT_MODEL) -> list:
    """Cùng ngôn ngữ: chỉnh từng dòng tới ~target_words(slot) nếu lệch band."""
    lines = [(t or "").strip() for t in lines]
    slots = list(slots or [])
    if not lines or not all(lines):
        raise OpenAIError("Có dòng rỗng trong danh sách cần chỉnh")
    if len(slots) != len(lines):
        raise OpenAIError("slots phải cùng số phần tử với lines")
    out = []
    for line, slot in zip(lines, slots):
        tw = target_words(slot)
        if _in_band(word_count(line), tw):
            out.append(line)
        else:
            out.append(fit_length(line, tw, api_key, model))
    return out


def shorten(text: str, ratio: float, api_key: str,
            model: str = DEFAULT_MODEL) -> str:
    """Rút text còn ~ratio độ dài, GIỮ NGUYÊN ngôn ngữ (wrapper fit_length)."""
    text = (text or "").strip()
    if not text:
        raise OpenAIError("Không có text để rút gọn")
    ratio = max(min(float(ratio), 0.95), 0.3)
    return fit_length(text, max(int(word_count(text) * ratio), 3), api_key, model)


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


def _cue_lang_mismatch(cues, expected_code: str) -> list:
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


def _cue_translations_from_groups(groups, n_asr: int) -> list:
    """Suy cue_translations nếu mỗi group đúng 1 ASR index; else lỗi."""
    cues = [""] * n_asr
    filled = set()
    for g in groups:
        idxs = g["source_segment_indices"]
        if len(idxs) != 1:
            raise OpenAIError(
                "Thiếu cue_translations và semantic_group span nhiều cue "
                "— không được merge TTS theo group")
        idx = idxs[0]
        if idx in filled:
            raise OpenAIError(f"cue {idx} bị trùng trong semantic_groups")
        cues[idx] = g["translation"]
        filled.add(idx)
    return cues


def _normalize_dub_script(data: dict, n_asr: int) -> dict:
    """Chuẩn hoá + validate reconstruct_script output."""
    groups_in = data.get("semantic_groups") or data.get("segments") or []
    raw_cues = data.get("cue_translations")
    if not groups_in and raw_cues is None:
        raise OpenAIError(
            "reconstruct_script thiếu cue_translations / semantic_groups")

    groups = []
    for i, g in enumerate(groups_in or []):
        start = float(g.get("start", 0))
        end = float(g.get("end", start + 0.4))
        if end <= start:
            end = start + 0.4
        idxs = g.get("source_segment_indices")
        if idxs is None and g.get("source_segments") is not None:
            idxs = g.get("source_segments")
        if not isinstance(idxs, list):
            idxs = [i]
        idxs = [int(x) for x in idxs]
        for x in idxs:
            if x < 0 or x >= n_asr:
                raise OpenAIError(f"source_segment_indices ngoài range: {idxs}")
        trans = (g.get("translation") or g.get("text") or "").strip()
        src = (g.get("source_text") or g.get("source") or "").strip()
        if not trans:
            raise OpenAIError(f"semantic_group {i} thiếu translation")
        groups.append({
            "id": int(g.get("id", i + 1)),
            "source_segment_indices": idxs,
            "start": start,
            "end": end,
            "source_text": src,
            "translation": trans,
        })

    if raw_cues is not None:
        if not isinstance(raw_cues, list) or len(raw_cues) != n_asr:
            raise OpenAIError(
                f"cue_translations phải có đúng {n_asr} phần tử, "
                f"nhận {len(raw_cues) if isinstance(raw_cues, list) else type(raw_cues)}")
        cue_translations = [
            (t if isinstance(t, str) else str(t or "")).strip()
            for t in raw_cues
        ]
    else:
        cue_translations = _cue_translations_from_groups(groups, n_asr)

    if not any(cue_translations):
        raise OpenAIError("cue_translations toàn rỗng")

    return {
        "source_language": (data.get("source_language") or "").strip(),
        "asr_confidence": (data.get("asr_confidence") or "medium").strip().lower(),
        "master_meaning": (data.get("master_meaning") or "").strip(),
        "master_translation": (data.get("master_translation") or "").strip(),
        "uncertain_spans": list(data.get("uncertain_spans") or []),
        "semantic_groups": groups,
        "cue_translations": cue_translations,
    }


def _repair_cue_languages(script: dict, mismatched: list, lang_name: str,
                          api_key: str, model: str = DEFAULT_MODEL) -> dict:
    """1 lần: viết lại đúng các cue lệch sang lang_name."""
    import json
    cues = list(script["cue_translations"])
    n = len(cues)
    payload = {
        "mismatched_indices": mismatched,
        "cue_translations": cues,
        "master_translation": script.get("master_translation") or "",
        "master_meaning": script.get("master_meaning") or "",
    }
    raw = _chat(
        (
            f"Some cue_translations are in the WRONG language. "
            f"Rewrite ONLY the cues at indices {mismatched} into {lang_name}. "
            f"Keep meaning aligned with master_translation / master_meaning. "
            f"Do not invent facts. Leave all other cues unchanged. "
            f"Return ONLY valid JSON: "
            f"{{\"cue_translations\": [exactly {n} strings]}}."
        ),
        json.dumps(payload, ensure_ascii=False),
        api_key, model,
    )
    data = _extract_json(raw)
    fixed = data.get("cue_translations")
    if not isinstance(fixed, list) or len(fixed) != n:
        raise OpenAIError(
            f"language repair trả cue_translations sai độ dài "
            f"(cần {n})")
    out = dict(script)
    out["cue_translations"] = [
        (t if isinstance(t, str) else str(t or "")).strip() for t in fixed
    ]
    # Cập nhật group translation nếu group chỉ 1 cue được sửa
    groups = []
    for g in out.get("semantic_groups") or []:
        gg = dict(g)
        idxs = gg.get("source_segment_indices") or []
        if len(idxs) == 1 and idxs[0] in mismatched:
            gg["translation"] = out["cue_translations"][idxs[0]]
        groups.append(gg)
    out["semantic_groups"] = groups
    return out


def reconstruct_script(segs, target_lang: str, api_key: str,
                       asr_meta=None, model: str = DEFAULT_MODEL) -> dict:
    """Phục hồi nghĩa TOÀN video + master translation + per-cue lines.

    Chưa đụng duration. Master meaning/translation là source of truth.
    target_lang='same' → sửa ASR nhẹ trong ngôn ngữ nguồn (Whisper detect).
    """
    segs = list(segs or [])
    if not segs:
        raise OpenAIError("Không có ASR segments để reconstruct")
    asr_meta = asr_meta or {}
    expected_code, lang_name, same_mode = _resolve_output_lang(
        target_lang, asr_meta)

    blocks = []
    for i, s in enumerate(segs):
        ss = float(s.get("speech_start", s["start"]))
        se = float(s.get("speech_end", s["end"]))
        t = (s.get("text") or "").replace("\n", " ").strip()
        blocks.append(f"[{i}] [{ss:.2f}-{se:.2f}]\n{t}")
    body = "\n\n".join(blocks)
    lang_det = asr_meta.get("language") or "unknown"
    lang_p = float(asr_meta.get("language_probability") or 0.0)
    conf = asr_meta.get("asr_confidence") or (
        "low" if lang_p and lang_p < 0.75 else "medium")

    n = len(segs)
    if same_mode:
        task = (
            f"Lightly repair ASR errors and allocate spoken lines in "
            f"{lang_name} only. Prefer keeping the original wording when a "
            f"cue is already clear. Do NOT creatively translate into another "
            f"language."
        )
    else:
        task = (
            f"Produce a natural spoken translation into {lang_name} only."
        )

    raw = _chat(
        (
            f"You are reconstructing an ASR transcript for video dubbing. "
            f"The ASR may contain severe phonetic errors, mixed-language "
            f"transcriptions, duplicated words, and incorrect segmentation. "
            f"Detected language hint: {lang_det} (p={lang_p:.2f}), "
            f"overall ASR confidence: {conf}. "
            f"Output language MUST be {lang_name} (code={expected_code}). "
            f"First infer the most likely intended message using the ENTIRE "
            f"transcript as context. Do not treat each ASR segment as an "
            f"independent sentence — adjacent segments may form one sentence. "
            f"Do not invent product claims, prices, or information that cannot "
            f"reasonably be inferred from context. "
            f"{task} "
            f"HARD RULE: master_translation, every cue_translations[i], and "
            f"every semantic_group.translation MUST be entirely in {lang_name}. "
            f"Do not mix languages (e.g. do not insert Vietnamese into an "
            f"Indonesian script, or vice versa). "
            f"CRITICAL: TTS keeps each ASR cue's original pause timing, so you "
            f"MUST allocate spoken lines into cue_translations — an array of "
            f"exactly {n} strings, index-aligned with ASR cues [0..{n - 1}]. "
            f"Each string is what should be spoken in that cue's speech window; "
            f"use \"\" only if that cue has no speech to dub. "
            f"DURATION-AWARE: match how much is said to each cue's "
            f"[speech_start–speech_end] length — do NOT pack most of the "
            f"script into an early cue when later cues still have time. "
            f"Whisper may split mid-sentence; still balance by timing. "
            f"cue_translations must only redistribute master_translation — "
            f"no new facts. "
            f"master_translation MUST be the FULL spoken script (every idea "
            f"that appears in cue_translations / groups — do not truncate). "
            f"semantic_groups may SPAN multiple ASR indices for meaning/"
            f"context only (not TTS units). "
            f"Return ONLY valid JSON (no markdown) with keys: "
            f"source_language, asr_confidence (low|medium|high), "
            f"master_meaning, master_translation, uncertain_spans (array), "
            f"cue_translations (array of {n} strings), "
            f"semantic_groups (array of objects with id, source_segment_indices, "
            f"start, end, source_text, translation). "
            f"start/end on groups are informational only."
        ),
        body, api_key, model,
    )
    data = _extract_json(raw)
    script = _normalize_dub_script(data, n)
    if not script["master_meaning"]:
        script["master_meaning"] = script["master_translation"][:240]
    if not script["master_translation"]:
        joined = " ".join(t for t in script["cue_translations"] if t)
        if not joined:
            joined = " ".join(
                g["translation"] for g in script["semantic_groups"])
        script["master_translation"] = joined
    # Đồng bộ master nếu groups/cues dài hơn (tránh master cắt cụt)
    script["master_translation"] = _fullest_master(
        script["master_translation"],
        script.get("cue_translations") or [],
        script.get("semantic_groups") or [],
    )

    script["output_lang_code"] = expected_code
    script["output_lang_name"] = lang_name
    script["language_repaired"] = False

    mismatched = _cue_lang_mismatch(script["cue_translations"], expected_code)
    if mismatched:
        script = _repair_cue_languages(
            script, mismatched, lang_name, api_key, model)
        script["output_lang_code"] = expected_code
        script["output_lang_name"] = lang_name
        script["language_repaired"] = True
        script["language_repair_indices"] = mismatched
        still = _cue_lang_mismatch(script["cue_translations"], expected_code)
        if still:
            raise OpenAIError(
                f"cue_translations vẫn lệch ngôn ngữ sau repair "
                f"(expected={expected_code}, cues={still}): "
                + " | ".join(
                    f"[{i}] {script['cue_translations'][i][:80]}"
                    for i in still))

    script.setdefault("cues_filled", False)
    script.setdefault("cues_filled_indices", [])
    script.setdefault("empty_cue_indices", [])
    script.setdefault("garbled_silent_indices", [])
    script.setdefault("trailing_cta_filled", False)
    empty_idxs = _empty_asr_cue_indices(segs, script["cue_translations"])
    if empty_idxs:
        script = _fill_empty_cues(
            script, segs, empty_idxs, lang_name, api_key, model)
        script["output_lang_code"] = expected_code
        script["output_lang_name"] = lang_name
        still_empty = _empty_asr_cue_indices(segs, script["cue_translations"])
        script["empty_cue_indices"] = still_empty
    script = _apply_trailing_cta_fill(script, segs)
    script["output_lang_code"] = expected_code
    script["output_lang_name"] = lang_name
    return script


def rephrase_for_duration(text: str, seconds: float, api_key: str, *,
                          master_meaning: str, prev_text: str = "",
                          next_text: str = "", shorter: bool = True,
                          target_lang: str = "", lang_name: str = "",
                          model: str = DEFAULT_MODEL) -> str:
    """Chỉ đổi phrasing để vừa ~seconds. Khóa intent dòng — có thể CANNOT_FIT."""
    text = (text or "").strip()
    if not text:
        raise OpenAIError("Không có text để rephrase")
    sec = max(float(seconds), 0.4)
    tw = target_words(sec)
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


def _asr_empty_should_stay_silent(asr_text: str, master_translation: str = "") -> bool:
    """True → không fill TTS; ASR rác / không suy được từ master.

    Ưu tiên looks_garbled; thêm heuristic ≥2 token TitleCase trong câu dài
    (kiểu 'Sing Jemah kanil...').
    """
    asr = (asr_text or "").strip()
    if not asr:
        return True
    if looks_garbled(asr):
        return True
    words = [w for w in re.sub(r"[,.!?]+", " ", asr).split() if w]
    if len(words) >= 4:
        caps = [
            w for w in words
            if len(w) > 1 and w[0].isupper() and w[1:].islower()
        ]
        if len(caps) >= 2:
            return True
    # Không bắt buộc overlap master (master có thể khác script khi dịch ID→VI)
    _ = master_translation
    return False


def _empty_asr_cue_indices(segs, cue_translations) -> list:
    """ASR còn text nhưng cue_translations[i] rỗng."""
    out = []
    cues = list(cue_translations or [])
    for i, seg in enumerate(segs or []):
        asr = (seg.get("text") or "").replace("\n", " ").strip()
        if not asr:
            continue
        cue = cues[i].strip() if i < len(cues) and cues[i] else ""
        if not cue:
            out.append(i)
    return out


def _fill_empty_cues(script: dict, segs, empty_idxs: list, lang_name: str,
                     api_key: str, model: str = DEFAULT_MODEL) -> dict:
    """1 lần: điền cue rỗng từ master; garbled → silence bắt buộc."""
    import json
    cues = list(script["cue_translations"])
    n = len(cues)
    master_t = script.get("master_translation") or ""
    out = dict(script)
    unc = list(out.get("uncertain_spans") or [])
    merged = list(cues)

    silent_idxs = []
    fill_idxs = []
    for i in empty_idxs:
        asr = (segs[i].get("text") or "").replace("\n", " ").strip()
        if _asr_empty_should_stay_silent(asr, master_t):
            silent_idxs.append(i)
            merged[i] = ""
            for tag in (f"garbled_asr[{i}]", f"empty_cue[{i}]"):
                if tag not in unc:
                    unc.append(tag)
        else:
            fill_idxs.append(i)

    filled_idxs = []
    if fill_idxs:
        asr_bits = []
        for i in fill_idxs:
            t = (segs[i].get("text") or "").replace("\n", " ").strip()
            asr_bits.append({"index": i, "asr_text": t})
        payload = {
            "fill_indices": fill_idxs,
            "asr_for_empty": asr_bits,
            "cue_translations": merged,
            "master_translation": master_t,
            "master_meaning": script.get("master_meaning") or "",
            "uncertain_spans": unc,
        }
        raw = _chat(
            (
                f"Some ASR cues have text but cue_translations is empty. "
                f"Write spoken lines in {lang_name} ONLY for indices "
                f"{fill_idxs}, and ONLY if the meaning can be inferred from "
                f"master_translation (same ad). "
                f"Leave all other cues unchanged. "
                f"If ASR does not match any idea in master_translation, "
                f"keep that cue as \"\" and add to uncertain_spans. "
                f"Do NOT phonetically translate or invent meaning from "
                f"gibberish ASR. Do not invent product claims or prices. "
                f"Return ONLY valid JSON: "
                f"{{\"cue_translations\": [exactly {n} strings], "
                f"\"uncertain_spans\": [array]}}."
            ),
            json.dumps(payload, ensure_ascii=False),
            api_key, model,
        )
        data = _extract_json(raw)
        fixed = data.get("cue_translations")
        if not isinstance(fixed, list) or len(fixed) != n:
            raise OpenAIError(
                f"empty-cue fill trả cue_translations sai độ dài (cần {n})")
        for i in fill_idxs:
            new_t = (
                fixed[i] if isinstance(fixed[i], str) else str(fixed[i] or "")
            ).strip()
            if new_t:
                merged[i] = new_t
                filled_idxs.append(i)
            else:
                merged[i] = ""
                tag = f"empty_cue[{i}]"
                if tag not in unc:
                    unc.append(tag)
        for u in data.get("uncertain_spans") or []:
            if u not in unc:
                unc.append(u)

    # Post-guard: index silent/garbled không được có text
    for i in silent_idxs:
        merged[i] = ""

    out["cue_translations"] = merged
    out["uncertain_spans"] = unc
    out["cues_filled"] = bool(filled_idxs)
    out["cues_filled_indices"] = filled_idxs
    out["empty_cue_indices"] = [
        i for i in empty_idxs if not (merged[i] or "").strip()
    ]
    out["garbled_silent_indices"] = silent_idxs
    return out


def _fullest_master(master: str, cues, groups) -> str:
    """Chọn bản master dài nhất hợp lệ (tránh master_translation cắt cụt)."""
    candidates = [(master or "").strip()]
    cue_join = " ".join((t or "").strip() for t in (cues or []) if (t or "").strip())
    if cue_join:
        candidates.append(cue_join)
    group_join = " ".join(
        (g.get("translation") or "").strip()
        for g in (groups or []) if (g.get("translation") or "").strip())
    if group_join:
        candidates.append(group_join)
    return max(candidates, key=len)


def _closing_line_from_master(master: str) -> str:
    """Câu kết / CTA = sentence cuối của master."""
    text = (master or "").strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?…])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return text
    return parts[-1]


def _trailing_silent_span(segs, cues):
    """(first_idx, last_idx, span_seconds) của đuôi cue rỗng, hoặc None."""
    cues = list(cues or [])
    n = len(cues)
    if not n or not segs:
        return None
    i = n - 1
    while i >= 0 and not (cues[i] or "").strip():
        i -= 1
    first = i + 1
    if first >= n:
        return None
    last = n - 1
    ss = float(segs[first].get("speech_start", segs[first]["start"]))
    se = float(segs[last].get("speech_end", segs[last]["end"]))
    return first, last, max(se - ss, 0.0)


def _short_cta_variant(cta: str) -> str:
    """Rút CTA ngắn hơn khi trùng cue trước (cùng intent, khác wording)."""
    t = (cta or "").strip()
    if not t:
        return t
    orig = t
    # Bỏ đuôi thời gian phổ biến
    t2 = re.sub(
        r"\s*(ngay\s+bây\s+giờ|ngay\s+lập\s+tức|bây\s+giờ|right\s+now|now)\s*[.!?…]*\s*$",
        ".",
        t,
        flags=re.IGNORECASE,
    )
    t2 = re.sub(r"\s+\.\s*$", ".", t2).strip()
    if t2 and t2.rstrip(".!?…") and t2 != orig:
        return t2 if t2.endswith((".", "!", "?")) else t2 + "."
    # Rút còn cụm nút / thử
    low = orig.lower()
    if "nhấn" in low or "bấm" in low or "click" in low or "tap" in low:
        if "dưới" in low or "below" in low:
            return "Nhấn bên dưới."
        return "Nhấn nút."
    if "thử" in low or "try" in low:
        return "Thử ngay."
    words = re.findall(r"\S+", orig.rstrip(".!?"))
    if len(words) > 3:
        return " ".join(words[:3]) + "."
    return orig


def _apply_trailing_cta_fill(script: dict, segs, min_span: float = 1.5) -> dict:
    """Đuôi silent/garbled dài → 1 CTA từ master (không phiên âm ASR)."""
    out = dict(script)
    out.setdefault("trailing_cta_filled", False)
    cues = list(out.get("cue_translations") or [])
    info = _trailing_silent_span(segs, cues)
    if not info:
        return out
    first, last, span = info
    if span < min_span:
        return out
    master = _fullest_master(
        out.get("master_translation") or "",
        cues,
        out.get("semantic_groups") or [],
    )
    out["master_translation"] = master
    cta = _closing_line_from_master(master)
    if not cta:
        return out
    prev = ""
    for j in range(first - 1, -1, -1):
        if (cues[j] or "").strip():
            prev = cues[j].strip()
            break
    # Trùng cue trước → dùng biến thể ngắn (vẫn fill, tránh miệng nói mà câm)
    if prev and cta == prev:
        cta = _short_cta_variant(cta)
        if not cta or cta == prev:
            cta = "Thử ngay."
        out["trailing_cta_variant"] = True
    cues[first] = cta
    out["cue_translations"] = cues
    out["trailing_cta_filled"] = True
    out["trailing_cta_index"] = first
    gs = [x for x in (out.get("garbled_silent_indices") or []) if x != first]
    out["garbled_silent_indices"] = gs
    unc = [
        u for u in (out.get("uncertain_spans") or [])
        if u not in (f"garbled_asr[{first}]", f"empty_cue[{first}]", first, str(first))
    ]
    out["uncertain_spans"] = unc
    return out


def _selfcheck():
    assert _parse_numbered("1. xin chao\n2. tam biet", 2) == ["xin chao", "tam biet"]
    assert _parse_numbered("2) hai\n1) mot", 2) == ["mot", "hai"]      # sai thứ tự
    assert _parse_numbered("Sure!\n1. a\n2. b\n", 2) == ["a", "b"]     # bỏ lời dẫn
    for raw, n in (("1. a", 2), ("1. a\n2. b\n3. c", 2), ("khong danh so", 1)):
        try:
            _parse_numbered(raw, n)
        except OpenAIError:
            pass
        else:
            raise AssertionError(f"lệch số dòng phải ném: {raw!r}")
    assert target_words(10) == 24 and word_count("a b c") == 3
    assert _in_band(24, 24) and not _in_band(5, 24)
    rule = _length_rule(30)
    assert "30" in rule and "dubbing" in rule.lower()
    assert "supporting detail" not in rule.lower()
    assert _length_rule(0) == ""
    # Heuristic tổng quát — không gắn case ASR cụ thể
    assert looks_garbled("")  # rỗng
    assert looks_garbled("Alpha Betagamma Deltaepsilon")  # TitleCase, không kết câu
    assert looks_garbled("abcdefgh ijklmnop qrstuvwx")  # nhiều token dài
    assert not looks_garbled("we need a loan but fear high fees.")
    assert _mostly_same_script("hello world abcdef", "hello world abcdef")
    assert _mostly_same_script("abcdef ghijkl", "prefix abcdef ghijkl suffix")
    assert not _mostly_same_script("abcdef ghijkl", "hoan toan khac roi")
    assert "garbled" in _asr_context_rule("Vietnamese").lower()
    # Language mismatch heuristic (VI diacritics)
    assert _resolve_output_lang("same", {"language": "id"})[0] == "id"
    assert "Indonesian" in _resolve_output_lang("same", {"language": "id"})[1]
    assert _resolve_output_lang("VI", {})[0] == "vi"
    id_cues = [
        "Butuh pinjaman tapi takut bunganya besar?",
        "Pakai aplikasi ini untuk menghitung bunga.",
        "Klik tombol di bawah sekarang.",
    ]
    assert _cue_lang_mismatch(id_cues, "id") == []
    mixed = list(id_cues)
    mixed[2] = "Nhấn nút bên dưới ngay bây giờ."
    assert _cue_lang_mismatch(mixed, "id") == [2]
    assert _cue_lang_mismatch(mixed, "vi")  # first ID lines lack VI marks
    assert 2 not in _cue_lang_mismatch(
        ["Nhấn nút bên dưới ngay bây giờ để đăng ký ngay."], "vi")
    sample = _normalize_dub_script({
        "source_language": "id",
        "asr_confidence": "low",
        "master_meaning": "loan app CTA",
        "master_translation": "Full VI script.",
        "uncertain_spans": [],
        "cue_translations": ["Một.", "Hai ba.", "CTA."],
        "semantic_groups": [
            {"id": 1, "source_segment_indices": [0], "start": 0.1, "end": 2.0,
             "source_text": "a", "translation": "Một."},
            {"id": 2, "source_segment_indices": [1, 2], "start": 3.0, "end": 9.0,
             "source_text": "b", "translation": "Hai ba. CTA."},
        ],
    }, 3)
    assert len(sample["semantic_groups"]) == 2
    assert sample["semantic_groups"][1]["source_segment_indices"] == [1, 2]
    assert sample["cue_translations"] == ["Một.", "Hai ba.", "CTA."]
    # Fallback: 1 group = 1 cue
    one_each = _normalize_dub_script({
        "master_meaning": "m", "master_translation": "t",
        "semantic_groups": [
            {"id": 1, "source_segment_indices": [0], "start": 0, "end": 1,
             "translation": "A"},
            {"id": 2, "source_segment_indices": [1], "start": 2, "end": 3,
             "translation": "B"},
        ],
    }, 2)
    assert one_each["cue_translations"] == ["A", "B"]
    # Span group without cue_translations → fail
    try:
        _normalize_dub_script({
            "master_meaning": "m",
            "semantic_groups": [
                {"id": 1, "source_segment_indices": [0, 1], "start": 0, "end": 9,
                 "translation": "merged"},
            ],
        }, 2)
    except OpenAIError:
        pass
    else:
        raise AssertionError("span group thiếu cue_translations phải ném")
    assert _extract_json('{"a":1}')["a"] == 1
    assert _extract_json("```json\n{\"a\": 2}\n```")["a"] == 2
    try:
        _normalize_dub_script({"master_meaning": "x"}, 1)
    except OpenAIError:
        pass
    else:
        raise AssertionError("thiếu cue_translations/groups phải ném")
    # Rephrase prompt locks same intent / no borrow; balanced longer/shorter
    short_p = _rephrase_system_prompt(
        seconds=1.5, target_words_n=4, shorter=True,
        master_meaning="app pitch", prev_text="a", next_text="b",
        lang_name="Vietnamese")
    assert "same intent" in short_p.lower()
    assert "do not borrow" in short_p.lower()
    assert "aggressive condense" in short_p.lower()
    long_p = _rephrase_system_prompt(
        seconds=3.0, target_words_n=8, shorter=False,
        master_meaning="app pitch", prev_text="", next_text="",
        lang_name="Indonesian")
    assert "do not borrow" in long_p.lower() or "Do not import" in long_p
    assert "same intent" in long_p.lower()
    assert "light expansion" in long_p.lower()
    assert "aim close" in long_p.lower()
    assert "stay near" in short_p.lower() or "target band" in short_p.lower()
    # Empty ASR cue helper + garbled → silence
    segs_e = [
        {"text": "hello there friends"},
        {"text": "Sing Jemah kanil noise"},
        {"text": ""},
    ]
    assert _empty_asr_cue_indices(segs_e, ["Xin chào.", "", ""]) == [1]
    assert _empty_asr_cue_indices(segs_e, ["a", "b", ""]) == []
    assert _asr_empty_should_stay_silent(
        "Sing Jemah kanil, tapi cepetan selamat tinggi.")
    assert not _asr_empty_should_stay_silent(
        "Butuh pinjaman tapi takut bunganya besar?")
    assert not _asr_empty_should_stay_silent(
        "Pakai aplikasi ini untuk menghitung bunga pinjaman.")
    # Trailing CTA fill
    assert _closing_line_from_master(
        "Câu một. Nhấn nút bên dưới ngay.") == "Nhấn nút bên dưới ngay."
    assert "CTA cuối" in _fullest_master(
        "Ngắn.", ["A.", "CTA cuối."], [{"translation": "A. B. CTA cuối."}])
    segs_t = [
        {"start": 0, "end": 2, "speech_start": 0, "speech_end": 2, "text": "a"},
        {"start": 2, "end": 4, "speech_start": 2, "speech_end": 4, "text": "b"},
        {"start": 9.0, "end": 11.0, "speech_start": 9.0, "speech_end": 11.0,
         "text": "Sing Jemah kanil junk here"},
    ]
    script_t = {
        "master_translation": (
            "Bạn cần vay? Dùng app. Nhấn nút bên dưới ngay bây giờ."),
        "cue_translations": ["Bạn cần vay?", "Dùng app.", ""],
        "semantic_groups": [],
        "garbled_silent_indices": [2],
        "uncertain_spans": ["garbled_asr[2]"],
    }
    filled = _apply_trailing_cta_fill(script_t, segs_t, min_span=1.5)
    assert filled["trailing_cta_filled"]
    assert filled["trailing_cta_index"] == 2
    assert "Nhấn nút" in filled["cue_translations"][2]
    assert "Jemah" not in filled["cue_translations"][2]
    # Trùng CTA với cue trước → biến thể ngắn, vẫn fill
    assert _short_cta_variant("Nhấn nút bên dưới ngay bây giờ.") != (
        "Nhấn nút bên dưới ngay bây giờ.")
    segs_dup = [
        {"start": 0, "end": 2, "speech_start": 0.0, "speech_end": 2.0, "text": "a"},
        {"start": 7.5, "end": 8.9, "speech_start": 7.57, "speech_end": 8.93,
         "text": "klik"},
        {"start": 9.1, "end": 11.0, "speech_start": 9.19, "speech_end": 10.99,
         "text": "Sing Jemah kanil junk"},
    ]
    script_dup = {
        "master_translation": (
            "Bạn cần vay? Dùng app. Nhấn nút bên dưới ngay bây giờ."),
        "cue_translations": [
            "Bạn cần vay?",
            "Nhấn nút bên dưới ngay bây giờ.",
            "",
        ],
        "semantic_groups": [],
        "garbled_silent_indices": [2],
        "uncertain_spans": ["garbled_asr[2]"],
    }
    filled_dup = _apply_trailing_cta_fill(script_dup, segs_dup, min_span=1.5)
    assert filled_dup["trailing_cta_filled"]
    assert filled_dup.get("trailing_cta_variant")
    assert filled_dup["cue_translations"][2]
    assert filled_dup["cue_translations"][2] != (
        "Nhấn nút bên dưới ngay bây giờ.")
    assert "Jemah" not in filled_dup["cue_translations"][2]
    print("openai_translate_api.py self-check OK")


if __name__ == "__main__":
    _selfcheck()
