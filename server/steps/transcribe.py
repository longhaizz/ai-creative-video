"""Turn the mix into timed cues with faster-whisper.

Runs in this process on the GPU. A Cue is a Whisper segment. Long takes are
split so TTS can fill the slot. speech_start and speech_end come from word
times: Whisper often opens a cue early and closes it late, and fitting TTS
to those wider times makes every sentence sound rushed.

Every cue is SPEAKER_00. There is no diarization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from server.jobs import PipelineError

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
SPEAKER_00 = "SPEAKER_00"
MAX_UTTERANCE_SECONDS = 6.0
MAX_SENTENCES = 2
WORD_GAP_SECONDS = 0.35
MAX_WORD_SECONDS = 1.2
_SENTENCE_END = re.compile(r"[.!?。！？]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


class WhisperModels:
    """Keeps the loaded whisper models between jobs.

    A job may ask for medium or large-v3. They stay in memory. They are
    small next to LatentSync, and there is room on a 48 GB card.
    """

    name = "whisper"

    def __init__(self, default_model: str = "medium", device: str = "cuda"):
        self.default_model = default_model
        self.device = device
        self._loaded: dict[str, object] = {}

    def load(self) -> None:
        self.get(self.default_model)

    def get(self, size: str):
        if size not in WHISPER_MODELS:
            raise PipelineError(f"Unknown whisper model: {size}", code="invalid_input")
        if size not in self._loaded:
            from faster_whisper import WhisperModel

            self._loaded[size] = WhisperModel(
                size, device=self.device, compute_type="float16"
            )
        return self._loaded[size]


def transcribe(models: WhisperModels, audio: Path, size: str, ctx=None):
    """Return (cues, meta). Uses `size` only — no large-v3 retry."""
    audio = Path(audio)
    if not audio.is_file():
        raise PipelineError(f"No such audio file: {audio.name}")

    cues, meta = _run_once(models, audio, size, ctx)
    quality = asr_quality(cues, meta["language_probability"])
    meta["confidence"] = _confidence(quality, meta["language_probability"])
    meta["whisper_model"] = size
    if ctx is not None:
        ctx.log(
            f"{len(cues)} cues after split, language {meta['language']} "
            f"(p={meta['language_probability']:.2f}), "
            f"confidence {meta['confidence']}"
        )
    return cues, meta


def _confidence(quality: dict, language_probability: float) -> str:
    if not quality["ok"]:
        return "low"
    return "high" if language_probability >= 0.85 else "medium"


def _run_once(models: WhisperModels, audio: Path, size: str, ctx):
    if ctx is not None:
        ctx.step(f"Reading the speech (whisper {size})")

    model = models.get(size)
    segments, info = model.transcribe(
        str(audio), language=None, vad_filter=True, word_timestamps=True
    )
    language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)

    raw_segments = []
    cues = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            raw_segments.append({
                "id": len(raw_segments),
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": text,
            })
        cues.extend(_cues_from_segment(segment))

    cues = [cue for cue in cues if cue["text"]]
    if not cues:
        raise PipelineError("No speech was found in the video", code="invalid_input")

    _write_transcript(audio, size, info, raw_segments, ctx)

    return cues, {
        "language": getattr(info, "language", "") or "",
        "language_probability": language_probability,
    }


def _write_transcript(audio: Path, size: str, info, raw_segments: list[dict], ctx):
    """CLI-shaped JSON of what Whisper heard, before TTS splits."""
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    payload = {
        "file": Path(audio).name,
        "language": getattr(info, "language", "") or "",
        "language_probability": round(
            float(getattr(info, "language_probability", 0.0) or 0.0), 4
        ),
        "duration": round(duration, 3),
        "model": size,
        "segments": raw_segments,
    }
    dest = Path(audio).parent / "transcript.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if ctx is None:
        return
    ctx.log(
        f"Whisper heard {len(raw_segments)} segments, "
        f"language {payload['language']} -> {dest.name}"
    )
    for seg in raw_segments:
        ctx.log(f"{seg['start']:.2f}-{seg['end']:.2f}  {seg['text']}")


def _cues_from_segment(segment) -> list[dict]:
    text = (segment.text or "").strip()
    if not text:
        return []
    avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
    no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
    words = []
    for word in segment.words or []:
        token = word.word or ""
        if not token.strip():
            continue
        start = float(word.start)
        end = _clamp_word_end(start, float(word.end))
        words.append({
            "word": token,
            "start": start,
            "end": end,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
        })
    if words:
        groups = _word_groups(words)
        cues = [
            _cue_from_group(group, avg_logprob, no_speech_prob)
            for group in groups
        ]
        if len(cues) == 1:
            cues[0]["start"] = float(segment.start)
            cues[0]["end"] = float(segment.end)
        return cues
    start = float(segment.start)
    end = float(segment.end)
    if end <= start:
        end = start + 0.4
    return [
        _make_cue(left, right, piece, avg_logprob, no_speech_prob)
        for left, right, piece in split_long_utterance(start, end, text)
    ]


def _cue_from_group(group: list[dict], avg_logprob: float, no_speech_prob: float) -> dict:
    text = _join_tokens([item["word"] for item in group])
    speech_start = float(group[0]["start"])
    speech_end = float(group[-1]["end"])
    if speech_end <= speech_start:
        speech_end = speech_start + 0.15
    return _make_cue(
        speech_start, speech_end, text, avg_logprob, no_speech_prob,
        start=speech_start, end=speech_end,
    )


def _make_cue(
    speech_start: float,
    speech_end: float,
    text: str,
    avg_logprob: float,
    no_speech_prob: float,
    start: float | None = None,
    end: float | None = None,
) -> dict:
    left = float(start if start is not None else speech_start)
    right = float(end if end is not None else speech_end)
    if right <= left:
        right = left + 0.4
    if speech_end <= speech_start:
        speech_end = speech_start + 0.4
    return {
        "start": left,
        "end": right,
        "speech_start": float(speech_start),
        "speech_end": float(speech_end),
        "text": (text or "").strip(),
        "speaker_id": SPEAKER_00,
        "avg_logprob": float(avg_logprob),
        "no_speech_prob": float(no_speech_prob),
    }


def _clamp_word_end(start: float, end: float) -> float:
    if end - start > MAX_WORD_SECONDS:
        return start + MAX_WORD_SECONDS
    return end


def _join_tokens(tokens: list[str]) -> str:
    if any(token[:1].isspace() for token in tokens):
        return "".join(tokens).strip()
    return " ".join(token.strip() for token in tokens if token.strip())


def _sentence_list(text: str) -> list[str]:
    return [
        part.strip()
        for part in _SENTENCE_SPLIT.split((text or "").strip())
        if part.strip()
    ]


def _word_groups(words: list[dict]) -> list[list[dict]]:
    if not words:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = []

    def flush():
        nonlocal current
        if current:
            groups.append(current)
            current = []

    def sentence_count(group: list[dict]) -> int:
        return sum(1 for item in group if _SENTENCE_END.search(item["word"]))

    for index, word in enumerate(words):
        if current:
            gap = word["start"] - current[-1]["end"]
            span = word["end"] - current[0]["start"]
            split = gap >= WORD_GAP_SECONDS
            if _SENTENCE_END.search(current[-1]["word"]):
                split = True
            if span > MAX_UTTERANCE_SECONDS and len(current) >= 2:
                split = True
            if split:
                flush()
        current.append(word)
        if sentence_count(current) >= MAX_SENTENCES and index + 1 < len(words):
            flush()
    flush()

    out: list[list[dict]] = []
    for group in groups:
        out.extend(_split_oversized_word_group(group))
    return out


def _split_oversized_word_group(group: list[dict]) -> list[list[dict]]:
    span = group[-1]["end"] - group[0]["start"]
    if span <= MAX_UTTERANCE_SECONDS or len(group) <= 1:
        return [group]

    best_gap = 0.0
    best_index = len(group) // 2
    for index in range(1, len(group)):
        gap = group[index]["start"] - group[index - 1]["end"]
        if gap > best_gap:
            best_gap = gap
            best_index = index

    if best_gap >= WORD_GAP_SECONDS * 0.5:
        return (
            _split_oversized_word_group(group[:best_index])
            + _split_oversized_word_group(group[best_index:])
        )
    mid = max(1, len(group) // 2)
    return (
        _split_oversized_word_group(group[:mid])
        + _split_oversized_word_group(group[mid:])
    )


def split_long_utterance(start: float, end: float, text: str) -> list[tuple[float, float, str]]:
    """Split a text-only span that is too long for TTS."""
    text = (text or "").strip()
    sentences = _sentence_list(text)
    span = end - start
    if len(sentences) <= MAX_SENTENCES and span <= MAX_UTTERANCE_SECONDS:
        return [(start, end, text)]
    if len(sentences) <= 1:
        return [(start, end, text)]
    weights = [max(len(sentence), 1) for sentence in sentences]
    total = sum(weights)
    cursor = start
    pieces = []
    for index, sentence in enumerate(sentences):
        if index == len(sentences) - 1:
            nxt = end
        else:
            nxt = cursor + span * (weights[index] / total)
        pieces.append((cursor, nxt, sentence))
        cursor = nxt
    return pieces


def asr_quality(cues: list[dict], language_probability: float = 0.0) -> dict:
    """Judge the whole transcript: {ok, level, reasons}.

    Whisper never says "I could not hear that". It writes something, and
    when the sound is bad the something is confident nonsense. These are the
    marks that nonsense leaves. Used for the confidence log only.
    """
    reasons = []
    count = len(cues) or 1
    probability = float(language_probability or 0.0)
    if 0 < probability < 0.75:
        reasons.append(f"language_prob={probability:.2f}<0.75")

    garbled = 0
    low_confidence = 0
    for cue in cues:
        text = (cue.get("text") or "").strip()
        words = [word for word in re.sub(r"[,.!?]+", " ", text).split() if word]
        if len(words) >= 2:
            titled = sum(
                1 for word in words
                if len(word) > 1 and word[0].isupper() and word[1:].islower()
            )
            if titled / len(words) >= 0.6 and not text.endswith((".", "!", "?")):
                garbled += 1
            longish = sum(1 for word in words if len(word) >= 8)
            if len(words) >= 3 and longish / len(words) >= 0.5:
                garbled += 1
        if float(cue.get("avg_logprob") or 0) < -1.0:
            low_confidence += 1

    if garbled / count >= 0.35:
        reasons.append(f"garbled_cues={garbled}/{count}")
    if low_confidence / count >= 0.35:
        reasons.append(f"low_confidence_cues={low_confidence}/{count}")

    ok = not reasons
    return {
        "ok": ok,
        "level": "good" if ok else "bad",
        "reasons": reasons,
        "language_probability": probability,
    }


def cue_needs_review(cue: dict) -> str:
    """A short reason this single cue looks wrong, or an empty string.

    High no_speech_prob is not a reason. Quiet Hindi VO often scores ~0.97
    and is still speech.
    """
    text = (cue.get("text") or "").strip()
    if not text:
        return "empty"
    if float(cue.get("avg_logprob") or 0) < -1.0:
        return f"avg_logprob={cue['avg_logprob']:.2f}"
    words = [word for word in text.replace(",", " ").split() if word]
    if len(words) >= 3:
        titled = sum(1 for word in words if word[:1].isupper() and word[1:].islower())
        if titled / len(words) >= 0.7 and not text.endswith((".", "!", "?")):
            return "odd capital letters, possibly invented"
    return ""
