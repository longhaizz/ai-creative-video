"""Turn the voice track into timed cues with faster-whisper.

Ported from spy-ads subtitle_api.py. Two things are gone: the model now
runs on the GPU, and it runs in this process. The desktop tool had to push
whisper into a child process because ctranslate2 next to Qt crashed the
whole app; there is no Qt here.

Each cue carries speech_start and speech_end, taken from the first and last
word. They are tighter than start and end, and the dub is fitted to them:
whisper often opens a cue early and closes it late, and using those wider
times makes every sentence sound rushed.
"""

from __future__ import annotations

import re
from pathlib import Path

from server.jobs import PipelineError

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")

# When the first pass looks this bad, it is worth paying for the big model.
RETRY_MODEL = "large-v3"


class WhisperModels:
    """Keeps the loaded whisper models between jobs.

    A job may ask for medium and then fall back to large-v3, and the next
    job usually asks for the same one again, so both stay in memory. They
    are small next to LatentSync, and there is room on a 48 GB card.
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
    """Return (cues, meta). Retries with a bigger model if the result is poor."""
    audio = Path(audio)
    if not audio.is_file():
        raise PipelineError(f"No such audio file: {audio.name}")

    cues, meta = _run_once(models, audio, size, ctx)
    quality = asr_quality(cues, meta["language_probability"])

    if not quality["ok"] and size != RETRY_MODEL:
        if ctx is not None:
            ctx.log(f"The transcript looks poor ({', '.join(quality['reasons'])})")
            ctx.log(f"Trying again with {RETRY_MODEL}")
        cues, meta = _run_once(models, audio, RETRY_MODEL, ctx)
        quality = asr_quality(cues, meta["language_probability"])

    meta["confidence"] = _confidence(quality, meta["language_probability"])
    if ctx is not None:
        ctx.log(
            f"{len(cues)} cues, language {meta['language']} "
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

    cues = []
    for segment in segments:
        words = [
            {
                "word": (word.word or "").strip(),
                "start": float(word.start),
                "end": float(word.end),
            }
            for word in (segment.words or [])
        ]
        # The word times are tighter than the segment times, so use them
        # when they are there.
        speech_start = words[0]["start"] if words else float(segment.start)
        speech_end = words[-1]["end"] if words else float(segment.end)
        if speech_end <= speech_start:
            speech_end = speech_start + 0.4
        cues.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speech_start": speech_start,
            "speech_end": speech_end,
            "text": (segment.text or "").strip(),
            "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
        })

    cues = [cue for cue in cues if cue["text"]]
    if not cues:
        raise PipelineError("No speech was found in the video", code="invalid_input")

    return cues, {
        "language": getattr(info, "language", "") or "",
        "language_probability": language_probability,
    }


def asr_quality(cues: list[dict], language_probability: float = 0.0) -> dict:
    """Judge the whole transcript: {ok, level, reasons}.

    Whisper never says "I could not hear that". It writes something, and
    when the sound is bad the something is confident nonsense. These are the
    marks that nonsense leaves.
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
            # Title Case On Every Word with no full stop is what whisper
            # produces when it is guessing at noise.
            titled = sum(
                1 for word in words
                if len(word) > 1 and word[0].isupper() and word[1:].islower()
            )
            if titled / len(words) >= 0.6 and not text.endswith((".", "!", "?")):
                garbled += 1
            # So is a run of long invented words.
            longish = sum(1 for word in words if len(word) >= 8)
            if len(words) >= 3 and longish / len(words) >= 0.5:
                garbled += 1
        if float(cue.get("avg_logprob") or 0) < -1.0:
            low_confidence += 1
        if float(cue.get("no_speech_prob") or 0) > 0.6:
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
    """A short reason this single cue looks wrong, or an empty string."""
    text = (cue.get("text") or "").strip()
    if not text:
        return "empty"
    if float(cue.get("no_speech_prob") or 0) > 0.6:
        return f"no_speech_prob={cue['no_speech_prob']:.2f}"
    if float(cue.get("avg_logprob") or 0) < -1.0:
        return f"avg_logprob={cue['avg_logprob']:.2f}"
    words = [word for word in text.replace(",", " ").split() if word]
    if len(words) >= 3:
        titled = sum(1 for word in words if word[:1].isupper() and word[1:].islower())
        if titled / len(words) >= 0.7 and not text.endswith((".", "!", "?")):
            return "odd capital letters, possibly invented"
    return ""
