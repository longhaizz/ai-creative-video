"""Glue leftover Whisper crumbs and cut a clone-reference wav.

Segmentation itself is Demucs + Whisper in the main process. This module
only post-processes cues: fold prefix crumbs into the next line, merge
short CTAs, and cut one SPEAKER_00 reference from the vocals stem.
"""

from __future__ import annotations

import re
from pathlib import Path

from server import config
from server.jobs import PipelineError
from server.steps import audio
from server.steps.transcribe import SPEAKER_00

MIN_REF_SECONDS = 3.0
# Whisper often leaves a 50–300ms first-word crumb ("I", "You're", "Easy")
# on the previous cut, then the full phrase in the next window. TTS
# cannot speak those, and the rewrite model merges them and returns the
# wrong cue_translations length.
PREFIX_CRUMB_SECONDS = 0.35
PREFIX_CRUMB_GAP_SECONDS = 0.8
SHORT_CTA_SECONDS = 0.6
SHORT_CTA_GAP_SECONDS = 0.5
SHORT_CTA_WORDS = (2, 5)


def attach_refs(cues: list[dict], vocals: Path, out_dir: Path) -> list[dict]:
    """Glue crumbs, force SPEAKER_00, cut one clone wav for the clip."""
    cues = glue_prefix_crumbs(cues)
    if not cues:
        raise PipelineError("No speech was found in the video", code="invalid_input")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cue in cues:
        cue["speaker_id"] = SPEAKER_00
    spans = [(float(c["start"]), float(c["end"])) for c in cues]
    ref = reference_for_spans(vocals, spans, out_dir / "ref_SPEAKER_00.wav")
    for cue in cues:
        cue["ref_wav"] = str(ref)
    return cues


def _plain_cue_text(text: str) -> str:
    return re.sub(r"[^\w]+", "", (text or ""), flags=re.UNICODE).casefold()


def _speech_dur(cue: dict) -> float:
    start = float(cue.get("speech_start", cue["start"]))
    end = float(cue.get("speech_end", cue["end"]))
    return max(end - start, 0.0)


def is_prefix_crumb(short: dict, full: dict) -> bool:
    """True when `short` is a leftover first-word of `full` (same speaker)."""
    if (short.get("speaker_id") or SPEAKER_00) != (
            full.get("speaker_id") or SPEAKER_00):
        return False
    if _speech_dur(short) > PREFIX_CRUMB_SECONDS:
        return False
    short_text = (short.get("text") or "").strip()
    full_text = (full.get("text") or "").strip()
    if not short_text or not full_text:
        return False
    if len(short_text.split()) > 2:
        return False
    plain_short = _plain_cue_text(short_text)
    plain_full = _plain_cue_text(full_text)
    if not plain_short or plain_short == plain_full:
        return False
    if not plain_full.startswith(plain_short):
        return False
    gap = float(full.get("start", 0)) - float(short.get("end", 0))
    return -0.05 <= gap <= PREFIX_CRUMB_GAP_SECONDS


def is_short_cta(cue: dict, nxt: dict | None) -> bool:
    """A 2–5 word aside too short to speak, sitting on the next line.

    'Coba sekarang' 0.37s then 'sebelum semua orang tahu' should be one cue.
    """
    if nxt is None:
        return False
    if (cue.get("speaker_id") or SPEAKER_00) != (
            nxt.get("speaker_id") or SPEAKER_00):
        return False
    words = len((cue.get("text") or "").split())
    lo, hi = SHORT_CTA_WORDS
    if not (lo <= words <= hi):
        return False
    if _speech_dur(cue) >= SHORT_CTA_SECONDS:
        return False
    gap = float(nxt.get("start", 0)) - float(cue.get("end", 0))
    return 0 <= gap < SHORT_CTA_GAP_SECONDS


def _join_cue_text(left: str, right: str) -> str:
    a = (left or "").strip()
    b = (right or "").strip()
    if not a:
        return b
    if not b:
        return a
    return f"{a} {b}"


def _merge_crumb_into(crumb: dict, full: dict) -> dict:
    out = dict(full)
    out["start"] = min(float(crumb["start"]), float(full["start"]))
    out["end"] = max(float(crumb["end"]), float(full["end"]))
    out["speech_start"] = min(
        float(crumb.get("speech_start", crumb["start"])),
        float(full.get("speech_start", full["start"])),
    )
    out["speech_end"] = max(
        float(crumb.get("speech_end", crumb["end"])),
        float(full.get("speech_end", full["end"])),
    )
    out["no_speech_prob"] = min(
        float(crumb.get("no_speech_prob") or 1.0),
        float(full.get("no_speech_prob") or 1.0),
    )
    return out


def glue_prefix_crumbs(cues: list[dict]) -> list[dict]:
    """Fold Whisper leftover prefixes into the following full phrase."""
    items = [dict(cue) for cue in cues if (cue.get("text") or "").strip()]
    index = 0
    while index < len(items) - 1:
        if is_prefix_crumb(items[index], items[index + 1]):
            items[index + 1] = _merge_crumb_into(items[index], items[index + 1])
            del items[index]
            if index:
                index -= 1
            continue
        index += 1
    merged = []
    index = 0
    while index < len(items):
        if index < len(items) - 1 and is_short_cta(items[index], items[index + 1]):
            nxt = _merge_crumb_into(items[index], items[index + 1])
            nxt["text"] = _join_cue_text(
                items[index].get("text") or "",
                items[index + 1].get("text") or "",
            )
            merged.append(nxt)
            index += 2
            continue
        merged.append(items[index])
        index += 1
    return merged


def reference_for_spans(
    vocals: Path, spans: list[tuple[float, float]], dest: Path,
) -> Path:
    """One clone wav for a speaker: longest clip, or concat if all are short.

    Never grows a short cue into a neighbour's time.
    """
    cleaned = [(float(s), float(e)) for s, e in spans if e > s]
    if not cleaned:
        raise PipelineError("No speech to clone from", code="internal")
    longest = max(cleaned, key=lambda pair: pair[1] - pair[0])
    if longest[1] - longest[0] >= MIN_REF_SECONDS or len(cleaned) == 1:
        return pad_reference(vocals, longest[0], longest[1], dest)
    parts = []
    for index, (start, end) in enumerate(sorted(cleaned)):
        part = dest.with_name(f"{dest.stem}_p{index:02d}{dest.suffix}")
        parts.append(pad_reference(vocals, start, end, part))
    return _concat_wavs(parts, dest)


def pad_reference(vocals: Path, start: float, end: float, dest: Path) -> Path:
    """Cut exactly this span from vocals."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = audio.duration(vocals)
    left = max(float(start), 0.0)
    right = min(float(end), total) if total > 0 else float(end)
    if right <= left:
        right = left + 0.05
    length = max(right - left, 0.05)
    audio.run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-ss", f"{left:.3f}", "-t", f"{length:.3f}",
        "-i", str(vocals), "-c:a", "pcm_s16le", str(dest),
    ])
    return dest


def _concat_wavs(parts: list[Path], dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    listing = dest.with_name(dest.stem + "_concat.txt")
    lines = []
    for part in parts:
        path = Path(part).resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{path}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audio.run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:a", "pcm_s16le", str(dest),
    ])
    return dest
