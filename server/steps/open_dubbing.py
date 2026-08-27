"""Split speech with the Open Dubbing venv: Demucs, VAD, Pyannote, Whisper.

The heavy libraries live in another interpreter (see OPEN_DUBBING_PYTHON),
the way VSR does, because pyannote's transformers stack does not sit next
to LatentSync. The worker waits and still holds the GPU, so it is still
one job at a time.

The child does not translate and does not speak. It writes utterances.json
and wav clips. The main pipeline rewrites those lines with OpenAI and
speaks them with VoxCPM.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from server import config
from server.jobs import JobCancelled, PipelineError
from server.steps import audio

LOG_EVERY_SECONDS = 2.0
TAIL_LINES = 25
MIN_REF_SECONDS = 3.0
# Whisper often leaves a 50–300ms first-word crumb ("I", "You're", "Easy")
# on the previous VAD cut, then the full phrase in the next window. TTS
# cannot speak those, and the rewrite model merges them and returns the
# wrong cue_translations length.
PREFIX_CRUMB_SECONDS = 0.35
PREFIX_CRUMB_GAP_SECONDS = 0.8
NO_SPEECH_CRUMB = 0.9

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "open_dubbing_segment.py"


def build_command(video: Path, out_dir: Path, whisper_model: str) -> list[str]:
    return [
        str(config.OPEN_DUBBING_PYTHON),
        str(SCRIPT),
        str(video),
        "--out", str(out_dir),
        "--whisper-model", whisper_model,
    ]


def segment(video: Path, work: Path, whisper_model: str, ctx=None) -> dict:
    """Run the child and return {cues, meta, vocals, music}."""
    if not (config.HF_TOKEN or "").strip():
        raise PipelineError(
            "HF_TOKEN is required for Open Dubbing (Pyannote)",
            code="invalid_input",
        )
    python = Path(config.OPEN_DUBBING_PYTHON)
    if not python.is_file():
        raise PipelineError(
            f"OPEN_DUBBING_PYTHON is not a file: {python}. "
            "Install the Open Dubbing venv (see server/docs/bare-metal.md).",
            code="invalid_input",
        )
    if not SCRIPT.is_file():
        raise PipelineError(f"Missing segment script: {SCRIPT}", code="internal")

    video = Path(video).resolve()
    out_dir = Path(work).resolve() / "od"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(video, out_dir, whisper_model)

    if ctx is not None:
        ctx.step("Segmenting speech (Open Dubbing)")
        ctx.log(f"Whisper {whisper_model}, VAD + Pyannote")

    env = {
        **os.environ,
        "HF_TOKEN": config.HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": config.HF_TOKEN,
        "MPLBACKEND": "Agg",
    }
    process = subprocess.Popen(
        command,
        cwd=str(out_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    tail: list[str] = []
    last_log = 0.0
    stdout = ""
    try:
        for line in process.stdout:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            stdout += raw + "\n"
            tail.append(raw.strip())
            del tail[:-TAIL_LINES]
            if ctx is None:
                continue
            if _cancelled(ctx):
                process.kill()
                raise JobCancelled()
            now = time.monotonic()
            if now - last_log >= LOG_EVERY_SECONDS:
                last_log = now
                ctx.log(raw.strip())
    finally:
        process.stdout.close()
        process.wait()

    if process.returncode != 0:
        raise PipelineError(
            "Open Dubbing segmentation failed:\n" + "\n".join(tail),
            code="internal",
        )

    payload = _last_json_object(stdout)
    if payload is None:
        raise PipelineError("Open Dubbing produced no JSON", code="internal")
    (out_dir / "utterances.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cues_from_payload(payload, out_dir)


def cues_from_payload(payload: dict, out_dir: Path) -> dict:
    """Turn the child JSON into the cue list timed_speech already knows."""
    vocals = Path(payload["vocals"])
    music = Path(payload["no_vocals"])
    if not vocals.is_file() or not music.is_file():
        raise PipelineError("Open Dubbing did not write vocals/no_vocals", code="internal")

    utterances = list(payload.get("utterances") or [])
    if not utterances:
        raise PipelineError("No speech was found in the video", code="invalid_input")

    cues = []
    for item in utterances:
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            end = start + 0.4
        speech_start = float(item.get("speech_start", start))
        speech_end = float(item.get("speech_end", end))
        if speech_end <= speech_start:
            speech_end = speech_start + 0.15
        speaker = item.get("speaker_id") or "SPEAKER_00"
        text = (item.get("text") or "").strip()
        cues.append({
            "start": start,
            "end": end,
            "speech_start": speech_start,
            "speech_end": speech_end,
            "text": text,
            "speaker_id": speaker,
            "avg_logprob": float(item.get("avg_logprob", 0.0) or 0.0),
            "no_speech_prob": (
                float(item.get("no_speech_prob", 0.0) or 0.0)
                if text else 1.0
            ),
        })

    cues = glue_prefix_crumbs(cues)
    if not cues:
        raise PipelineError("No speech was found in the video", code="invalid_input")

    refs = {}
    for speaker in dict.fromkeys(cue["speaker_id"] for cue in cues):
        spans = [(c["start"], c["end"]) for c in cues if c["speaker_id"] == speaker]
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in speaker)
        refs[speaker] = reference_for_spans(
            vocals, spans, out_dir / f"ref_{safe}.wav",
        )
    for cue in cues:
        cue["ref_wav"] = str(refs[cue["speaker_id"]])

    meta = {
        "language": payload.get("language") or "",
        "language_probability": float(payload.get("language_probability") or 0.0),
        "whisper_model": payload.get("whisper_model") or "",
    }
    return {"cues": cues, "meta": meta, "vocals": vocals, "music": music}


def _plain_cue_text(text: str) -> str:
    return re.sub(r"[^\w]+", "", (text or ""), flags=re.UNICODE).casefold()


def _speech_dur(cue: dict) -> float:
    start = float(cue.get("speech_start", cue["start"]))
    end = float(cue.get("speech_end", cue["end"]))
    return max(end - start, 0.0)


def is_prefix_crumb(short: dict, full: dict) -> bool:
    """True when `short` is a leftover first-word of `full` (same speaker)."""
    if (short.get("speaker_id") or "SPEAKER_00") != (
            full.get("speaker_id") or "SPEAKER_00"):
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


def is_no_speech_crumb(cue: dict) -> bool:
    if float(cue.get("no_speech_prob") or 0) < NO_SPEECH_CRUMB:
        return False
    if _speech_dur(cue) > 0.5:
        return False
    return len((cue.get("text") or "").split()) <= 2


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
    items = [dict(cue) for cue in cues]
    index = 0
    while index < len(items) - 1:
        if is_prefix_crumb(items[index], items[index + 1]):
            items[index + 1] = _merge_crumb_into(items[index], items[index + 1])
            del items[index]
            if index:
                index -= 1
            continue
        index += 1
    return [cue for cue in items if not is_no_speech_crumb(cue)]


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


def _last_json_object(text: str) -> dict | None:
    """The child prints logs, then one JSON object on a line."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _cancelled(ctx) -> bool:
    try:
        ctx.check_cancel()
    except JobCancelled:
        return True
    return False
