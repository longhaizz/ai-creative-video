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
import subprocess
import time
from pathlib import Path

from server import config
from server.jobs import JobCancelled, PipelineError
from server.steps import audio

LOG_EVERY_SECONDS = 2.0
TAIL_LINES = 25
MIN_REF_SECONDS = 3.0

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
    for index, item in enumerate(utterances):
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            end = start + 0.4
        ref = pad_reference(vocals, start, end, out_dir / f"ref_{index:03d}.wav")
        cues.append({
            "start": start,
            "end": end,
            "speech_start": start,
            "speech_end": end,
            "text": (item.get("text") or "").strip(),
            "speaker_id": item.get("speaker_id") or "SPEAKER_00",
            "ref_wav": str(ref),
            "avg_logprob": 0.0,
            "no_speech_prob": 0.0 if (item.get("text") or "").strip() else 1.0,
        })

    meta = {
        "language": payload.get("language") or "",
        "language_probability": float(payload.get("language_probability") or 0.0),
    }
    return {"cues": cues, "meta": meta, "vocals": vocals, "music": music}


def pad_reference(vocals: Path, start: float, end: float, dest: Path) -> Path:
    """Widen a short clone clip to MIN_REF_SECONDS, clamped to the file."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = audio.duration(vocals)
    span = max(end - start, 0.05)
    if span >= MIN_REF_SECONDS or total <= MIN_REF_SECONDS:
        left, right = max(start, 0.0), min(end, total)
    else:
        extra = MIN_REF_SECONDS - span
        left = max(0.0, start - extra / 2)
        right = min(total, left + MIN_REF_SECONDS)
        left = max(0.0, right - MIN_REF_SECONDS)
    length = max(right - left, 0.05)
    audio.run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-ss", f"{left:.3f}", "-t", f"{length:.3f}",
        "-i", str(vocals), "-c:a", "pcm_s16le", str(dest),
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
