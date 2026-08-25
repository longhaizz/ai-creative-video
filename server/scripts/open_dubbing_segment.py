"""Segment a video: Demucs + Silero VAD + Pyannote + Whisper.

Runs in the Open Dubbing venv, not the main one. Prints JSON to stdout:

    {
      "language": "en",
      "language_probability": 0.98,
      "vocals": ".../vocals.wav",
      "no_vocals": ".../no_vocals.wav",
      "utterances": [
        {"start": 1.2, "end": 3.4, "speaker_id": "SPEAKER_00",
         "text": "...", "wav": ".../utt_000.wav"}
      ]
    }

Does not translate and does not speak. The main pipeline does those.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path


SAMPLE_RATE = 16000
MIN_SILENCE_MS = 350
SPEECH_PAD_MS = 150
# VAD only cuts on a long pause. A continuous take of five sentences
# becomes one cue, and TTS cannot fill that slot. Split those.
MAX_UTTERANCE_SECONDS = 6.0
MAX_SENTENCES = 2
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_SENTENCE_END = re.compile(r"[.!?。！？]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--whisper-model", default="medium")
    args = parser.parse_args()

    video = args.video.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not video.is_file():
        print(f"No such video: {video}", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if not token:
        print("HF_TOKEN is required for Pyannote", file=sys.stderr)
        return 2

    try:
        result = segment(video, out, args.whisper_model, token)
    except Exception:
        traceback.print_exc()
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


def segment(video: Path, out: Path, whisper_size: str, token: str) -> dict:
    mix = out / "mix.wav"
    _ffmpeg(["-y", "-loglevel", "error", "-i", str(video),
             "-vn", "-ac", "1", "-c:a", "pcm_s16le", str(mix)])

    vocals, no_vocals = _demucs(mix, out)
    vocals_16k = out / "vocals_16k.wav"
    _ffmpeg(["-y", "-loglevel", "error", "-i", str(vocals),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(vocals_16k)])

    windows = _vad_windows(vocals_16k)
    if not windows:
        raise RuntimeError("No speech was found in the video")

    speakers = _diarize(vocals_16k, token)
    assigned = [_best_speaker(start, end, speakers) for start, end in windows]

    from faster_whisper import WhisperModel

    try:
        model = WhisperModel(whisper_size, device="cuda", compute_type="float16")
    except Exception:
        model = WhisperModel(whisper_size, device="cpu", compute_type="int8")

    language, language_probability = _detect_language(model, vocals_16k)
    utterances = []
    for index, ((start, end), speaker) in enumerate(zip(windows, assigned)):
        wav = out / f"utt_{index:03d}.wav"
        _cut(vocals, start, end, wav)
        kwargs = {"vad_filter": False, "word_timestamps": True}
        if language:
            kwargs["language"] = language
        segments, _info = model.transcribe(str(wav), **kwargs)
        pieces = _pieces_from_segments(list(segments), start, end)
        if not pieces:
            utterances.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker_id": speaker,
                "text": "",
                "wav": str(wav),
            })
            continue
        for piece_start, piece_end, text in pieces:
            utterances.append({
                "start": round(piece_start, 3),
                "end": round(piece_end, 3),
                "speaker_id": _best_speaker(piece_start, piece_end, speakers),
                "text": text,
                "wav": str(wav),
            })

    return {
        "language": language,
        "language_probability": language_probability,
        "vocals": str(vocals),
        "no_vocals": str(no_vocals),
        "utterances": utterances,
    }


def _ffmpeg(args: list[str]) -> None:
    command = ["ffmpeg", *args]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {(result.stderr or '')[-300:]}")


def _cut(src: Path, start: float, end: float, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    length = max(end - start, 0.05)
    _ffmpeg([
        "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-i", str(src), "-c:a", "pcm_s16le", str(dest),
    ])


def _demucs(mix: Path, out: Path) -> tuple[Path, Path]:
    import torch
    from demucs.api import Separator, save_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    separator = Separator(model="htdemucs", device=device)
    _origin, stems = separator.separate_audio_file(str(mix))
    if "vocals" not in stems:
        raise RuntimeError("demucs returned no vocals stem")
    music = sum(source for name, source in stems.items() if name != "vocals")
    vocals = out / "vocals.wav"
    no_vocals = out / "no_vocals.wav"
    save_audio(stems["vocals"], str(vocals), samplerate=separator.samplerate)
    save_audio(music, str(no_vocals), samplerate=separator.samplerate)
    return vocals, no_vocals


def _detect_language(model, wav: Path) -> tuple[str, float]:
    """One language for the whole clip, so cues do not flip mid-video."""
    _segments, info = model.transcribe(str(wav), vad_filter=True)
    for _ in _segments:
        break
    language = getattr(info, "language", "") or ""
    probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    return language, probability


def _sentence_list(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT.split((text or "").strip()) if part.strip()]


def _word_token(item) -> str:
    if isinstance(item, dict):
        return item.get("word") or item.get("text") or ""
    return getattr(item, "word", None) or getattr(item, "text", None) or ""


def _word_times(item) -> tuple[float, float]:
    if isinstance(item, dict):
        return float(item["start"]), float(item["end"])
    return float(item.start), float(item.end)


def _join_tokens(tokens: list[str]) -> str:
    if any(token[:1].isspace() for token in tokens):
        return "".join(tokens).strip()
    return " ".join(token.strip() for token in tokens if token.strip())


def split_long_utterance(
    start: float,
    end: float,
    text: str,
    words: list | None = None,
    word_origin: float | None = None,
) -> list[tuple[float, float, str]]:
    """Turn one VAD window into sentence-sized cues.

    `words` times are relative to `word_origin` (the clip). Default origin
    is `start`, which matches a window that is the whole clip.
    """
    text = (text or "").strip()
    sentences = _sentence_list(text)
    span = end - start
    if len(sentences) <= MAX_SENTENCES and span <= MAX_UTTERANCE_SECONDS:
        return [(start, end, text)]
    if len(sentences) <= 1:
        return [(start, end, text)]

    origin = start if word_origin is None else word_origin
    by_words = _split_on_word_punctuation(start, end, words, origin) if words else None
    if by_words and len(by_words) > 1:
        return by_words
    return _split_proportional(start, end, sentences)


def _split_on_word_punctuation(
    start: float, end: float, words: list, origin: float,
) -> list[tuple[float, float, str]] | None:
    groups: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    for item in words:
        token = _word_token(item)
        if not str(token).strip():
            continue
        rel_start, rel_end = _word_times(item)
        current.append((origin + rel_start, origin + rel_end, token))
        if _SENTENCE_END.search(token):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    if len(groups) <= 1:
        return None
    pieces = []
    for group in groups:
        piece_start = max(start, group[0][0])
        piece_end = min(end, max(group[-1][1], piece_start + 0.15))
        tokens = [item[2] for item in group]
        pieces.append((piece_start, piece_end, _join_tokens(tokens)))
    return pieces


def _split_proportional(
    start: float, end: float, sentences: list[str],
) -> list[tuple[float, float, str]]:
    weights = [max(len(sentence), 1) for sentence in sentences]
    total = sum(weights)
    span = end - start
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


def _pieces_from_segments(segments, origin: float, window_end: float):
    pieces: list[tuple[float, float, str]] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        start = origin + float(segment.start)
        end = origin + float(segment.end)
        words = getattr(segment, "words", None)
        pieces.extend(
            split_long_utterance(start, end, text, words, word_origin=origin)
        )
    out = []
    for start, end, text in pieces:
        start = max(start, origin)
        end = min(end, window_end)
        if text and end - start >= 0.15:
            out.append((start, end, text))
    return out


def _vad_windows(wav: Path) -> list[tuple[float, float]]:
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(str(wav), sampling_rate=SAMPLE_RATE)
    chunks = get_speech_timestamps(
        audio,
        VadOptions(
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        ),
    )
    windows = []
    for chunk in chunks:
        start = chunk["start"] / SAMPLE_RATE
        end = chunk["end"] / SAMPLE_RATE
        if end - start >= 0.15:
            windows.append((start, end))
    return windows


def _diarize(wav: Path, token: str) -> list[dict]:
    """Pyannote turns. Empty list if it cannot run — callers use SPEAKER_00."""
    try:
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        diarization = pipeline(str(wav))
        return _turns_from_diarization(diarization)
    except Exception as error:
        print(f"Pyannote failed, using SPEAKER_00: {error}", file=sys.stderr)
        return []


def _turns_from_diarization(diarization) -> list[dict]:
    """Pyannote 3 returns Annotation; 4 wraps it in DiarizeOutput."""
    annotation = getattr(diarization, "speaker_diarization", diarization)
    turns = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "speaker_id": str(speaker),
        })
    return turns


def _best_speaker(start: float, end: float, turns: list[dict]) -> str:
    if not turns:
        return "SPEAKER_00"
    best_id = None
    best_overlap = 0.0
    for turn in turns:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = turn["speaker_id"]
    if best_id is not None:
        return best_id
    center = (start + end) / 2.0
    nearest = min(
        turns,
        key=lambda turn: abs(center - (turn["start"] + turn["end"]) / 2.0),
    )
    return nearest["speaker_id"]


if __name__ == "__main__":
    raise SystemExit(main())
