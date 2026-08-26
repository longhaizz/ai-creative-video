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
# A leftover shorter than this after a 6s split is not its own cue.
# zh→en cannot fit a real sentence in 0.5–1.6s; attach it to the neighbour.
MIN_UTTERANCE_SECONDS = 2.0
MAX_SENTENCES = 2
# Pause between words within a VAD window (e.g. "Go lower." as its own cue).
WORD_GAP_SECONDS = 0.35
# Only fold a short piece into its neighbour when they nearly touch.
MERGE_GAP_SECONDS = WORD_GAP_SECONDS
RETRY_WHISPER_MODEL = "large-v3"
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

    model = _load_whisper(whisper_size)
    language, language_probability = _detect_language(model, vocals_16k)
    model, segments, all_words, language_probability, whisper_used = _transcribe_full(
        model, vocals_16k, language, language_probability, whisper_size,
    )

    per_window = _assign_words_to_windows(all_words, windows)
    utterances = []
    for index, ((start, end), speaker, window_words) in enumerate(
        zip(windows, assigned, per_window)
    ):
        wav = out / f"utt_{index:03d}.wav"
        _cut(vocals, start, end, wav)
        pieces = _pieces_from_words(window_words, start, end)
        if not pieces:
            utterances.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "speech_start": round(start, 3),
                "speech_end": round(end, 3),
                "speaker_id": speaker,
                "text": "",
                "avg_logprob": 0.0,
                "no_speech_prob": 1.0,
                "wav": str(wav),
            })
            continue
        for piece in pieces:
            utterances.append({
                "start": round(piece["start"], 3),
                "end": round(piece["end"], 3),
                "speech_start": round(piece["speech_start"], 3),
                "speech_end": round(piece["speech_end"], 3),
                # One VAD window is one take. Sentence splits must not
                # pick up Pyannote flipping SPEAKER_00/01 mid-ad.
                "speaker_id": speaker,
                "text": piece["text"],
                "avg_logprob": piece["avg_logprob"],
                "no_speech_prob": piece["no_speech_prob"],
                "wav": str(wav),
            })

    utterances = _merge_short_utterances(utterances)
    return {
        "language": language,
        "language_probability": language_probability,
        "whisper_model": whisper_used,
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


def _load_whisper(size: str):
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(size, device="cuda", compute_type="float16")
    except Exception:
        return WhisperModel(size, device="cpu", compute_type="int8")


def _transcribe_kwargs(language: str) -> dict:
    kwargs = {
        "vad_filter": False,
        "word_timestamps": True,
        "beam_size": 5,
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    prompt = (os.environ.get("WHISPER_INITIAL_PROMPT") or "").strip()
    if prompt:
        kwargs["initial_prompt"] = prompt
    return kwargs


def _detect_language(model, wav: Path) -> tuple[str, float]:
    """One language for the whole clip, so cues do not flip mid-video."""
    _segments, info = model.transcribe(str(wav), vad_filter=True)
    for _ in _segments:
        break
    language = getattr(info, "language", "") or ""
    probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    return language, probability


    return language, probability


def _run_transcribe(model, wav: Path, language: str):
    segments, info = model.transcribe(str(wav), **_transcribe_kwargs(language))
    return list(segments), info


def _flatten_words(segments) -> list[dict]:
    words = []
    for segment in segments:
        avg_logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
        no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        for word in segment.words or []:
            token = (word.word or "").strip()
            if not token:
                continue
            words.append({
                "word": word.word or "",
                "start": float(word.start),
                "end": float(word.end),
                "avg_logprob": avg_logprob,
                "no_speech_prob": no_speech_prob,
            })
    return words


def _segments_to_cues(segments) -> list[dict]:
    cues = []
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        cues.append({
            "text": text,
            "avg_logprob": float(getattr(segment, "avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0) or 0.0),
        })
    return cues


def _asr_quality(cues: list[dict], language_probability: float = 0.0) -> dict:
    """Same heuristics as server/steps/transcribe.py (this script runs in venv-od)."""
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
        if float(cue.get("no_speech_prob") or 0) > 0.6:
            low_confidence += 1

    if garbled / count >= 0.35:
        reasons.append(f"garbled_cues={garbled}/{count}")
    if low_confidence / count >= 0.35:
        reasons.append(f"low_confidence_cues={low_confidence}/{count}")

    return {"ok": not reasons, "reasons": reasons}


def _transcribe_full(model, wav: Path, language: str, language_probability: float,
                     whisper_size: str):
    segments, info = _run_transcribe(model, wav, language)
    language_probability = float(
        getattr(info, "language_probability", language_probability) or language_probability
    )
    quality = _asr_quality(_segments_to_cues(segments), language_probability)
    used = whisper_size

    if not quality["ok"] and whisper_size != RETRY_WHISPER_MODEL:
        print(
            f"Transcript looks poor ({', '.join(quality['reasons'])}); "
            f"retrying with {RETRY_WHISPER_MODEL}",
            file=sys.stderr,
        )
        model = _load_whisper(RETRY_WHISPER_MODEL)
        segments, info = _run_transcribe(model, wav, language)
        language_probability = float(
            getattr(info, "language_probability", language_probability)
            or language_probability
        )
        used = RETRY_WHISPER_MODEL

    return model, segments, _flatten_words(segments), language_probability, used


def _assign_words_to_windows(
    words: list[dict], windows: list[tuple[float, float]],
) -> list[list[dict]]:
    """Each Whisper word belongs to exactly one VAD window.

    Overlap on the boundary used to copy the last character into the next
    cue ("兰花" / "花很多人"). Ties go to the earlier window.
    """
    buckets: list[list[dict]] = [[] for _ in windows]
    if not windows:
        return buckets
    for word in words:
        if not (word.get("word") or "").strip():
            continue
        start = float(word["start"])
        end = float(word["end"])
        best_i = 0
        best_ov = -1.0
        for index, (left, right) in enumerate(windows):
            overlap = max(0.0, min(end, right) - max(start, left))
            if overlap > best_ov:
                best_ov = overlap
                best_i = index
        if best_ov <= 0:
            mid = (start + end) / 2.0
            best_i = min(
                range(len(windows)),
                key=lambda i: min(
                    abs(mid - windows[i][0]), abs(mid - windows[i][1]),
                ),
            )
        buckets[best_i].append(word)
    return buckets


def _merge_short_utterances(utterances: list[dict]) -> list[dict]:
    """Fold a cue shorter than MIN into its neighbour when they nearly touch."""
    if len(utterances) < 2:
        return utterances
    out = [dict(utterances[0])]
    for item in utterances[1:]:
        prev = out[-1]
        same = (prev.get("speaker_id") or "SPEAKER_00") == (
            item.get("speaker_id") or "SPEAKER_00"
        )
        gap = float(item["start"]) - float(prev["end"])
        prev_dur = float(prev["end"]) - float(prev["start"])
        dur = float(item["end"]) - float(item["start"])
        if same and gap <= MERGE_GAP_SECONDS and (
            dur < MIN_UTTERANCE_SECONDS or prev_dur < MIN_UTTERANCE_SECONDS
        ):
            prev["end"] = item["end"]
            prev["speech_end"] = item.get("speech_end", item["end"])
            left = (prev.get("text") or "").strip()
            right = (item.get("text") or "").strip()
            prev["text"] = f"{left} {right}".strip()
            prev["no_speech_prob"] = max(
                float(prev.get("no_speech_prob") or 0),
                float(item.get("no_speech_prob") or 0),
            )
        else:
            out.append(dict(item))
    return out


def _pieces_from_words(
    words: list[dict], window_start: float, window_end: float,
) -> list[dict]:
    pieces = []
    for group in _word_groups(words, window_start, window_end):
        text = _join_tokens([item["word"] for item in group])
        if not text:
            continue
        speech_start = max(window_start, group[0]["start"])
        speech_end = min(window_end, group[-1]["end"])
        if speech_end <= speech_start:
            speech_end = speech_start + 0.15
        pieces.append({
            "start": speech_start,
            "end": speech_end,
            "speech_start": speech_start,
            "speech_end": speech_end,
            "text": text,
            "avg_logprob": sum(item["avg_logprob"] for item in group) / len(group),
            "no_speech_prob": max(item["no_speech_prob"] for item in group),
        })
    return pieces


def _word_groups(
    words: list[dict], window_start: float, window_end: float,
) -> list[list[dict]]:
    cleaned = [
        word for word in words
        if (word.get("word") or "").strip()
        and word["end"] > window_start and word["start"] < window_end
    ]
    if not cleaned:
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

    for index, word in enumerate(cleaned):
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
        if sentence_count(current) >= MAX_SENTENCES and index + 1 < len(cleaned):
            flush()
    flush()

    out: list[list[dict]] = []
    for group in groups:
        out.extend(_split_oversized_word_group(group, window_start, window_end))
    return _merge_short_word_groups(out)


def _group_span(group: list[dict]) -> float:
    return float(group[-1]["end"]) - float(group[0]["start"])


def _merge_short_word_groups(groups: list[list[dict]]) -> list[list[dict]]:
    """A 6s hard split must not leave a 0.5s orphan as its own cue."""
    if len(groups) < 2:
        return groups
    out = [list(groups[0])]
    for group in groups[1:]:
        prev = out[-1]
        gap = float(group[0]["start"]) - float(prev[-1]["end"])
        if gap <= MERGE_GAP_SECONDS and (
            _group_span(group) < MIN_UTTERANCE_SECONDS
            or _group_span(prev) < MIN_UTTERANCE_SECONDS
        ):
            out[-1] = prev + list(group)
        else:
            out.append(list(group))
    return out


def _split_oversized_word_group(
    group: list[dict], window_start: float, window_end: float,
) -> list[list[dict]]:
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
        left = _split_oversized_word_group(group[:best_index], window_start, window_end)
        right = _split_oversized_word_group(group[best_index:], window_start, window_end)
        return left + right

    mid = max(1, len(group) // 2)
    left = _split_oversized_word_group(group[:mid], window_start, window_end)
    right = _split_oversized_word_group(group[mid:], window_start, window_end)
    return left + right


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
    by_gaps = _split_on_word_gaps(start, end, words, origin) if words else None
    if by_gaps and len(by_gaps) > 1:
        return by_gaps
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


def _split_on_word_gaps(
    start: float, end: float, words: list, origin: float,
    min_gap: float = WORD_GAP_SECONDS,
) -> list[tuple[float, float, str]] | None:
    groups: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str]] = []
    previous_end = None
    for item in words:
        token = _word_token(item)
        if not str(token).strip():
            continue
        rel_start, rel_end = _word_times(item)
        abs_start = origin + rel_start
        abs_end = origin + rel_end
        if previous_end is not None and abs_start - previous_end >= min_gap and current:
            groups.append(current)
            current = []
        current.append((abs_start, abs_end, token))
        previous_end = abs_end
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
