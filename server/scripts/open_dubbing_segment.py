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
# Pause between words within a VAD window (e.g. "Go lower." as its own cue).
WORD_GAP_SECONDS = 0.35
# Ignore Pyannote crumbs under this length when cutting a VAD window.
MIN_SPEAKER_TURN_SECONDS = 0.25
# A brief mid-take flip (A-B-A) shorter than this is noise; real asides are longer.
ABSORB_SPEAKER_ISLAND_SECONDS = 0.5
RETRY_WHISPER_MODEL = "large-v3"
# Peak at or above this is already audible. Below it, raise toward TARGET.
QUIET_PEAK_DB = -12.0
TARGET_PEAK_DB = -3.0
MAX_BOOST_DB = 24.0
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
    gain = _boost_if_quiet(mix)
    if gain:
        print(
            f"Source audio was quiet, boosted {gain:.1f} dB so speech can "
            f"be heard",
            file=sys.stderr,
        )

    vocals, no_vocals = _demucs(mix, out)
    vocals_16k = out / "vocals_16k.wav"
    _ffmpeg(["-y", "-loglevel", "error", "-i", str(vocals),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(vocals_16k)])

    vad_windows = _vad_windows(vocals_16k)
    if not vad_windows:
        raise RuntimeError("No speech was found in the video")

    turns = _diarize(vocals_16k, token)
    # One VAD blob can hold two people ("Tolong…? Help me."). Cut on turns.
    windows = _windows_with_speakers(vad_windows, turns)

    from faster_whisper import WhisperModel

    model = _load_whisper(whisper_size)
    language, language_probability = _detect_language(model, vocals_16k)
    model, segments, all_words, language_probability, whisper_used = _transcribe_full(
        model, vocals_16k, language, language_probability, whisper_size,
    )

    per_window = _assign_words_to_windows(
        all_words, [(start, end) for start, end, _speaker in windows],
    )
    utterances = []
    for index, ((start, end, speaker), window_words) in enumerate(
        zip(windows, per_window)
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
                "speaker_id": speaker,
                "text": piece["text"],
                "avg_logprob": piece["avg_logprob"],
                "no_speech_prob": piece["no_speech_prob"],
                "wav": str(wav),
            })

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


def _max_volume_db(path: Path):
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    text = (result.stderr or "") + (result.stdout or "")
    match = re.search(r"max_volume:\s*([-\d.]+)\s*dB", text)
    if not match:
        return None
    return float(match.group(1))


def _boost_if_quiet(path: Path) -> float:
    """Raise a too-quiet mix so VAD, Whisper, and clone can hear it.

    Returns the gain in dB, or 0 if the file was already loud enough.
    """
    peak = _max_volume_db(path)
    if peak is None or peak >= QUIET_PEAK_DB:
        return 0.0
    gain = min(TARGET_PEAK_DB - peak, MAX_BOOST_DB)
    if gain < 1.0:
        return 0.0
    tmp = path.with_name(path.stem + "_boost.wav")
    _ffmpeg([
        "-y", "-loglevel", "error", "-i", str(path),
        "-af", f"volume={gain:.2f}dB", "-c:a", "pcm_s16le", str(tmp),
    ])
    tmp.replace(path)
    return gain


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
    """Each Whisper word belongs to exactly one window (ties → earlier)."""
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


def _windows_with_speakers(
    vad_windows: list[tuple[float, float]], turns: list[dict],
) -> list[tuple[float, float, str]]:
    """Cut each VAD window on Pyannote speaker changes.

    Real dialogue ("Tolong…? Help me.") becomes separate cues with the right
    speaker_id. A short A-B-A island mid-narrator is absorbed so workout ads
    do not flip clone voice every sentence.
    """
    if not vad_windows:
        return []
    if not turns:
        return [(start, end, "SPEAKER_00") for start, end in vad_windows]

    ordered = sorted(turns, key=lambda turn: (turn["start"], turn["end"]))
    out: list[tuple[float, float, str]] = []
    for start, end in vad_windows:
        cuts = {start, end}
        for turn in ordered:
            if start < turn["start"] < end:
                cuts.add(float(turn["start"]))
            if start < turn["end"] < end:
                cuts.add(float(turn["end"]))
        points = sorted(cuts)
        raw: list[list] = []
        for left, right in zip(points, points[1:]):
            if right - left < 1e-6:
                continue
            raw.append([left, right, _best_speaker(left, right, ordered)])
        if not raw:
            out.append((start, end, _best_speaker(start, end, ordered)))
            continue

        spans: list[list] = [raw[0]]
        for seg in raw[1:]:
            if seg[2] == spans[-1][2]:
                spans[-1][1] = seg[1]
            else:
                spans.append(seg)

        folded: list[list] = [spans[0]]
        for seg in spans[1:]:
            if seg[1] - seg[0] < MIN_SPEAKER_TURN_SECONDS:
                folded[-1][1] = seg[1]
            elif folded[-1][1] - folded[-1][0] < MIN_SPEAKER_TURN_SECONDS:
                seg = [folded[-1][0], seg[1], seg[2]]
                folded[-1] = seg
            else:
                folded.append(seg)
        folded[0][0] = start
        folded[-1][1] = end
        for left, right, speaker in _absorb_speaker_islands(folded):
            out.append((left, right, speaker))
    return out


def _absorb_speaker_islands(spans: list[list]) -> list[tuple[float, float, str]]:
    """Fold a short different-speaker island between the same neighbours."""
    if len(spans) < 3:
        return [(float(s[0]), float(s[1]), str(s[2])) for s in spans]
    merged: list[list] = []
    index = 0
    while index < len(spans):
        left, right, speaker = spans[index]
        dur = right - left
        if (
            0 < index < len(spans) - 1
            and dur < ABSORB_SPEAKER_ISLAND_SECONDS
            and spans[index - 1][2] == spans[index + 1][2] != speaker
            and merged
        ):
            merged[-1][1] = right
            index += 1
            continue
        if merged and merged[-1][2] == speaker:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right, speaker])
        index += 1
    # After absorbing an island, the next span may match the previous speaker.
    tight: list[list] = []
    for seg in merged:
        if tight and tight[-1][2] == seg[2]:
            tight[-1][1] = seg[1]
        else:
            tight.append(seg)
    return [(float(s[0]), float(s[1]), str(s[2])) for s in tight]


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
