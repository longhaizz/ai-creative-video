"""Open Dubbing segmentation, checked without Pyannote and without a GPU.

The child venv is a black box. These tests cover the command line, the JSON
the child must print, and that original-voice clone uses each cue's ref wav.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from server.jobs import PipelineError
from server.steps.open_dubbing import (
    build_command,
    cues_from_payload,
    pad_reference,
    segment,
)


def test_command_points_at_the_od_python_and_script(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.OPEN_DUBBING_PYTHON", "/opt/venv-od/bin/python")
    command = build_command(tmp_path / "in.mp4", tmp_path / "od", "medium")
    assert command[0] == "/opt/venv-od/bin/python"
    assert command[1].endswith("open_dubbing_segment.py")
    assert command[2].endswith("in.mp4")
    assert command[command.index("--whisper-model") + 1] == "medium"


def test_missing_hf_token_is_invalid_input(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.HF_TOKEN", "")
    monkeypatch.setattr("server.config.OPEN_DUBBING_PYTHON", str(tmp_path / "python"))
    (tmp_path / "python").write_text("", encoding="utf-8")
    with pytest.raises(PipelineError) as error:
        segment(tmp_path / "video.mp4", tmp_path, "medium")
    assert error.value.code == "invalid_input"


def test_cues_from_payload_keep_timing_and_pad_refs(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")

    written: list[tuple[float, float, Path]] = []

    def fake_pad(src, start, end, dest):
        written.append((start, end, dest))
        Path(dest).write_bytes(b"r")
        return Path(dest)

    monkeypatch.setattr("server.steps.open_dubbing.pad_reference", fake_pad)

    payload = {
        "language": "vi",
        "language_probability": 0.9,
        "vocals": str(vocals),
        "no_vocals": str(music),
        "utterances": [
            {"start": 1.0, "end": 2.0, "speaker_id": "SPEAKER_00", "text": "xin chao"},
            {"start": 4.0, "end": 5.5, "speaker_id": "SPEAKER_01", "text": "tam biet"},
        ],
    }
    result = cues_from_payload(payload, tmp_path)
    assert result["vocals"] == vocals
    assert result["music"] == music
    assert result["meta"]["language"] == "vi"
    assert len(result["cues"]) == 2
    assert result["cues"][0]["speech_start"] == 1.0
    assert result["cues"][0]["text"] == "xin chao"
    assert result["cues"][1]["speaker_id"] == "SPEAKER_01"
    assert written[0][:2] == (1.0, 2.0)
    assert written[1][:2] == (4.0, 5.5)
    assert Path(result["cues"][0]["ref_wav"]).name == "ref_SPEAKER_00.wav"
    assert result["cues"][0]["ref_wav"] != result["cues"][1]["ref_wav"]


def test_empty_utterances_are_invalid_input(tmp_path):
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    with pytest.raises(PipelineError) as error:
        cues_from_payload(
            {"vocals": str(vocals), "no_vocals": str(music), "utterances": []},
            tmp_path,
        )
    assert error.value.code == "invalid_input"


def test_original_speak_uses_the_cue_ref(tmp_path):
    """The clone path must not reuse one vocals file for every line."""
    refs = []

    def speak(text, out_wav, cue=None):
        refs.append(cue["ref_wav"] if cue else None)
        Path(out_wav).write_bytes(b"x")
        return Path(out_wav)

    cues = [
        {"start": 0, "end": 1, "speech_start": 0, "speech_end": 1,
         "text": "a", "ref_wav": str(tmp_path / "r0.wav")},
        {"start": 2, "end": 3, "speech_start": 2, "speech_end": 3,
         "text": "b", "ref_wav": str(tmp_path / "r1.wav")},
    ]
    for cue in cues:
        speak(cue["text"], tmp_path / f"{cue['text']}.wav", cue)
    assert refs == [str(tmp_path / "r0.wav"), str(tmp_path / "r1.wav")]


def test_same_speaker_cues_share_one_ref(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    written = []

    def fake_pad(src, start, end, dest):
        written.append((start, end, dest))
        Path(dest).write_bytes(b"r")
        return Path(dest)

    monkeypatch.setattr("server.steps.open_dubbing.pad_reference", fake_pad)
    result = cues_from_payload(
        {
            "vocals": str(vocals),
            "no_vocals": str(music),
            "utterances": [
                {"start": 0.0, "end": 4.0, "speaker_id": "SPEAKER_00", "text": "a"},
                {"start": 5.0, "end": 6.0, "speaker_id": "SPEAKER_00", "text": "b"},
            ],
        },
        tmp_path,
    )
    assert result["cues"][0]["ref_wav"] == result["cues"][1]["ref_wav"]
    assert written == [(0.0, 4.0, tmp_path / "ref_SPEAKER_00.wav")]


def test_short_same_speaker_lines_are_concatenated(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    commands = []
    monkeypatch.setattr("server.steps.audio.duration", lambda path: 10.0)

    def fake_ffmpeg(command):
        commands.append(command)
        Path(command[-1]).write_bytes(b"x")
        return ""

    monkeypatch.setattr("server.steps.audio.run_ffmpeg", fake_ffmpeg)
    result = cues_from_payload(
        {
            "vocals": str(vocals),
            "no_vocals": str(music),
            "utterances": [
                {"start": 5.12, "end": 5.74, "speaker_id": "SPEAKER_01",
                 "text": "Get down lower."},
                {"start": 6.41, "end": 7.39, "speaker_id": "SPEAKER_01",
                 "text": "Yes, that's better."},
            ],
        },
        tmp_path,
    )
    assert result["cues"][0]["ref_wav"] == result["cues"][1]["ref_wav"]
    assert any("concat" in command for command in commands)


def test_pad_reference_cuts_the_span_and_does_not_widen(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"v")
    dest = tmp_path / "ref.wav"
    commands = []

    monkeypatch.setattr("server.steps.audio.duration", lambda path: 10.0)

    def fake_ffmpeg(command):
        commands.append(command)
        Path(command[-1]).write_bytes(b"r")
        return ""

    monkeypatch.setattr("server.steps.audio.run_ffmpeg", fake_ffmpeg)
    pad_reference(vocals, 1.0, 1.5, dest)
    assert dest.is_file()
    ss = commands[0][commands[0].index("-ss") + 1]
    length = float(commands[0][commands[0].index("-t") + 1])
    assert abs(float(ss) - 1.0) < 0.02
    assert abs(length - 0.5) < 0.02


def test_missing_od_python_is_invalid_input(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.HF_TOKEN", "hf_x")
    monkeypatch.setattr(
        "server.config.OPEN_DUBBING_PYTHON", str(tmp_path / "no-such-python")
    )
    with pytest.raises(PipelineError) as error:
        segment(tmp_path / "video.mp4", tmp_path, "medium")
    assert error.value.code == "invalid_input"


def test_missing_speaker_id_becomes_speaker_00(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    monkeypatch.setattr(
        "server.steps.open_dubbing.pad_reference",
        lambda src, start, end, dest: Path(dest),
    )
    result = cues_from_payload(
        {
            "vocals": str(vocals),
            "no_vocals": str(music),
            "utterances": [{"start": 0.0, "end": 1.0, "text": "hi"}],
        },
        tmp_path,
    )
    assert result["cues"][0]["speaker_id"] == "SPEAKER_00"


def test_turns_from_pyannote3_annotation():
    mod = _od_script()

    class Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class Annotation:
        def itertracks(self, yield_label=True):
            yield Segment(0.0, 1.2), None, "SPEAKER_01"

    assert mod._turns_from_diarization(Annotation()) == [
        {"start": 0.0, "end": 1.2, "speaker_id": "SPEAKER_01"},
    ]


def test_turns_from_pyannote4_diarize_output():
    mod = _od_script()

    class Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class Annotation:
        def itertracks(self, yield_label=True):
            yield Segment(0.4, 2.0), None, "SPEAKER_02"

    class DiarizeOutput:
        speaker_diarization = Annotation()

    assert mod._turns_from_diarization(DiarizeOutput()) == [
        {"start": 0.4, "end": 2.0, "speaker_id": "SPEAKER_02"},
    ]


def test_best_speaker_uses_overlap_then_nearest():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "open_dubbing_segment.py"
    spec = importlib.util.spec_from_file_location("od_seg", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    turns = [
        {"start": 0.0, "end": 1.0, "speaker_id": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "speaker_id": "SPEAKER_01"},
    ]
    assert mod._best_speaker(0.2, 0.8, turns) == "SPEAKER_00"
    assert mod._best_speaker(2.1, 2.9, turns) == "SPEAKER_01"
    assert mod._best_speaker(1.8, 1.9, turns) == "SPEAKER_01"
    assert mod._best_speaker(0.0, 1.0, []) == "SPEAKER_00"
    # A 5.7s VAD window is still the narrator; the 0.6s aside is not.
    mixed = [
        {"start": 0.0, "end": 5.0, "speaker_id": "SPEAKER_00"},
        {"start": 5.12, "end": 5.74, "speaker_id": "SPEAKER_01"},
    ]
    assert mod._best_speaker(0.0, 5.74, mixed) == "SPEAKER_00"
    assert mod._best_speaker(5.12, 5.74, mixed) == "SPEAKER_01"


def test_dialogue_vad_window_splits_by_speaker():
    """Indonesian app ad: host and English prompt must not share one cue."""
    mod = _od_script()
    vad = [(0.0, 4.15)]
    turns = [
        {"start": 0.0, "end": 1.4, "speaker_id": "SPEAKER_02"},
        {"start": 1.4, "end": 2.0, "speaker_id": "SPEAKER_01"},
        {"start": 2.0, "end": 3.1, "speaker_id": "SPEAKER_02"},
        {"start": 3.1, "end": 4.15, "speaker_id": "SPEAKER_01"},
    ]
    windows = mod._windows_with_speakers(vad, turns)
    assert [(round(s, 2), round(e, 2), sp) for s, e, sp in windows] == [
        (0.0, 1.4, "SPEAKER_02"),
        (1.4, 2.0, "SPEAKER_01"),
        (2.0, 3.1, "SPEAKER_02"),
        (3.1, 4.15, "SPEAKER_01"),
    ]


def test_short_speaker_island_is_absorbed_into_narrator():
    """Pyannote flipping 00 mid-workout must not become its own cue."""
    mod = _od_script()
    vad = [(0.0, 23.7)]
    workout = [
        {"start": 0.0, "end": 3.06, "speaker_id": "SPEAKER_01"},
        {"start": 3.06, "end": 3.4, "speaker_id": "SPEAKER_00"},
        {"start": 3.4, "end": 23.7, "speaker_id": "SPEAKER_01"},
    ]
    windows = mod._windows_with_speakers(vad, workout)
    assert len(windows) == 1
    assert windows[0][2] == "SPEAKER_01"
    assert windows[0][0] == 0.0
    assert windows[0][1] == 23.7


def test_aside_after_narrator_stays_its_own_speaker():
    """A real 0.6s coach line after the take stays SPEAKER_01."""
    mod = _od_script()
    vad = [(0.0, 5.74)]
    turns = [
        {"start": 0.0, "end": 5.0, "speaker_id": "SPEAKER_00"},
        {"start": 5.12, "end": 5.74, "speaker_id": "SPEAKER_01"},
    ]
    windows = mod._windows_with_speakers(vad, turns)
    assert len(windows) == 2
    assert windows[0][2] == "SPEAKER_00"
    assert windows[1][2] == "SPEAKER_01"


def test_a_word_on_the_speaker_boundary_is_not_copied():
    mod = _od_script()
    windows = [(0.0, 1.4), (1.4, 2.0)]
    words = [
        {"word": "apa?", "start": 1.2, "end": 1.35,
         "avg_logprob": -0.2, "no_speech_prob": 0.05},
        {"word": "Help", "start": 1.38, "end": 1.55,
         "avg_logprob": -0.2, "no_speech_prob": 0.05},
        {"word": "me.", "start": 1.55, "end": 1.8,
         "avg_logprob": -0.2, "no_speech_prob": 0.05},
    ]
    buckets = mod._assign_words_to_windows(words, windows)
    assert [w["word"] for w in buckets[0]] == ["apa?"]
    assert [w["word"] for w in buckets[1]] == ["Help", "me."]


def test_language_is_detected_once_and_locked_on_every_cue():
    """Cues must not pick their own language. One detect, then lock."""
    import importlib.util
    import inspect

    path = Path(__file__).resolve().parents[1] / "scripts" / "open_dubbing_segment.py"
    spec = importlib.util.spec_from_file_location("od_seg_lang", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Info:
        language = "vi"
        language_probability = 0.97

    class Segment:
        def __init__(self, text):
            self.text = text

    calls = []

    class FakeModel:
        def transcribe(self, wav, **kwargs):
            calls.append(kwargs)
            return iter([Segment("xin chao")]), Info()

    language, probability = mod._detect_language(FakeModel(), Path("vocals.wav"))
    assert language == "vi"
    assert probability == 0.97
    assert calls[0]["vad_filter"] is True
    assert "language" not in calls[0]

    src = inspect.getsource(mod.segment)
    assert "_transcribe_full" in src
    assert "_detect_language" in src
    assert "mix_16k" in src
    assert "_orphan_word_windows" in src


def test_merge_intervals_unions_mix_and_vocals_vad():
    mod = _od_script()
    vocals = [(0.0, 8.2), (30.0, 37.1)]
    mix = [(0.0, 8.0), (12.4, 16.2), (21.0, 24.5), (29.8, 37.0)]
    merged = mod._merge_intervals([vocals, mix])
    assert merged[0] == (0.0, 8.2)
    assert (12.4, 16.2) in merged
    assert (21.0, 24.5) in merged
    assert merged[-1][0] <= 30.0
    assert merged[-1][1] >= 37.0


def test_uncovered_intervals_are_the_mix_only_middle():
    mod = _od_script()
    vocals = [(0.0, 8.2), (30.0, 37.1)]
    mix = [(0.0, 8.0), (12.4, 16.2), (21.0, 24.5), (29.8, 37.0)]
    missed = mod._uncovered_intervals(mix, vocals)
    assert missed == [(12.4, 16.2), (21.0, 24.5)]


def test_orphan_words_in_the_vad_hole_become_a_window():
    """Hindi ad: VAD heard 0–8s and 30s+, Whisper still heard the demo VO."""
    mod = _od_script()
    windows = [(0.0, 8.2), (30.0, 37.1)]
    words = [
        {"word": "Hello", "start": 0.2, "end": 0.6, "no_speech_prob": 0.05},
        {"word": "Use", "start": 12.5, "end": 12.8, "no_speech_prob": 0.08},
        {"word": " this", "start": 12.8, "end": 13.1, "no_speech_prob": 0.08},
        {"word": " app", "start": 13.1, "end": 13.5, "no_speech_prob": 0.08},
        {"word": " now.", "start": 13.5, "end": 13.9, "no_speech_prob": 0.08},
        {"word": "Download", "start": 30.2, "end": 30.7, "no_speech_prob": 0.04},
    ]
    extra = mod._orphan_word_windows(words, windows)
    assert len(extra) == 1
    assert extra[0][0] == 12.5
    assert extra[0][1] == 13.9


def test_orphan_windows_skip_b_roll_stamps_and_no_speech():
    mod = _od_script()
    windows = [(0.0, 5.0)]
    words = [
        {"word": "रखूं।", "start": 14.6, "end": 37.3, "no_speech_prob": 0.003},
        {"word": "hi", "start": 20.0, "end": 20.2, "no_speech_prob": 0.97},
        {"word": "there", "start": 20.2, "end": 20.5, "no_speech_prob": 0.97},
    ]
    assert mod._orphan_word_windows(words, windows) == []


def test_transcribe_kwargs_lock_language_and_beam():
    mod = _od_script()
    kwargs = mod._transcribe_kwargs("en")
    assert kwargs["language"] == "en"
    assert kwargs["beam_size"] == 5
    assert kwargs["word_timestamps"] is True


def _od_script():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "open_dubbing_segment.py"
    spec = importlib.util.spec_from_file_location("od_seg_split", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_short_line_is_not_split():
    mod = _od_script()
    assert mod.split_long_utterance(1.0, 3.0, "Hello there.") == [
        (1.0, 3.0, "Hello there."),
    ]


def test_two_short_sentences_stay_one_cue():
    """A 4s window with two sentences is still one breath."""
    mod = _od_script()
    text = "Hello there. How are you?"
    assert mod.split_long_utterance(0.0, 4.0, text) == [(0.0, 4.0, text)]


def test_a_long_window_of_sentences_is_split():
    """The 15s VAD blob from a continuous take must not stay one cue."""
    mod = _od_script()
    text = (
        "At the same time, it shows you how to perform each exercise. "
        "You won't have to worry about what to do. "
        "And even if you don't workout at the gym, this is a complete home workout plan. "
        "All you have to do is apply it and see how it goes. "
        "The app link is in the description."
    )
    pieces = mod.split_long_utterance(9.29, 24.96, text)
    assert len(pieces) == 5
    assert pieces[0][0] == 9.29
    assert pieces[-1][1] == 24.96
    assert all(piece[2].endswith((".", "?")) for piece in pieces)
    # Each cue is a sentence, not the whole paragraph.
    assert all(len(mod._sentence_list(piece[2])) == 1 for piece in pieces)


def test_word_gap_splits_short_asides():
    """A pause before 'Go lower.' should become its own cue."""
    mod = _od_script()
    words = [
        {"word": "cheers", "start": 13.5, "end": 13.9, "avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"word": " you", "start": 13.9, "end": 14.1, "avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"word": " up.", "start": 14.1, "end": 14.4, "avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"word": " Go", "start": 17.0, "end": 17.2, "avg_logprob": -0.2, "no_speech_prob": 0.01},
        {"word": " lower.", "start": 17.2, "end": 17.5, "avg_logprob": -0.2, "no_speech_prob": 0.01},
    ]
    groups = mod._word_groups(words, 13.0, 18.0)
    assert len(groups) == 2
    assert "lower" in groups[1][-1]["word"]


def test_pieces_from_words_carry_speech_bounds():
    mod = _od_script()
    words = [
        {"word": "Hello", "start": 1.0, "end": 1.3, "avg_logprob": -0.3, "no_speech_prob": 0.02},
        {"word": " there.", "start": 1.3, "end": 1.7, "avg_logprob": -0.3, "no_speech_prob": 0.02},
    ]
    pieces = mod._pieces_from_words(words, 0.5, 2.0)
    assert len(pieces) == 1
    assert pieces[0]["speech_start"] == 1.0
    assert pieces[0]["speech_end"] == 1.7
    assert pieces[0]["text"] == "Hello there."


def test_a_one_word_vad_blob_is_clamped_to_speech():
    """Whisper stamped 'रखूं।' across 22s of B-roll. Slot the word, not the blob."""
    mod = _od_script()
    words = [{
        "word": "रखूं।",
        "start": 14.602,
        "end": 37.366,
        "avg_logprob": -0.04,
        "no_speech_prob": 0.003,
    }]
    pieces = mod._pieces_from_words(words, 14.602, 37.366)
    assert len(pieces) == 1
    assert pieces[0]["speech_start"] == 14.602
    assert pieces[0]["speech_end"] <= 14.602 + 1.5
    assert pieces[0]["speech_end"] - pieces[0]["speech_start"] <= 1.5
    assert "रखूं" in pieces[0]["text"]


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not on PATH",
)


def _sine(path: Path, db: str, seconds: float = 0.4):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-af", f"volume={db}", "-c:a", "pcm_s16le", str(path),
        ],
        check=True, capture_output=True,
    )


@needs_ffmpeg
def test_quiet_mix_is_boosted_so_speech_can_be_heard(tmp_path):
    mod = _od_script()
    quiet = tmp_path / "quiet.wav"
    _sine(quiet, "-12dB")
    before = mod._max_volume_db(quiet)
    assert before is not None and before < mod.QUIET_PEAK_DB
    gain = mod._boost_if_quiet(quiet)
    assert gain >= 1.0
    after = mod._max_volume_db(quiet)
    assert after is not None and after > before + 5


@needs_ffmpeg
def test_already_loud_mix_is_left_alone(tmp_path):
    mod = _od_script()
    loud = tmp_path / "loud.wav"
    _sine(loud, "16dB")
    gain = mod._boost_if_quiet(loud)
    assert gain == 0.0


def test_cues_from_payload_use_speech_bounds(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    monkeypatch.setattr(
        "server.steps.open_dubbing.pad_reference",
        lambda src, start, end, dest: Path(dest),
    )
    result = cues_from_payload(
        {
            "vocals": str(vocals),
            "no_vocals": str(music),
            "utterances": [{
                "start": 0.0,
                "end": 3.0,
                "speech_start": 0.2,
                "speech_end": 2.8,
                "speaker_id": "SPEAKER_00",
                "text": "hi",
                "avg_logprob": -0.4,
                "no_speech_prob": 0.05,
            }],
        },
        tmp_path,
    )
    cue = result["cues"][0]
    assert cue["speech_start"] == 0.2
    assert cue["speech_end"] == 2.8
    assert cue["avg_logprob"] == -0.4
    assert cue["no_speech_prob"] == 0.05


def _cue(**kwargs):
    item = {
        "start": 0.0,
        "end": 1.0,
        "speech_start": 0.0,
        "speech_end": 1.0,
        "speaker_id": "SPEAKER_00",
        "text": "hi",
        "avg_logprob": -0.2,
        "no_speech_prob": 0.05,
    }
    item.update(kwargs)
    return item


def test_prefix_crumbs_are_glued_into_the_next_cue():
    from server.steps.open_dubbing import glue_prefix_crumbs

    cues = [
        _cue(start=28.04, end=28.82, speech_start=28.04, speech_end=28.82,
             text="I gotta go."),
        _cue(start=29.14, end=29.8, speech_start=29.14, speech_end=29.8,
             text="Ikutin aku."),
        _cue(start=30.0, end=30.166, speech_start=30.0, speech_end=30.166,
             text="I"),
        _cue(start=30.346, end=31.02, speech_start=30.346, speech_end=31.02,
             text="I gotta go."),
        _cue(start=36.16, end=36.214, speech_start=36.16, speech_end=36.214,
             text="You're"),
        _cue(start=36.362, end=37.36, speech_start=36.362, speech_end=37.36,
             text="You're very kind."),
        _cue(start=64.33, end=65.42, speech_start=64.33, speech_end=65.42,
             text="Today is very cold."),
        _cue(start=65.42, end=65.718, speech_start=65.42, speech_end=65.718,
             text="Ikitin", no_speech_prob=0.998),
    ]
    out = glue_prefix_crumbs(cues)
    texts = [c["text"] for c in out]
    assert texts == [
        "I gotta go.",
        "Ikutin aku.",
        "I gotta go.",
        "You're very kind.",
        "Today is very cold.",
    ]
    glued = next(c for c in out if c["text"] == "You're very kind.")
    assert glued["speech_start"] == 36.16
    assert glued["speech_end"] == 37.36


def test_tiny_tail_word_is_dropped():
    """Arabic leftover 'Water' 0.28s must not be dubbed."""
    from server.steps.open_dubbing import glue_prefix_crumbs

    cues = [
        _cue(start=4.7, end=10.7, speech_start=4.7, speech_end=10.7,
             text="this traffic is for you"),
        _cue(start=10.7, end=10.98, speech_start=10.7, speech_end=10.98,
             text="الماء", no_speech_prob=0.13),
    ]
    out = glue_prefix_crumbs(cues)
    assert [c["text"] for c in out] == ["this traffic is for you"]


def test_one_word_no_speech_blob_is_dropped():
    """Hindi 'पूरी' sitting on a 13s empty VAD window."""
    from server.steps.open_dubbing import glue_prefix_crumbs

    cues = [
        _cue(start=14.86, end=19.44, speech_start=14.86, speech_end=19.44,
             text="You will instantly process your fingerprint for results"),
        _cue(start=19.56, end=33.08, speech_start=19.56, speech_end=33.08,
             text="पूरी", no_speech_prob=0.97),
        _cue(start=34.08, end=37.49, speech_start=34.08, speech_end=37.49,
             text="download and try it free"),
    ]
    out = glue_prefix_crumbs(cues)
    assert [c["text"] for c in out] == [
        "You will instantly process your fingerprint for results",
        "download and try it free",
    ]


def test_short_spoken_aside_is_kept():
    from server.steps.open_dubbing import glue_prefix_crumbs

    cues = [
        _cue(start=1.44, end=2.08, speech_start=1.44, speech_end=2.08,
             text="Help me."),
    ]
    assert glue_prefix_crumbs(cues)[0]["text"] == "Help me."


def test_short_cta_merges_into_the_next_cue():
    from server.steps.open_dubbing import glue_prefix_crumbs

    cues = [
        _cue(start=8.26, end=8.63, speech_start=8.26, speech_end=8.63,
             text="Coba sekarang"),
        _cue(start=8.906, end=9.814, speech_start=8.906, speech_end=9.814,
             text="sebelum semua orang tahu."),
    ]
    out = glue_prefix_crumbs(cues)
    assert len(out) == 1
    assert out[0]["text"] == "Coba sekarang sebelum semua orang tahu."
    assert out[0]["speech_start"] == 8.26
    assert out[0]["speech_end"] == 9.814


def test_arabic_medium_is_retried_on_large_v3():
    mod = _od_script()
    cues = [{"text": "ok line", "avg_logprob": -0.2, "no_speech_prob": 0.1}]
    quality = mod._asr_quality(cues, 1.0, "ar")
    assert quality["ok"] is False
    assert any("ar" in r for r in quality["reasons"])


def test_high_no_speech_segment_is_retried():
    mod = _od_script()
    cues = [
        {"text": "hello there friends", "avg_logprob": -0.2, "no_speech_prob": 0.1},
        {"text": "world today now", "avg_logprob": -0.2, "no_speech_prob": 0.97},
    ]
    quality = mod._asr_quality(cues, 1.0, "en")
    assert quality["ok"] is False
    assert any("high_no_speech" in r for r in quality["reasons"])


def test_prefix_crumbs_do_not_merge_across_speakers():
    from server.steps.open_dubbing import glue_prefix_crumbs

    cues = [
        _cue(start=2.28, end=2.55, speech_start=2.28, speech_end=2.55,
             speaker_id="SPEAKER_01", text="Kalau"),
        _cue(start=2.634, end=3.42, speech_start=2.634, speech_end=3.42,
             speaker_id="SPEAKER_02", text="Kalau kasih tahu saya?"),
    ]
    out = glue_prefix_crumbs(cues)
    assert [c["text"] for c in out] == ["Kalau", "Kalau kasih tahu saya?"]


def test_cues_from_payload_glues_prefix_crumbs(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    monkeypatch.setattr(
        "server.steps.open_dubbing.pad_reference",
        lambda src, start, end, dest: Path(dest),
    )
    result = cues_from_payload(
        {
            "vocals": str(vocals),
            "no_vocals": str(music),
            "utterances": [
                {
                    "start": 30.0, "end": 30.166,
                    "speech_start": 30.0, "speech_end": 30.166,
                    "speaker_id": "SPEAKER_00", "text": "I",
                    "avg_logprob": -0.2, "no_speech_prob": 0.05,
                },
                {
                    "start": 30.346, "end": 31.02,
                    "speech_start": 30.346, "speech_end": 31.02,
                    "speaker_id": "SPEAKER_00", "text": "I gotta go.",
                    "avg_logprob": -0.2, "no_speech_prob": 0.05,
                },
            ],
        },
        tmp_path,
    )
    assert len(result["cues"]) == 1
    assert result["cues"][0]["text"] == "I gotta go."
    assert result["cues"][0]["speech_start"] == 30.0


def test_word_timestamps_keep_the_gaps_between_sentences():
    """Pauses in the original stay as gaps on the timeline."""
    mod = _od_script()
    text = "Do this. Then that. All done."
    # Times are relative to the clip (the VAD window), same as Whisper.
    words = [
        {"word": "Do", "start": 0.0, "end": 0.2},
        {"word": " this.", "start": 0.2, "end": 0.6},
        {"word": " Then", "start": 1.2, "end": 1.4},
        {"word": " that.", "start": 1.4, "end": 1.8},
        {"word": " All", "start": 2.5, "end": 2.7},
        {"word": " done.", "start": 2.7, "end": 3.0},
    ]
    pieces = mod.split_long_utterance(10.0, 14.0, text, words)
    assert len(pieces) == 3
    assert pieces[0] == (10.0, 10.6, "Do this.")
    assert pieces[1][0] == 11.2
    assert pieces[1][1] == 11.8
    assert pieces[2][0] == 12.5
    assert pieces[2][1] == 13.0
    assert pieces[1][0] - pieces[0][1] == pytest.approx(0.6)


class _FakeProcess:
    def __init__(self, text, code=0):
        import io

        self.returncode = code
        self.stdout = io.StringIO(text)

    def kill(self):
        return None

    def wait(self):
        return self.returncode


def test_segment_reads_child_json(monkeypatch, tmp_path):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    monkeypatch.setattr("server.config.HF_TOKEN", "hf_x")
    monkeypatch.setattr("server.config.OPEN_DUBBING_PYTHON", str(python))
    monkeypatch.setattr(
        "server.steps.open_dubbing.pad_reference",
        lambda src, start, end, dest: Path(dest).write_bytes(b"r") or Path(dest),
    )

    def fake_popen(*args, **kwargs):
        od = tmp_path / "od"
        od.mkdir(exist_ok=True)
        vocals = od / "vocals.wav"
        music = od / "no_vocals.wav"
        vocals.write_bytes(b"v")
        music.write_bytes(b"m")
        import json

        body = json.dumps({
            "language": "en",
            "language_probability": 0.99,
            "vocals": str(vocals),
            "no_vocals": str(music),
            "utterances": [
                {"start": 0.0, "end": 1.2, "speaker_id": "SPEAKER_00", "text": "hello"},
            ],
        })
        return _FakeProcess("loading model\n" + body + "\n")

    monkeypatch.setattr("server.steps.open_dubbing.subprocess.Popen", fake_popen)
    result = segment(tmp_path / "video.mp4", tmp_path, "medium")
    assert len(result["cues"]) == 1
    assert result["cues"][0]["text"] == "hello"
    assert result["meta"]["language"] == "en"
    assert (tmp_path / "od" / "utterances.json").is_file()


def test_dub_runs_vsr_then_od_then_lipsync(monkeypatch, tmp_path):
    from server.pipeline import Models, _dub
    from server.schemas import DubParams

    order = []
    refs = []
    video = tmp_path / "video.mp4"
    video.write_bytes(b"vid")
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    ref0 = tmp_path / "r0.wav"
    ref1 = tmp_path / "r1.wav"
    ref0.write_bytes(b"r")
    ref1.write_bytes(b"r")

    class Ctx:
        def __init__(self):
            self.params = DubParams(
                remove_subtitle=True, lipsync=True, voice_mode="original"
            )
            self.workdir = tmp_path
            self.job_id = "t"

        def log(self, message):
            return None

        def step(self, name):
            return None

        def check_cancel(self):
            return None

    def fake_vsr(src, dest, *args, **kwargs):
        order.append("vsr")
        dest.write_bytes(b"clean")
        return dest

    def fake_segment(src, work, whisper_model, ctx=None):
        order.append("od")
        return {
            "cues": [
                {"start": 0, "end": 1, "speech_start": 0, "speech_end": 1,
                 "text": "a", "ref_wav": str(ref0)},
                {"start": 2, "end": 3, "speech_start": 2, "speech_end": 3,
                 "text": "b", "ref_wav": str(ref1)},
            ],
            "meta": {"language": "en", "language_probability": 1.0},
            "vocals": vocals,
            "music": music,
        }

    def fake_timed(cues, work, video_seconds, speak, *args, **kwargs):
        order.append("tts")
        for index, cue in enumerate(cues):
            speak(cue["text"], work / f"c{index}.wav", cue)
        out = work / "speech.wav"
        out.write_bytes(b"s")
        return out

    class Voice:
        name = "voxcpm"

        def speak(self, text, out_wav, cfg, steps, reference_wav=None):
            refs.append(str(reference_wav) if reference_wav else None)
            Path(out_wav).write_bytes(b"x")
            return Path(out_wav)

    class Lipsync:
        name = "latentsync"

        def run_shots(self, video, speech, out, work, steps, guidance, ctx=None):
            order.append("lipsync")
            Path(out).write_bytes(b"lip")
            return Path(out)

    def fake_bleed(music, vocals, dest):
        Path(dest).write_bytes(b"m")
        return Path(dest)

    def fake_mix(speech, music, dest, seconds=None):
        Path(dest).write_bytes(b"x")
        return Path(dest)

    def fake_mux(picture, mixed, dest):
        Path(dest).write_bytes(b"out")
        return Path(dest)

    monkeypatch.setattr("server.pipeline.vsr.remove_subtitles", fake_vsr)
    monkeypatch.setattr("server.pipeline.open_dubbing.segment", fake_segment)
    monkeypatch.setattr("server.pipeline.audio.duration", lambda path: 10.0)
    monkeypatch.setattr("server.pipeline.audio.video_size", lambda path: (640, 360))
    monkeypatch.setattr("server.pipeline.audio.suppress_vocal_bleed", fake_bleed)
    monkeypatch.setattr("server.steps.synth.timed_speech", fake_timed)
    monkeypatch.setattr("server.pipeline.audio.mix_audio", fake_mix)
    monkeypatch.setattr(
        "server.pipeline.audio.make_audible",
        lambda src, dest: Path(dest).write_bytes(Path(src).read_bytes()) or Path(dest),
    )
    monkeypatch.setattr("server.pipeline.audio.mux_audio", fake_mux)

    result = _dub(Ctx(), Models(voice=Voice(), lipsync=Lipsync()))
    assert order == ["vsr", "od", "tts", "lipsync"]
    assert refs == [str(ref0), str(ref1)]
    assert result.is_file()


def test_uploaded_reference_wins_over_cue_ref(monkeypatch, tmp_path):
    from server.pipeline import Models, _dub
    from server.schemas import DubParams

    video = tmp_path / "video.mp4"
    video.write_bytes(b"vid")
    uploaded = tmp_path / "reference_audio.wav"
    uploaded.write_bytes(b"u")
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    refs = []

    class Ctx:
        def __init__(self):
            self.params = DubParams(voice_mode="original")
            self.workdir = tmp_path
            self.job_id = "t"

        def log(self, message):
            return None

        def step(self, name):
            return None

        def check_cancel(self):
            return None

    class Voice:
        def speak(self, text, out_wav, cfg, steps, reference_wav=None):
            refs.append(Path(reference_wav) if reference_wav else None)
            Path(out_wav).write_bytes(b"x")
            return Path(out_wav)

    monkeypatch.setattr(
        "server.pipeline.open_dubbing.segment",
        lambda *a, **k: {
            "cues": [{
                "start": 0, "end": 1, "speech_start": 0, "speech_end": 1,
                "text": "a", "ref_wav": str(tmp_path / "r0.wav"),
            }],
            "meta": {},
            "vocals": vocals,
            "music": music,
        },
    )
    monkeypatch.setattr("server.pipeline.audio.duration", lambda path: 4.0)
    monkeypatch.setattr("server.pipeline.audio.video_size", lambda path: (64, 64))
    monkeypatch.setattr(
        "server.pipeline.audio.suppress_vocal_bleed",
        lambda music, vocals, dest: dest.write_bytes(b"m") or dest,
    )

    def fake_timed(cues, work, video_seconds, speak, *args, **kwargs):
        speak(cues[0]["text"], work / "c0.wav", cues[0])
        out = work / "speech.wav"
        out.write_bytes(b"s")
        return out

    monkeypatch.setattr("server.steps.synth.timed_speech", fake_timed)

    def fake_mix(speech, music, dest, seconds=None):
        Path(dest).write_bytes(b"x")
        return Path(dest)

    def fake_mux(picture, mixed, dest):
        Path(dest).write_bytes(b"out")
        return Path(dest)

    monkeypatch.setattr("server.pipeline.audio.mix_audio", fake_mix)
    monkeypatch.setattr(
        "server.pipeline.audio.make_audible",
        lambda src, dest: Path(dest).write_bytes(Path(src).read_bytes()) or Path(dest),
    )
    monkeypatch.setattr("server.pipeline.audio.mux_audio", fake_mux)

    _dub(Ctx(), Models(voice=Voice(), lipsync=None))
    assert refs == [uploaded] or refs == [uploaded.resolve()]


def test_speak_clones_from_the_uploaded_audio(tmp_path):
    from server.pipeline import Models, _speak
    from server.schemas import SpeakParams

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"a")
    refs = []

    class Ctx:
        def __init__(self):
            self.params = SpeakParams(text="hello")
            self.workdir = tmp_path

        def log(self, message):
            return None

        def step(self, name):
            return None

        def check_cancel(self):
            return None

    class Voice:
        def speak(self, text, out_wav, cfg, steps, reference_wav=None):
            refs.append((text, Path(reference_wav)))
            Path(out_wav).write_bytes(b"wav")
            return Path(out_wav)

    result = _speak(Ctx(), Models(voice=Voice(), lipsync=None))
    assert result.read_bytes() == b"wav"
    assert refs == [("hello", audio)]
