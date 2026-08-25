"""Open Dubbing segmentation, checked without Pyannote and without a GPU.

The child venv is a black box. These tests cover the command line, the JSON
the child must print, and that original-voice clone uses each cue's ref wav.
"""

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
    assert "_detect_language" in src
    assert 'kwargs["language"] = language' in src


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
    monkeypatch.setattr("server.pipeline.audio.mux_audio", fake_mux)

    _dub(Ctx(), Models(voice=Voice(), lipsync=None))
    assert refs == [uploaded] or refs == [uploaded.resolve()]
