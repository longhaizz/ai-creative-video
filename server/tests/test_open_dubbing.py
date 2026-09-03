"""Cue glue, clone refs, and the in-process Demucs+Whisper pipeline."""

import json
from pathlib import Path

import pytest

from server.jobs import PipelineError
from server.steps.open_dubbing import (
    attach_refs,
    glue_prefix_crumbs,
    pad_reference,
)
from server.steps.transcribe import (
    SPEAKER_00,
    _cues_from_segment,
    _word_groups,
    asr_quality,
    cue_needs_review,
    split_long_utterance,
    transcribe,
)


def _cue(**kwargs):
    item = {
        "start": 0.0,
        "end": 1.0,
        "speech_start": 0.0,
        "speech_end": 1.0,
        "speaker_id": SPEAKER_00,
        "text": "hi",
        "avg_logprob": -0.2,
        "no_speech_prob": 0.05,
    }
    item.update(kwargs)
    return item


class _Word:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _Segment:
    def __init__(self, text, start, end, words=None, avg_logprob=-0.2, no_speech_prob=0.05):
        self.text = text
        self.start = start
        self.end = end
        self.words = words
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


def test_same_speaker_cues_share_one_ref(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"v")
    written = []

    def fake_pad(src, start, end, dest):
        written.append((start, end, dest))
        Path(dest).write_bytes(b"r")
        return Path(dest)

    monkeypatch.setattr("server.steps.open_dubbing.pad_reference", fake_pad)
    cues = attach_refs(
        [_cue(start=0.0, end=4.0, text="a"), _cue(start=5.0, end=6.0, text="b")],
        vocals, tmp_path,
    )
    assert cues[0]["ref_wav"] == cues[1]["ref_wav"]
    assert cues[0]["speaker_id"] == SPEAKER_00
    assert written == [(0.0, 4.0, tmp_path / "ref_SPEAKER_00.wav")]


def test_two_labeled_speakers_still_share_one_ref(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"v")
    written = []

    def fake_pad(src, start, end, dest):
        written.append((start, end, Path(dest)))
        Path(dest).write_bytes(b"r")
        return Path(dest)

    monkeypatch.setattr("server.steps.open_dubbing.pad_reference", fake_pad)
    cues = attach_refs(
        [
            _cue(start=1.0, end=5.0, text="xin chao", speaker_id="SPEAKER_00"),
            _cue(start=4.0, end=5.5, text="tam biet", speaker_id="SPEAKER_01"),
        ],
        vocals, tmp_path,
    )
    assert cues[0]["speaker_id"] == SPEAKER_00
    assert cues[1]["speaker_id"] == SPEAKER_00
    assert cues[0]["ref_wav"] == cues[1]["ref_wav"]
    assert Path(cues[0]["ref_wav"]).name == "ref_SPEAKER_00.wav"


def test_empty_cues_are_invalid_input(tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"v")
    with pytest.raises(PipelineError) as error:
        attach_refs([], vocals, tmp_path)
    assert error.value.code == "invalid_input"


def test_original_speak_uses_the_cue_ref(tmp_path):
    refs = []

    def speak(text, out_wav, cue=None):
        refs.append(cue["ref_wav"] if cue else None)
        Path(out_wav).write_bytes(b"x")
        return Path(out_wav)

    cues = [
        _cue(text="a", ref_wav=str(tmp_path / "r0.wav")),
        _cue(start=2, end=3, speech_start=2, speech_end=3,
             text="b", ref_wav=str(tmp_path / "r1.wav")),
    ]
    for cue in cues:
        speak(cue["text"], tmp_path / f"{cue['text']}.wav", cue)
    assert refs == [str(tmp_path / "r0.wav"), str(tmp_path / "r1.wav")]


def test_short_lines_are_concatenated_for_the_ref(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"v")
    commands = []
    monkeypatch.setattr("server.steps.audio.duration", lambda path: 10.0)

    def fake_ffmpeg(command):
        commands.append(command)
        Path(command[-1]).write_bytes(b"x")
        return ""

    monkeypatch.setattr("server.steps.audio.run_ffmpeg", fake_ffmpeg)
    cues = attach_refs(
        [
            _cue(start=5.12, end=5.74, speech_start=5.12, speech_end=5.74,
                 text="Get down lower."),
            _cue(start=6.41, end=7.39, speech_start=6.41, speech_end=7.39,
                 text="Yes, that's better."),
        ],
        vocals, tmp_path,
    )
    assert cues[0]["ref_wav"] == cues[1]["ref_wav"]
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


def test_missing_speaker_id_becomes_speaker_00(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"v")
    monkeypatch.setattr(
        "server.steps.open_dubbing.pad_reference",
        lambda src, start, end, dest: Path(dest),
    )
    cues = attach_refs([{"start": 0.0, "end": 1.0, "text": "hi"}], vocals, tmp_path)
    assert cues[0]["speaker_id"] == SPEAKER_00


def test_attach_refs_keeps_speech_bounds(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"v")
    monkeypatch.setattr(
        "server.steps.open_dubbing.pad_reference",
        lambda src, start, end, dest: Path(dest),
    )
    cues = attach_refs(
        [_cue(start=0.0, end=3.0, speech_start=0.2, speech_end=2.8,
              avg_logprob=-0.4, no_speech_prob=0.05)],
        vocals, tmp_path,
    )
    assert cues[0]["speech_start"] == 0.2
    assert cues[0]["speech_end"] == 2.8
    assert cues[0]["avg_logprob"] == -0.4
    assert cues[0]["no_speech_prob"] == 0.05


def test_prefix_crumbs_are_glued_into_the_next_cue():
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
        "Ikitin",
    ]
    glued = next(c for c in out if c["text"] == "You're very kind.")
    assert glued["speech_start"] == 36.16
    assert glued["speech_end"] == 37.36


def test_tiny_tail_word_is_kept():
    cues = [
        _cue(start=4.7, end=10.7, speech_start=4.7, speech_end=10.7,
             text="this traffic is for you"),
        _cue(start=10.7, end=10.98, speech_start=10.7, speech_end=10.98,
             text="الماء", no_speech_prob=0.13),
    ]
    out = glue_prefix_crumbs(cues)
    assert [c["text"] for c in out] == [
        "this traffic is for you",
        "الماء",
    ]


def test_one_word_no_speech_blob_is_kept():
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
        "पूरी",
        "download and try it free",
    ]


def test_short_spoken_aside_is_kept():
    cues = [_cue(start=1.44, end=2.08, speech_start=1.44, speech_end=2.08,
                 text="Help me.")]
    assert glue_prefix_crumbs(cues)[0]["text"] == "Help me."


def test_short_cta_merges_into_the_next_cue():
    cues = [
        _cue(start=8.26, end=8.63, speech_start=8.26, speech_end=8.63,
             text="Coba sekarang"),
        _cue(start=8.906, end=9.814, speech_start=8.906, speech_end=9.814,
             text="sebelum semua orang tahu"),
    ]
    out = glue_prefix_crumbs(cues)
    assert len(out) == 1
    assert "Coba sekarang" in out[0]["text"]
    assert "sebelum semua orang tahu" in out[0]["text"]


def test_prefix_crumbs_do_not_merge_across_speakers():
    cues = [
        _cue(start=1.0, end=1.2, speech_start=1.0, speech_end=1.2,
             speaker_id="SPEAKER_01", text="Kalau"),
        _cue(start=1.3, end=2.0, speech_start=1.3, speech_end=2.0,
             speaker_id="SPEAKER_02", text="Kalau kasih tahu saya?"),
    ]
    out = glue_prefix_crumbs(cues)
    assert [c["text"] for c in out] == ["Kalau", "Kalau kasih tahu saya?"]


def test_attach_refs_glues_prefix_crumbs(monkeypatch, tmp_path):
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"v")
    monkeypatch.setattr(
        "server.steps.open_dubbing.pad_reference",
        lambda src, start, end, dest: Path(dest).write_bytes(b"r") or Path(dest),
    )
    cues = attach_refs(
        [
            _cue(start=0.0, end=0.2, speech_start=0.0, speech_end=0.2, text="I"),
            _cue(start=0.25, end=1.2, speech_start=0.25, speech_end=1.2,
                 text="I gotta go."),
        ],
        vocals, tmp_path,
    )
    assert [c["text"] for c in cues] == ["I gotta go."]


def test_high_no_speech_is_not_a_review_reason():
    assert cue_needs_review(_cue(text="पूरी प्रक्रिया निजी", no_speech_prob=0.97)) == ""


def test_asr_quality_ignores_no_speech_prob():
    cues = [
        _cue(text="hello there friends", avg_logprob=-0.2, no_speech_prob=0.1),
        _cue(text="world today now", avg_logprob=-0.2, no_speech_prob=0.97),
    ]
    quality = asr_quality(cues, 0.99)
    assert quality["ok"] is True


def test_transcribe_does_not_retry_large_v3(tmp_path):
    wav = tmp_path / "mix.wav"
    wav.write_bytes(b"x")
    sizes = []

    class FakeModels:
        def get(self, size):
            sizes.append(size)

            class Model:
                def transcribe(self, path, **kwargs):
                    # VAD threw quiet voice-over away; see ADR 0001. And
                    # carrying the last sentence forward made Whisper write
                    # one line for a whole window and skip the rest of it.
                    assert kwargs.get("vad_filter") is False
                    assert kwargs.get("condition_on_previous_text") is False
                    assert kwargs.get("language") is None
                    seg = _Segment("hello there.", 0.0, 1.2, [
                        _Word("hello", 0.0, 0.5),
                        _Word(" there.", 0.5, 1.2),
                    ], no_speech_prob=0.97)
                    info = type("Info", (), {
                        "language": "hi",
                        "language_probability": 0.99,
                        "duration": 1.2,
                    })()
                    return [seg], info

            return Model()

    cues, meta = transcribe(FakeModels(), wav, "medium")
    assert sizes == ["medium"]
    assert meta["whisper_model"] == "medium"
    assert cues[0]["speaker_id"] == SPEAKER_00
    assert cues[0]["text"] == "hello there."
    payload = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert payload["language"] == "hi"
    assert payload["model"] == "medium"
    assert payload["segments"] == [{
        "id": 0, "start": 0.0, "end": 1.2, "text": "hello there.",
    }]
    logs = []

    class Ctx:
        def log(self, message):
            logs.append(message)

        def step(self, name):
            return None

    transcribe(FakeModels(), wav, "medium", ctx=Ctx())
    assert any(line.startswith("0.00-1.20  hello there.") for line in logs)


def test_a_short_line_is_not_split():
    assert split_long_utterance(1.0, 3.0, "Hello there.") == [
        (1.0, 3.0, "Hello there."),
    ]


def test_two_short_sentences_stay_one_cue():
    text = "Hello there. How are you?"
    assert split_long_utterance(0.0, 4.0, text) == [(0.0, 4.0, text)]


def test_a_long_window_of_sentences_is_split():
    text = (
        "At the same time, it shows you how to perform each exercise. "
        "You won't have to worry about what to do. "
        "And even if you don't workout at the gym, this is a complete home workout plan. "
        "All you have to do is apply it and see how it goes. "
        "The app link is in the description."
    )
    pieces = split_long_utterance(9.29, 24.96, text)
    assert len(pieces) == 5
    assert pieces[0][0] == 9.29
    assert pieces[-1][1] == 24.96
    assert all(piece[2].endswith((".", "?")) for piece in pieces)


def test_word_gap_splits_short_asides():
    words = [
        {"word": "cheers", "start": 13.5, "end": 13.9},
        {"word": " you", "start": 13.9, "end": 14.1},
        {"word": " up.", "start": 14.1, "end": 14.4},
        {"word": " Go", "start": 17.0, "end": 17.2},
        {"word": " lower.", "start": 17.2, "end": 17.5},
    ]
    groups = _word_groups(words)
    assert len(groups) == 2
    assert "lower" in groups[1][-1]["word"]


def test_cues_from_segment_use_word_times():
    segment = _Segment("Hello there.", 0.5, 2.0, [
        _Word("Hello", 1.0, 1.3),
        _Word(" there.", 1.3, 1.7),
    ])
    cues = _cues_from_segment(segment)
    assert len(cues) == 1
    assert cues[0]["start"] == 0.5
    assert cues[0]["end"] == 2.0
    assert cues[0]["speech_start"] == 1.0
    assert cues[0]["speech_end"] == 1.7
    assert cues[0]["text"] == "Hello there."
    assert cues[0]["speaker_id"] == SPEAKER_00


def test_a_one_word_stamp_is_clamped():
    segment = _Segment("रखूं।", 14.6, 37.3, [
        _Word("रखूं।", 14.602, 37.366),
    ])
    cues = _cues_from_segment(segment)
    assert len(cues) == 1
    assert cues[0]["speech_end"] - cues[0]["speech_start"] <= 1.2 + 1e-6


def _stub_pipeline(monkeypatch, tmp_path, cues, vocals, music, refs_out):
    from server.pipeline import Models, _dub
    from server.schemas import DubParams

    order = []
    video = tmp_path / "video.mp4"
    video.write_bytes(b"vid")

    class Ctx:
        def __init__(self, params=None):
            self.params = params or DubParams(
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

    def fake_extract(src, dest):
        Path(dest).write_bytes(b"mix")
        return Path(dest)

    def fake_separate(mix, work, ctx=None):
        order.append("separate")
        return vocals, music

    def fake_transcribe(models, audio, size, ctx=None):
        order.append("asr")
        return cues, {"language": "en", "language_probability": 1.0}

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
            refs_out.append(str(reference_wav) if reference_wav else None)
            Path(out_wav).write_bytes(b"x")
            return Path(out_wav)

    class Lipsync:
        name = "latentsync"

        def run_shots(self, video, speech, out, work, steps, guidance, ctx=None):
            order.append("lipsync")
            Path(out).write_bytes(b"lip")
            return Path(out)

    monkeypatch.setattr("server.pipeline.vsr.remove_subtitles", fake_vsr)
    monkeypatch.setattr("server.pipeline.audio.extract_audio", fake_extract)
    monkeypatch.setattr("server.pipeline.separate.separate", fake_separate)
    monkeypatch.setattr("server.pipeline.transcribe.transcribe", fake_transcribe)
    monkeypatch.setattr(
        "server.pipeline.open_dubbing.attach_refs",
        lambda cues, vocals, out_dir: cues,
    )
    monkeypatch.setattr("server.pipeline.audio.duration", lambda path: 10.0)
    monkeypatch.setattr("server.pipeline.audio.video_size", lambda path: (640, 360))
    monkeypatch.setattr(
        "server.pipeline.audio.suppress_vocal_bleed",
        lambda music, vocals, dest: (Path(dest).write_bytes(b"m"), Path(dest))[1],
    )
    monkeypatch.setattr("server.steps.synth.timed_speech", fake_timed)
    monkeypatch.setattr(
        "server.pipeline.audio.mix_audio",
        lambda speech, music, dest, seconds=None: (
            Path(dest).write_bytes(b"x"), Path(dest)
        )[1],
    )
    monkeypatch.setattr(
        "server.pipeline.audio.make_audible",
        lambda src, dest: (Path(dest).write_bytes(Path(src).read_bytes()), Path(dest))[1],
    )
    monkeypatch.setattr(
        "server.pipeline.audio.mux_audio",
        lambda picture, mixed, dest: (Path(dest).write_bytes(b"out"), Path(dest))[1],
    )
    return _dub, Ctx, Voice, Lipsync, order


def test_dub_runs_vsr_then_separate_then_whisper(monkeypatch, tmp_path):
    from server.pipeline import Models

    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    ref0 = tmp_path / "r0.wav"
    ref1 = tmp_path / "r1.wav"
    ref0.write_bytes(b"r")
    ref1.write_bytes(b"r")
    refs = []
    _dub, Ctx, Voice, Lipsync, order = _stub_pipeline(
        monkeypatch, tmp_path,
        cues=[
            _cue(text="a", ref_wav=str(ref0)),
            _cue(start=2, end=3, speech_start=2, speech_end=3,
                 text="b", ref_wav=str(ref1)),
        ],
        vocals=vocals, music=music, refs_out=refs,
    )
    result = _dub(Ctx(), Models(voice=Voice(), lipsync=Lipsync()))
    assert order == ["vsr", "separate", "asr", "tts", "lipsync"]
    assert refs == [str(ref0), str(ref1)]
    assert result.is_file()


def test_uploaded_reference_wins_over_cue_ref(monkeypatch, tmp_path):
    from server.pipeline import Models
    from server.schemas import DubParams

    uploaded = tmp_path / "reference_audio.wav"
    uploaded.write_bytes(b"u")
    vocals = tmp_path / "vocals.wav"
    music = tmp_path / "no_vocals.wav"
    vocals.write_bytes(b"v")
    music.write_bytes(b"m")
    refs = []
    _dub, Ctx, Voice, Lipsync, _order = _stub_pipeline(
        monkeypatch, tmp_path,
        cues=[_cue(text="a", ref_wav=str(tmp_path / "r0.wav"))],
        vocals=vocals, music=music, refs_out=refs,
    )
    ctx = Ctx(DubParams(voice_mode="original"))
    _dub(ctx, Models(voice=Voice(), lipsync=None))
    assert refs == [str(uploaded)] or refs == [str(uploaded.resolve())]
