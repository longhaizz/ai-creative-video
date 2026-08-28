"""Laying the new voice on the timeline, block by block.

This is the part of the pipeline that fails quietly: nothing crashes when a
line lands in the wrong place or a sentence is cut short, the video just
stops matching the sound.

The TTS model is replaced by ffmpeg tones of a chosen length and the
listener by a dict, so the whole decision tree runs for real, including the
speed changes and the choice between takes.
"""

import json
import shutil
import subprocess

import pytest

from server.jobs import PipelineError
from server.steps import synth
from server.steps.audio import duration
from server.steps.synth import (
    DRIFT_CAP,
    MAX_BLOCK_SECONDS,
    MAX_HESITATION,
    MIN_GAP,
    SOFT_TEMPO,
    Pace,
    build_blocks,
    ends_sentence,
    longest_pause,
    split_to_cap,
    text_error,
    timed_speech,
    with_voice_instruction,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not on PATH",
)


def cue(start, end, text="hello there"):
    return {"start": start, "end": end, "speech_start": start,
            "speech_end": end, "text": text}


# -- voice presets ----------------------------------------------------------


def test_the_preset_goes_in_front_of_the_line():
    assert with_voice_instruction("hello", "male_old").startswith("(An elderly man")
    assert with_voice_instruction("hello", "male_old").endswith("hello")


def test_an_unknown_preset_is_refused():
    with pytest.raises(PipelineError) as error:
        with_voice_instruction("hello", "robot")
    assert error.value.code == "invalid_input"


def test_an_empty_line_is_refused():
    with pytest.raises(PipelineError):
        with_voice_instruction("   ", "male_old")


# -- building the blocks ----------------------------------------------------


def test_a_short_pause_keeps_one_block():
    """A breath inside a sentence is not a place to restart the voice."""
    blocks = build_blocks([cue(0.0, 2.0), cue(2.2, 4.0)], [], 10.0)
    assert len(blocks) == 1
    assert blocks[0]["start"] == 0.0 and blocks[0]["end"] == 4.0


def test_a_real_pause_ends_the_block():
    blocks = build_blocks([cue(0.0, 2.0), cue(3.0, 4.0)], [], 10.0)
    assert len(blocks) == 2
    assert blocks[0]["next_start"] == 3.0, "the pause is room block 1 may use"


def test_a_scene_cut_ends_the_block_and_marks_it_hard():
    blocks = build_blocks([cue(0.0, 2.0), cue(2.2, 4.0)], [2.1], 6.0)
    assert len(blocks) == 2
    assert blocks[0]["hard"] is True, "no drift is allowed across a cut"


def test_a_long_run_of_speech_is_split():
    """VoxCPM wanders on long takes, so a block has a ceiling."""
    cues = [cue(i * 2.0, i * 2.0 + 1.9) for i in range(8)]
    blocks = build_blocks(cues, [], 20.0)
    assert len(blocks) > 1
    assert all(b["end"] - b["start"] <= MAX_BLOCK_SECONDS + 2 for b in blocks)


def test_the_last_block_may_use_the_rest_of_the_video():
    blocks = build_blocks([cue(0.0, 2.0)], [], 10.0)
    assert blocks[0]["next_start"] == 10.0


def test_silent_cues_are_left_out():
    blocks = build_blocks([cue(0.0, 2.0), cue(2.2, 3.0, text="")], [], 6.0)
    assert len(blocks) == 1
    assert blocks[0]["end"] == 2.0


# -- the measured word rate -------------------------------------------------


def test_the_pace_walks_towards_the_real_voice():
    """The old code used one constant for every language. It was wrong."""
    pace = Pace(2.4)
    for _ in range(10):
        pace.observe(10, 2.0)  # the voice really says 5 words a second
    assert 4.5 < pace.value < 5.1
    assert pace.words_for(2.0) >= 9


def test_a_tiny_sample_does_not_move_the_pace():
    pace = Pace(4.0)
    pace.observe(1, 0.2)
    assert pace.value == 4.0


# -- judging a take ---------------------------------------------------------


def test_a_take_that_says_the_line_scores_clean():
    assert text_error("xin chào các bạn", "xin chào các bạn") < 0.01


def test_a_take_that_says_something_else_scores_badly():
    assert text_error("xin chào các bạn", "hôm nay trời mưa") > 0.5


# -- the whole pass ---------------------------------------------------------


def tone_maker(lengths):
    """Stand-in for VoxCPM. `lengths` maps a line to a list of take lengths."""
    calls = []

    def speak(text, out_wav, cue=None):
        calls.append(text)
        seconds = lengths[text]
        if isinstance(seconds, list):
            seconds = seconds[min(calls.count(text), len(seconds)) - 1]
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
             "-c:a", "pcm_s16le", str(out_wav)],
            check=True, capture_output=True,
        )
        return out_wav

    speak.calls = calls
    return speak


def listener(heard=None):
    """Stand-in for whisper. Says it heard exactly what it was given."""
    def listen(wav, lang):
        text = heard if heard is not None else _asked_for(wav)
        return {"text": text, "words": []}
    return listen


_ASKED: dict = {}


def _asked_for(wav):
    return _ASKED.get(str(wav), "")


def run_blocks(monkeypatch, tmp_path, cues, lines, lengths, scenes=(),
               video_seconds=10.0, rewrite=None):
    """Run timed_speech with the model and the translator replaced."""
    monkeypatch.setattr(
        synth, "translate_blocks",
        lambda blocks, lang, key, asr_meta=None: {
            "lines": lines,
            "master_meaning": "meaning",
            "master_translation": " ".join(lines),
            "output_lang_code": "vi",
            "output_lang_name": "Vietnamese",
        },
    )
    if rewrite is not None:
        monkeypatch.setattr(synth, "rephrase_for_duration", rewrite)

    speak = tone_maker(lengths)

    def speak_and_remember(text, out_wav, cue=None):
        path = speak(text, out_wav, cue)
        _ASKED[str(path)] = text
        return path

    out = timed_speech(
        cues, tmp_path, video_seconds, speak_and_remember, "key", "vi",
        meta={"language": "en"}, ctx=None, listen=listener(),
        scenes=list(scenes),
    )
    spoken = json.loads((tmp_path / "spoken_cues.json").read_text(encoding="utf-8"))
    return out, spoken, speak


@needs_ffmpeg
def test_a_block_that_fits_is_left_alone(monkeypatch, tmp_path):
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một ở đây.", "câu hai ở đây."],
        lengths={"câu một ở đây.": 2.0, "câu hai ở đây.": 2.0},
    )
    assert duration(out) == pytest.approx(10.0, abs=0.2)
    assert [s["start"] for s in spoken] == [0.0, 3.0], "kept the original clock"


@needs_ffmpeg
def test_a_long_block_eats_the_pause_instead_of_being_cut(monkeypatch, tmp_path):
    """2s of speech plus a 1s pause is 3s of room, and nothing is trimmed."""
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một dài hơn.", "câu hai ở đây."],
        lengths={"câu một dài hơn.": 2.8, "câu hai ở đây.": 2.0},
    )
    assert spoken[0]["end"] - spoken[0]["start"] == pytest.approx(2.8, abs=0.1)
    assert spoken[1]["start"] == pytest.approx(3.0, abs=0.05), "no drift needed"


@needs_ffmpeg
def test_an_overlong_block_pushes_the_next_one_but_not_past_the_cap(
        monkeypatch, tmp_path):
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một rất dài.", "câu hai ở đây."],
        lengths={"câu một rất dài.": 3.4, "câu hai ở đây.": 2.0},
        rewrite=lambda *a, **k: "",  # the rewrite gives up
    )
    drift = spoken[1]["start"] - 3.0
    assert 0 < drift <= DRIFT_CAP + 0.01, drift
    assert spoken[0]["end"] <= spoken[1]["start"] - MIN_GAP + 0.01


@needs_ffmpeg
def test_the_next_block_returns_to_the_original_clock(monkeypatch, tmp_path):
    """Drift is never carried past one block: every anchor resets it."""
    cues = [cue(0.0, 2.0), cue(3.0, 4.0), cue(6.0, 8.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một rất dài.", "câu hai.", "câu ba ở đây."],
        lengths={"câu một rất dài.": 3.4, "câu hai.": 1.0,
                 "câu ba ở đây.": 2.0},
        rewrite=lambda *a, **k: "",
    )
    assert spoken[1]["start"] > 3.0, "block 2 starts late"
    assert spoken[2]["start"] == pytest.approx(6.0, abs=0.05), "block 3 is on time"


@needs_ffmpeg
def test_a_scene_cut_is_never_crossed_by_rushing_the_voice(monkeypatch, tmp_path):
    """A rushed voice is heard by everyone; a late line by almost no one."""
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một rất dài.", "câu hai ở đây."],
        lengths={"câu một rất dài.": 3.4, "câu hai ở đây.": 2.0},
        scenes=[2.5],
        rewrite=lambda *a, **k: "",
    )
    spoken_length = spoken[0]["end"] - spoken[0]["start"]
    assert spoken_length >= 3.4 / SOFT_TEMPO - 0.1, "never faster than the cap"
    assert spoken[1]["start"] - 3.0 < 0.4, "and the overrun stays small"


@needs_ffmpeg
def test_a_shorter_line_is_asked_for_before_the_speed_is_touched(
        monkeypatch, tmp_path):
    asked = []

    def rewrite(text, seconds, api_key, **kw):
        asked.append((text, round(seconds, 2), kw.get("target_words_n")))
        return "câu ngắn."

    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, speak = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một rất dài.", "câu hai ở đây."],
        lengths={"câu một rất dài.": 3.4, "câu ngắn.": 1.5,
                 "câu hai ở đây.": 2.0},
        rewrite=rewrite,
    )
    assert asked, "the long block must ask for a shorter line first"
    assert asked[0][2], "the rewrite gets the measured word budget"
    assert "câu ngắn." in speak.calls
    assert spoken[1]["start"] == pytest.approx(3.0, abs=0.05), "no drift left"


@needs_ffmpeg
def test_a_babbling_take_loses_to_the_good_one(monkeypatch, tmp_path):
    """VoxCPM sometimes runs away. The second take must win."""
    cues = [cue(0.0, 2.0)]
    out, spoken, speak = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một ở đây."],
        lengths={"câu một ở đây.": [20.0, 2.0]},
        video_seconds=6.0,
    )
    assert speak.calls.count("câu một ở đây.") >= 2, "a bad take is not kept"
    assert spoken[0]["end"] - spoken[0]["start"] < 4.0


@needs_ffmpeg
def test_every_sentence_of_a_block_gets_its_own_subtitle(monkeypatch, tmp_path):
    cues = [cue(0.0, 4.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["Câu một. Câu hai."],
        lengths={"Câu một. Câu hai.": 4.0},
        video_seconds=6.0,
    )
    assert [s["text"] for s in spoken] == ["Câu một.", "Câu hai."]
    assert spoken[0]["end"] <= spoken[1]["start"] + 0.01


# -- keeping the seam off the middle of a sentence --------------------------


def test_a_long_run_is_split_after_a_full_stop():
    """A seam inside a sentence is the one seam a listener notices."""
    cues = [
        cue(0.0, 3.0, "first sentence ends here."),
        cue(3.1, 6.0, "second sentence ends here."),
        cue(6.1, 9.0, "third one runs on and on"),
        cue(9.1, 12.0, "and still keeps going."),
    ]
    pieces = split_to_cap(cues)
    assert len(pieces) == 2
    assert ends_sentence(pieces[0][-1]), "the cut lands on a full stop"


def test_a_run_with_no_full_stop_is_split_at_the_widest_pause():
    cues = [
        cue(0.0, 3.0, "no punctuation here"),
        cue(3.1, 6.0, "still nothing"),
        cue(6.5, 9.0, "after the widest pause"),
        cue(9.1, 12.0, "and the end"),
    ]
    pieces = split_to_cap(cues)
    assert len(pieces) == 2
    assert pieces[1][0]["start"] == 6.5, "split where the speaker breathed"


def test_a_pause_without_a_full_stop_still_ends_a_block_when_it_is_long():
    """Whisper often leaves the full stop out. A second of silence is a break."""
    blocks = build_blocks([cue(0.0, 2.0, "no stop here"),
                           cue(3.5, 5.0, "next thought")], [], 8.0)
    assert len(blocks) == 2


def test_a_small_pause_mid_sentence_keeps_one_block():
    blocks = build_blocks([cue(0.0, 2.0, "half a thought"),
                           cue(2.2, 4.0, "the other half.")], [], 8.0)
    assert len(blocks) == 1, "0.2s is a breath, not a break"


# -- judging how fluent a take is -------------------------------------------


def test_a_take_that_stops_mid_sentence_is_spotted():
    words = [{"start": 0.0, "end": 0.4}, {"start": 1.8, "end": 2.2}]
    assert longest_pause(words) > MAX_HESITATION


def test_a_take_that_runs_on_has_no_long_pause():
    words = [{"start": 0.0, "end": 0.4}, {"start": 0.5, "end": 0.9}]
    assert longest_pause(words) < MAX_HESITATION


@needs_ffmpeg
def test_a_stumbling_take_loses_to_a_fluent_one(monkeypatch, tmp_path):
    from server.steps import synth

    monkeypatch.setattr(
        synth, "translate_blocks",
        lambda blocks, lang, key, asr_meta=None: {
            "lines": ["câu một ở đây."],
            "master_meaning": "m",
            "master_translation": "câu một ở đây.",
            "output_lang_code": "vi",
            "output_lang_name": "Vietnamese",
        },
    )
    speak = tone_maker({"câu một ở đây.": 2.0})
    seen = []

    def listen(wav, lang):
        seen.append(wav)
        if len(seen) == 1:  # the first take says it all, but stops halfway
            return {"text": "câu một ở đây.",
                    "words": [{"start": 0.0, "end": 0.3},
                              {"start": 1.5, "end": 1.9}]}
        return {"text": "câu một ở đây.",
                "words": [{"start": 0.0, "end": 0.3},
                          {"start": 0.4, "end": 0.9}]}

    timed_speech([cue(0.0, 2.0)], tmp_path, 6.0, speak, "key", "vi",
                 meta={"language": "en"}, ctx=None, listen=listen, scenes=[])
    assert len(seen) >= 2, "a take that hesitates is not kept"


@needs_ffmpeg
def test_the_silence_a_take_is_padded_with_is_removed(tmp_path):
    """A block must start speaking on the beat it was given."""
    import subprocess

    from server.steps.audio import clean_take, duration

    raw = tmp_path / "raw.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1",
         "-af", "adelay=300:all=1,apad=pad_dur=0.4",
         "-c:a", "pcm_s16le", str(raw)],
        check=True, capture_output=True,
    )
    assert duration(raw) == pytest.approx(1.7, abs=0.05)
    assert duration(clean_take(raw, tmp_path / "clean.wav")) < 1.3
