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

from server import config
from server.jobs import PipelineError
from pathlib import Path

from server.steps import duration as duration_model
from server.steps import synth
from server.steps.audio import duration
from server.steps.synth import (
    DRIFT_CAP,
    LAST_HIGH,
    MAX_FIT_TRIES,
    MAX_BLOCK_SECONDS,
    MAX_HESITATION,
    MIN_GAP,
    build_blocks,
    ends_sentence,
    longest_pause,
    split_to_cap,
    text_error,
    timed_speech,
    with_voice_instruction,
)

@pytest.fixture(autouse=True)
def no_rewrites(monkeypatch):
    """No test may call OpenAI.

    Handing the same line back is what a translator with nothing new to
    offer does, and fit_block stops as soon as it sees a line it has already
    spoken. Tests that want the rewrite path patch this again themselves.
    """
    monkeypatch.setattr(
        synth, "rewrite_line",
        lambda line, attempts, seconds, words, lang_name, api_key,
        model=None: line,
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


def variants(lines):
    """Turn plain lines into the three lengths the translator returns."""
    out = []
    for line in lines:
        if isinstance(line, dict):
            out.append(line)
        else:
            out.append({"short": line, "normal": line, "long": line})
    return out


def run_blocks(monkeypatch, tmp_path, cues, lines, lengths, scenes=(),
               video_seconds=10.0):
    """Run timed_speech with the model and the translator replaced."""
    entries = variants(lines)
    monkeypatch.setattr(
        synth, "translate_blocks",
        lambda blocks, lang, key, asr_meta=None: {
            "lines": entries,
            "master_meaning": "meaning",
            "master_translation": " ".join(e["normal"] for e in entries),
            "output_lang_code": "vi",
            "output_lang_name": "Vietnamese",
        },
    )
    # The history file is per test, never the real one under data/.
    monkeypatch.setattr(config, "DURATION_DATA", tmp_path / "duration.csv")

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
def test_a_block_stops_when_the_speaker_stops(monkeypatch, tmp_path):
    """The pause after a block belongs to the pause, not to the block.

    2s of speech and then a 1s pause used to count as 3s of room, so the dub
    kept talking while the speaker had already stopped. Now 2.8s of speech is
    squeezed towards the 2.0s the speaker used.
    """
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một dài hơn.", "câu hai ở đây."],
        lengths={"câu một dài hơn.": 2.8, "câu hai ở đây.": 2.0},
    )
    spoken_length = spoken[0]["end"] - spoken[0]["start"]
    assert spoken_length < 2.8 - 0.2, "it no longer eats the pause"
    assert spoken_length == pytest.approx(2.8 / LAST_HIGH, abs=0.15)
    assert spoken[1]["start"] == pytest.approx(3.0, abs=0.05), "no drift needed"


@needs_ffmpeg
def test_an_overlong_block_pushes_the_next_one_but_not_past_the_cap(
        monkeypatch, tmp_path):
    """A line no squeeze can save moves the next block, by the cap at most.

    6s spoken into a 2s slot is still 4.8s after the widest squeeze. The
    block behind it gives up the cap and no more: a block that starts a
    second late is further out than one with a moment of overlap.
    """
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một rất dài.", "câu hai ở đây."],
        lengths={"câu một rất dài.": 6.0, "câu hai ở đây.": 2.0},
    )
    drift = spoken[1]["start"] - 3.0
    assert 0 < drift <= DRIFT_CAP + 0.01, drift


@needs_ffmpeg
def test_the_next_block_returns_to_the_original_clock(monkeypatch, tmp_path):
    """Drift is never carried past one block: every anchor resets it."""
    cues = [cue(0.0, 2.0), cue(3.0, 4.0), cue(6.0, 8.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một rất dài.", "câu hai.", "câu ba ở đây."],
        lengths={"câu một rất dài.": 6.0, "câu hai.": 1.0,
                 "câu ba ở đây.": 2.0},
    )
    assert spoken[1]["start"] > 3.0, "block 2 starts late"
    assert spoken[2]["start"] == pytest.approx(6.0, abs=0.05), "block 3 is on time"


@needs_ffmpeg
def test_a_line_too_long_is_squeezed_to_the_wide_band_and_no_further(
        monkeypatch, tmp_path):
    """A block lands on the speaker's own end, but the voice has a floor.

    3.4s of speech for a 2.0s slot needs 1.7x, which is tape-speed. The
    block is squeezed by the widest we allow and runs over by the rest.
    """
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một rất dài.", "câu hai ở đây."],
        lengths={"câu một rất dài.": 3.4, "câu hai ở đây.": 2.0},
        scenes=[2.5],
    )
    spoken_length = spoken[0]["end"] - spoken[0]["start"]
    assert spoken_length == pytest.approx(3.4 / LAST_HIGH, abs=0.15),         "squeezed by the wide band, not past it"
    assert spoken[1]["start"] - 3.0 < 0.4, "and the overrun stays small"


@needs_ffmpeg
def test_a_long_speech_takes_the_long_line(monkeypatch, tmp_path):
    """The room is how long the speaker spoke, not the pause after them.

    4.5s of speech takes the 4.4s wording. The old rule counted the silence
    up to the next block as room too, which is how the dub ended up talking
    over a pause the speaker had left on purpose.
    """
    cues = [cue(0.0, 4.5, "one."), cue(6.0, 7.0, "two.")]
    entry = {"short": "ngắn.", "normal": "vừa vừa thôi bạn.",
             "long": "dài hơn nhiều, đủ để lấp hết chỗ trống này."}
    out, spoken, speak = run_blocks(
        monkeypatch, tmp_path, cues, lines=[entry, "câu hai."],
        lengths={entry["short"]: 0.6, entry["normal"]: 1.4,
                 entry["long"]: 4.4, "câu hai.": 1.0},
    )
    assert spoken[0]["text"] == entry["long"], spoken
    assert spoken[0]["end"] - spoken[0]["start"] > 1.0, "the room is filled"


@needs_ffmpeg
def test_a_tight_room_takes_the_short_line(monkeypatch, tmp_path):
    """One second of speech takes the wording the model reads as 0.69s."""
    cues = [cue(0.0, 1.0, "one."), cue(1.4, 3.0, "two.")]
    entry = {"short": "ngắn.", "normal": "vừa vừa thôi bạn.",
             "long": "dài hơn nhiều, đủ để lấp hết chỗ trống này."}
    out, spoken, speak = run_blocks(
        monkeypatch, tmp_path, cues, lines=[entry, "câu hai."],
        lengths={entry["short"]: 0.6, entry["normal"]: 1.4,
                 entry["long"]: 4.4, "câu hai.": 1.0},
    )
    assert speak.calls[0] == entry["short"], speak.calls
    assert spoken[1]["start"] == pytest.approx(1.4, abs=0.05), "no drift"


@needs_ffmpeg
def test_every_take_is_written_to_the_history(monkeypatch, tmp_path):
    """The length guess only gets better if the takes are kept."""
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một ở đây.", "câu hai ở đây."],
        lengths={"câu một ở đây.": 2.0, "câu hai ở đây.": 2.0},
    )
    rows = (tmp_path / "duration.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("lang,"), rows[0]
    assert len(rows) == 3, "a header and one row per take"
    assert rows[1].startswith("vi,")


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
            "lines": variants(["câu một ở đây."]),
            "master_meaning": "m",
            "master_translation": "câu một ở đây.",
            "output_lang_code": "vi",
            "output_lang_name": "Vietnamese",
        },
    )
    monkeypatch.setattr(config, "DURATION_DATA", tmp_path / "duration.csv")
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


# -- the number we judge a run by -------------------------------------------


class Recorder:
    """A ctx that only remembers what was logged."""

    def __init__(self):
        self.lines = []

    def log(self, message):
        self.lines.append(message)

    def step(self, name):
        self.lines.append(name)

    def check_cancel(self):
        pass

    def report(self, prefix):
        return next(m for m in self.lines if m.startswith(prefix))


@needs_ffmpeg
def test_silence_is_counted_only_while_the_speaker_talks(monkeypatch, tmp_path):
    """The pause after a block, and the tail of the video, are not holes."""
    entries = variants(["câu một ở đây."])
    monkeypatch.setattr(
        synth, "translate_blocks",
        lambda blocks, lang, key, asr_meta=None: {
            "lines": entries, "master_meaning": "m",
            "master_translation": "m", "output_lang_code": "vi",
            "output_lang_name": "Vietnamese",
        },
    )
    monkeypatch.setattr(config, "DURATION_DATA", tmp_path / "duration.csv")
    ctx = Recorder()
    synth.timed_speech(
        [cue(0.0, 2.0, "one.")], tmp_path, 30.0,
        tone_maker({"câu một ở đây.": 2.0}), "key", "vi",
        meta={"language": "en"}, ctx=ctx, listen=listener(), scenes=[],
    )
    filling = ctx.report("Filling:")
    assert "0.0s silent" in filling, filling
    assert "0 holes" in filling, filling


@needs_ffmpeg
def test_a_block_left_half_silent_is_reported(monkeypatch, tmp_path):
    entries = variants(["ngắn."])
    monkeypatch.setattr(
        synth, "translate_blocks",
        lambda blocks, lang, key, asr_meta=None: {
            "lines": entries, "master_meaning": "m",
            "master_translation": "m", "output_lang_code": "vi",
            "output_lang_name": "Vietnamese",
        },
    )
    monkeypatch.setattr(config, "DURATION_DATA", tmp_path / "duration.csv")
    ctx = Recorder()
    synth.timed_speech(
        [cue(0.0, 4.0, "one.")], tmp_path, 10.0,
        tone_maker({"ngắn.": 1.0}), "key", "vi",
        meta={"language": "en"}, ctx=ctx, listen=listener(), scenes=[],
    )
    filling = ctx.report("Filling:")
    assert "1 holes" in filling, filling


@needs_ffmpeg
def test_a_block_that_ends_early_is_stretched(monkeypatch, tmp_path):
    """The hole this fixes: the dub going quiet while the speaker talks on.

    Nothing used to lengthen a take. 1.8s of voice for 2.0s of speech was
    left as 1.8s, and the last 200ms were silence over a moving mouth.
    """
    cues = [cue(0.0, 2.0), cue(3.0, 5.0)]
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, cues,
        lines=["câu một ở đây.", "câu hai ở đây."],
        lengths={"câu một ở đây.": 1.8, "câu hai ở đây.": 2.0},
    )
    assert spoken[0]["end"] - spoken[0]["start"] == pytest.approx(2.0, abs=0.05)


@needs_ffmpeg
def test_a_new_line_is_asked_for_when_no_wording_fits(monkeypatch, tmp_path):
    """All three lengths miss, so one more is written and it is the one used."""
    entry = {"short": "quá ngắn.", "normal": "quá ngắn.", "long": "quá ngắn."}
    asked = []

    def rewrite(line, attempts, seconds, words, lang_name, api_key,
                model=None):
        asked.append((words, len(attempts)))
        return "câu vừa in đúng chỗ trống này."

    monkeypatch.setattr(synth, "rewrite_line", rewrite)
    out, spoken, _ = run_blocks(
        monkeypatch, tmp_path, [cue(0.0, 3.0), cue(4.0, 5.0)],
        lines=[entry, "câu hai."],
        lengths={"quá ngắn.": 0.5, "câu vừa in đúng chỗ trống này.": 3.0,
                 "câu hai.": 1.0},
    )
    assert asked, "a line this far off must be written again"
    assert spoken[0]["text"] == "câu vừa in đúng chỗ trống này."


def test_the_loop_stops_when_there_is_nothing_new_to_say(monkeypatch, tmp_path):
    """A translator with no other wording must not be paid four times."""
    entry = {"short": "một câu.", "normal": "một câu.", "long": "một câu."}
    calls = []

    monkeypatch.setattr(
        synth, "rewrite_line",
        lambda line, attempts, seconds, words, lang_name, api_key,
        model=None: (calls.append(words) or line),
    )
    model = duration_model.load(tmp_path / "duration.csv")

    def speak(text, out_wav, cue=None):
        Path(out_wav).write_bytes(b"")
        return Path(out_wav)

    def fake_best_take(line, work, name, *args, **kwargs):
        return {"path": Path(work) / f"{name}.wav", "length": 9.0, "heard": "",
                "words": [], "error": 0.0, "hesitation": 0.0, "text": line,
                "takes": 1}

    monkeypatch.setattr(synth, "best_take", fake_best_take)
    take = synth.fit_block(
        entry, 1.0, tmp_path, 0, speak, None, "vi", "Vietnamese",
        model, duration_model.Speed(), "key", None, lambda message: None,
    )
    assert take["tries"] == 1, "only the one wording was ever spoken"
    assert len(calls) == 1, "asked once, and never again once it repeated"


def test_a_sliver_is_folded_into_the_block_before_it():
    """Whisper cut "11.26" in two and left 0.44s holding ".26".

    No sentence fits 0.44s, so that block was squeezed to the floor and
    still ran a second into the next one. It is a tail, not a block.
    """
    cues = [cue(0.0, 2.0, "eleven"), cue(2.1, 2.54, ".26")]
    blocks = build_blocks(cues, [], 10.0)
    assert len(blocks) == 1, blocks
    assert blocks[0]["text"] == "eleven .26"


def test_a_sliver_across_a_scene_cut_is_left_alone():
    """A cut is an anchor. Nothing is allowed to be spoken across one."""
    cues = [cue(0.0, 2.0, "eleven"), cue(2.1, 2.54, ".26")]
    blocks = build_blocks(cues, [2.05], 10.0)
    assert len(blocks) == 2, blocks


def test_the_take_that_says_the_words_wins_over_the_one_that_fits():
    """Length alone kept a take that said something else at the right time."""
    entry = {"short": "một câu.", "normal": "hai câu.", "long": "ba câu."}
    lengths = iter([2.0, 1.0])      # the second one lands on the target
    errors = iter([0.0, 0.9])       # but it says something else

    def fake_best_take(line, work, name, *args, **kwargs):
        return {"path": Path(work) / f"{name}.wav", "length": next(lengths),
                "heard": "", "words": [], "error": next(errors),
                "hesitation": 0.0, "text": line, "takes": 1}

    import server.steps.synth as s
    saved = s.best_take
    s.best_take = fake_best_take
    try:
        take = s.fit_block(
            entry, 1.0, Path("."), 0, None, None, "vi", "Vietnamese",
            duration_model.load(None), duration_model.Speed(), "",
            None, lambda message: None,
        )
    finally:
        s.best_take = saved
    assert take["text"] == "một câu.", "the clean take is kept"


def test_a_wording_of_the_wrong_size_is_not_spoken_at_all(monkeypatch):
    """Four tries is the whole budget; a third of the room does not earn one.

    The first block of a real job spent three of its four tries on wordings
    measured at 1.30x, 1.35x and 0.39x, and the one rewrite that had the
    speed to work with ran out of tries at 1.09x.
    """
    entry = {"short": "một.", "normal": "hai câu ở đây.",
             "long": "ba câu dài hơn nhiều so với hai câu ở đây."}
    asked = []

    def fake_best_take(line, work, name, *args, **kwargs):
        return {"path": Path(work) / f"{name}.wav", "length": 9.0, "heard": "",
                "words": [], "error": 0.0, "hesitation": 0.0, "text": line,
                "takes": 1}

    monkeypatch.setattr(synth, "best_take", fake_best_take)
    monkeypatch.setattr(
        synth, "rewrite_line",
        lambda line, attempts, seconds, words, lang_name, api_key,
        model=None: (asked.append(words) or f"câu viết lại số {len(asked)}."),
    )
    take = synth.fit_block(
        entry, 1.4, Path("."), 0, None, None, "vi", "Vietnamese",
        duration_model.load(None), duration_model.Speed(), "key",
        None, lambda message: None,
    )
    assert take["tries"] == MAX_FIT_TRIES
    assert len(asked) >= 2, "the budget went on written lines, not guesses"


def test_the_word_budget_comes_from_the_line_that_was_spoken():
    """20 words took 5s, so 6s is worth 24 of them. No model needed.

    This is what asked for 33 words three times over while every take came
    back a quarter too long: the fit does not move inside one job, so it
    answered the same thing however loudly the measurements disagreed.
    """
    model = duration_model.load(None)
    speed = duration_model.Speed()
    said = [(" ".join(["từ"] * 20), 5.0)]
    assert synth._word_budget(said, 6.0, model, speed, "vi") == 24


def test_the_word_budget_asks_the_model_only_before_anything_is_spoken():
    model = duration_model.load(None)
    speed = duration_model.Speed()
    assert (synth._word_budget([], 6.0, model, speed, "vi")
            == model.words_for(6.0, "vi"))


def test_the_word_budget_steps_over_a_take_that_babbled():
    """The line nearest the target anchors it, so a runaway is passed over."""
    model = duration_model.load(None)
    speed = duration_model.Speed()
    said = [(" ".join(["từ"] * 12), 2.7), (" ".join(["từ"] * 12), 8.1)]
    assert synth._word_budget(said, 3.28, model, speed, "vi") == 15
