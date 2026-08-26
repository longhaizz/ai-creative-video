"""Fitting the new voice to the old timing.

This is the part of the pipeline that has no obvious right answer and fails
quietly: nothing crashes when a line lands in the wrong place, the video
just stops matching the sound.

The TTS model is replaced by ffmpeg tones of a chosen length, so the whole
decision tree runs for real, including the speed changes.
"""

import shutil
import subprocess

import pytest

from server.jobs import PipelineError
from server.steps.audio import duration
from server.steps.synth import (
    RATIO_KEEP_HI,
    RATIO_KEEP_LO,
    SOFT_SPEEDUP,
    cue_slots,
    fit_cue,
    with_voice_instruction,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not on PATH",
)


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


# -- working out the time slots ---------------------------------------------


def cue(start, end, text="hi"):
    return {"start": start, "end": end, "speech_start": start,
            "speech_end": end, "text": text}


def test_a_slot_is_as_long_as_the_speech_it_replaces():
    slots = cue_slots([cue(0.0, 2.0), cue(5.0, 7.0)], video_seconds=10.0)
    assert slots[0]["target"] == 2.0
    assert slots[1]["target"] == 2.0


def test_the_window_reaches_to_the_next_cue_not_just_its_own_end():
    """The pause after a sentence is room the voice may use if it has to."""
    slots = cue_slots([cue(0.0, 2.0), cue(5.0, 7.0)], video_seconds=10.0)
    assert slots[0]["window"] == 5.0, "2s of speech plus a 3s pause"
    assert slots[0]["target"] == 2.0, "but it should still aim for 2s"


def test_the_last_cue_may_use_the_rest_of_the_video():
    slots = cue_slots([cue(0.0, 2.0), cue(5.0, 7.0)], video_seconds=10.0)
    assert slots[1]["window"] == 5.0


def test_a_target_never_reaches_into_the_next_cue():
    """Whisper can end a cue after the next one starts."""
    slots = cue_slots([cue(0.0, 6.0), cue(4.0, 8.0)], video_seconds=10.0)
    assert slots[0]["target"] == 4.0, "clipped to where the next cue begins"


def test_a_silent_cue_still_holds_its_place():
    """An empty cue keeps its slot, so the one before cannot take its time."""
    cues = [cue(0.0, 2.0), cue(3.0, 5.0, text=""), cue(6.0, 8.0)]
    slots = cue_slots(cues, video_seconds=10.0)
    assert slots[0]["window"] == 3.0, "not 6.0: the silent cue is in the way"


# -- fitting one line -------------------------------------------------------


def tone_maker(tmp_path, lengths):
    """A stand-in for the TTS model. `lengths` maps a line to its seconds."""
    calls = []

    def speak(text, out_wav):
        calls.append(text)
        seconds = lengths[text]
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
             "-c:a", "pcm_s16le", str(out_wav)],
            check=True, capture_output=True,
        )
        return out_wav

    speak.calls = calls
    return speak


def slot(target, window=None):
    return {"start": 0.0, "end": target, "target": target,
            "window": window if window is not None else target}


@needs_ffmpeg
def test_a_line_that_already_fits_is_left_alone(tmp_path):
    speak = tone_maker(tmp_path, {"hello": 2.0})
    out = fit_cue("hello", slot(2.0), tmp_path, 0, speak)
    assert abs(duration(out) - 2.0) < 0.05


@needs_ffmpeg
def test_a_slightly_long_line_is_sped_up(tmp_path):
    speak = tone_maker(tmp_path, {"hello": 2.2})
    out = fit_cue("hello", slot(2.0), tmp_path, 0, speak)
    assert duration(out) < 2.2
    assert abs(duration(out) - 2.0) < 0.1


@needs_ffmpeg
def test_a_slightly_short_line_is_slowed_down(tmp_path):
    speak = tone_maker(tmp_path, {"hello": 1.75})
    out = fit_cue("hello", slot(2.0), tmp_path, 0, speak)
    assert duration(out) > 1.75


@needs_ffmpeg
def test_a_far_too_long_line_is_rewritten(tmp_path):
    speak = tone_maker(tmp_path, {"a long line": 4.0, "short": 2.1})
    asked = []

    def rewrite(text, seconds, shorter):
        asked.append(shorter)
        return "short"

    out = fit_cue("a long line", slot(2.0), tmp_path, 0, speak, rewrite)
    assert asked == [True], "it should ask for something shorter"
    assert speak.calls == ["a long line", "short"]
    assert abs(duration(out) - 2.0) < 0.15


@needs_ffmpeg
def test_a_far_too_short_line_asks_for_a_longer_one(tmp_path):
    speak = tone_maker(tmp_path, {"hi": 1.0, "hello there friend": 1.95})
    asked = []

    def rewrite(text, seconds, shorter):
        asked.append(shorter)
        return "hello there friend"

    fit_cue("hi", slot(2.0), tmp_path, 0, speak, rewrite)
    assert asked == [False], "it should ask for something longer"


@needs_ffmpeg
def test_a_rewrite_that_fits_no_better_is_dropped(tmp_path):
    """Keep the best take, not the newest one."""
    speak = tone_maker(tmp_path, {"long": 4.0, "also long": 4.5})

    def rewrite(text, seconds, shorter):
        return "also long"

    out = fit_cue("long", slot(2.0), tmp_path, 0, speak, rewrite)
    # 4.0 is nearer to 2.0 than 4.5, so the first take wins and is squeezed.
    assert duration(out) < 4.0


@needs_ffmpeg
def test_a_line_that_still_overruns_is_cut(tmp_path):
    """A line that still overruns after speed-up is cut to the window.

    Overlapping the next cue is worse than losing the tail of a word, as long
    as most of the line would remain. A cut that would keep less than half
    is skipped — see test_a_tiny_window_is_not_cut_to_unintelligible.
    """
    speak = tone_maker(tmp_path, {"very long": 6.0})
    out = fit_cue("very long", slot(2.0, window=2.0), tmp_path, 0, speak)
    assert duration(out) <= 2.0 + 0.05
    assert duration(out) < 6.0


@needs_ffmpeg
def test_a_line_may_use_the_pause_after_it(tmp_path):
    """With a wide window there is no need to speed anything up."""
    speak = tone_maker(tmp_path, {"hello": 2.0})
    out = fit_cue("hello", slot(2.0, window=6.0), tmp_path, 0, speak)
    assert abs(duration(out) - 2.0) < 0.05


@needs_ffmpeg
def test_it_stays_off_the_next_cue_when_the_limits_allow(tmp_path):
    """Overlapping speech is worse than speech that is a little too fast.

    This one fits inside the limits. When it does not, see
    test_a_tight_window_is_cut_to_the_window: the tail is cut rather than
    the next cue overlapped.
    """
    speak = tone_maker(tmp_path, {"hello": 3.0})
    out = fit_cue("hello", slot(2.8, window=2.5), tmp_path, 0, speak)
    assert duration(out) <= 2.5 + 0.05


@needs_ffmpeg
def test_cannot_fit_falls_back_to_the_speed_change(tmp_path):
    from server.steps.synth import CANNOT_FIT

    speak = tone_maker(tmp_path, {"long": 4.0})

    def rewrite(text, seconds, shorter):
        return CANNOT_FIT

    # A wide window, so only the soft limit applies.
    out = fit_cue("long", slot(2.0, window=9.0), tmp_path, 0, speak, rewrite)
    assert speak.calls == ["long"], "no second take when nothing can be said"
    assert abs(duration(out) - 4.0 / SOFT_SPEEDUP) < 0.05, "capped at the soft limit"


@needs_ffmpeg
def test_a_tight_window_is_cut_to_the_window(tmp_path):
    """Speed-up first, then cut. The next cue must not be overlapped."""
    speak = tone_maker(tmp_path, {"long": 4.0})
    out = fit_cue("long", slot(2.0, window=2.0), tmp_path, 0, speak)
    length = duration(out)

    assert length < 4.0 / SOFT_SPEEDUP - 0.05, "faster than the soft cap alone"
    assert length <= 2.0 + 0.05, "must not run into the next cue"


@needs_ffmpeg
def test_the_keep_band_is_the_one_the_code_uses(tmp_path):
    """A ratio just inside the band is untouched, just outside is not."""
    inside = RATIO_KEEP_HI - 0.02
    outside = RATIO_KEEP_HI + 0.05
    speak = tone_maker(tmp_path, {"a": 2.0 * inside, "b": 2.0 * outside})

    kept = fit_cue("a", slot(2.0, window=9.0), tmp_path, 0, speak)
    assert abs(duration(kept) - 2.0 * inside) < 0.05

    changed = fit_cue("b", slot(2.0, window=9.0), tmp_path, 1, speak)
    assert duration(changed) < 2.0 * outside - 0.05


def test_the_bands_make_sense():
    assert RATIO_KEEP_LO < 1.0 < RATIO_KEEP_HI


@needs_ffmpeg
def test_a_still_long_line_is_rewritten_twice(tmp_path):
    """Cue 5 on the orchid clip: 4.64s then 4.00s for a 1.65s slot."""
    speak = tone_maker(tmp_path, {
        "a long line": 4.64,
        "still long": 4.00,
        "short": 1.70,
    })
    asked = []

    def rewrite(text, seconds, shorter):
        asked.append(text)
        return {"a long line": "still long", "still long": "short"}[text]

    out = fit_cue("a long line", slot(1.65, window=9.0), tmp_path, 0, speak, rewrite)
    assert asked == ["a long line", "still long"]
    assert abs(duration(out) - 1.70) < 0.15


@needs_ffmpeg
def test_a_hopeless_rewrite_is_not_asked_again(tmp_path):
    """Cue 11: 5.60s then 3.52s for a 0.52s slot. A second rewrite drifted."""
    speak = tone_maker(tmp_path, {"long": 5.60, "still huge": 3.52})
    asked = []

    def rewrite(text, seconds, shorter):
        asked.append(text)
        return "still huge"

    out = fit_cue("long", slot(0.52, window=0.84), tmp_path, 0, speak, rewrite)
    assert asked == ["long"]
    assert duration(out) > 1.5, "must not chop 3.5s down to half a second"


@needs_ffmpeg
def test_a_tiny_window_is_not_cut_to_unintelligible(tmp_path):
    speak = tone_maker(tmp_path, {"very long": 6.0})
    out = fit_cue("very long", slot(0.52, window=0.52), tmp_path, 0, speak)
    assert duration(out) > 2.0
