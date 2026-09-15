"""Telling subtitles from other text by what was said.

The module lives in the subtitle remover's venv, but it imports nothing
from there, so it is loaded here straight from its file.
"""

import importlib.util
import json
from pathlib import Path

_PATH = (Path(__file__).resolve().parents[2]
         / "video-subtitle-remover" / "backend" / "tools" / "speech_match.py")
_spec = importlib.util.spec_from_file_location("speech_match", _PATH)
sm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sm)

FPS = 10.0
SUB = (100, 900, 800, 860)      # the subtitle line
LOGO = (200, 400, 600, 660)     # text in the band that nobody says
CUES = [{"start": 0.0, "end": 10.0, "text": "Hôm nay mình chia sẻ một mẹo nhỏ"}]


def _reads(sub_text, frames=range(1, 30, 3)):
    return {n: [(SUB, sub_text), (LOGO, "SALE 50%")] for n in frames}


# -- scoring one read ------------------------------------------------------


def test_the_same_line_scores_high():
    assert sm.contained("Hôm nay mình chia sẻ", CUES[0]["text"]) >= sm.MIN_SCORE


def test_half_a_sentence_still_scores_high():
    """A karaoke subtitle shows part of what is said."""
    assert sm.contained("chia sẻ một", CUES[0]["text"]) >= sm.MIN_SCORE


def test_small_misreads_still_score_high():
    assert sm.contained("Hôm nav mình chia sé môt mẹo", CUES[0]["text"]) >= sm.MIN_SCORE


def test_a_logo_scores_low_even_if_its_letters_are_in_the_speech():
    """Letters picked one by one out of a long sentence are chance."""
    assert sm.contained("SALE 50%", CUES[0]["text"]) < sm.MIN_SCORE
    assert sm.contained("nhanh", CUES[0]["text"]) < sm.MIN_SCORE


def test_a_short_read_says_nothing():
    assert sm.contained("OK", "ok ok ok") == 0.0


# -- learning the line -----------------------------------------------------


def test_the_logo_is_off_the_line():
    band = sm.subtitle_band(_reads("mình chia sẻ một mẹo"), CUES, FPS)
    assert band is not None
    assert sm.on_the_line(SUB, band)
    assert not sm.on_the_line(LOGO, band)


def test_two_lines_of_subtitles_are_both_kept():
    upper = (100, 900, 740, 800)
    reads = {
        n: [(upper, "Hôm nay mình"), (SUB, "chia sẻ một mẹo"), (LOGO, "SALE")]
        for n in range(1, 30, 3)
    }
    band = sm.subtitle_band(reads, CUES, FPS)
    assert sm.on_the_line(upper, band) and sm.on_the_line(SUB, band)
    assert not sm.on_the_line(LOGO, band)


def test_a_subtitle_in_another_language_turns_the_filter_off():
    """Nothing matches, so there is no line, and the caller keeps every box."""
    assert sm.subtitle_band(_reads("Today I share a small tip"), CUES, FPS) is None


def test_text_shown_when_nobody_speaks_does_not_match():
    quiet = [{"start": 50.0, "end": 55.0, "text": CUES[0]["text"]}]
    assert sm.subtitle_band(_reads("mình chia sẻ một mẹo"), quiet, FPS) is None


def test_too_few_matching_frames_is_not_enough():
    reads = _reads("mình chia sẻ một mẹo", frames=[1, 4])
    assert sm.subtitle_band(reads, CUES, FPS) is None


# -- the file the server writes --------------------------------------------


def test_load_speech_reads_what_the_server_writes(tmp_path):
    path = tmp_path / "speech_cues.json"
    path.write_text(json.dumps({"language": "vi", "cues": CUES + [
        {"start": 11.0, "end": 12.0, "text": "  "}]}), encoding="utf-8")
    speech = sm.load_speech(str(path))
    assert speech == {"language": "vi", "cues": CUES}
    assert sm.REC_MODELS[speech["language"]] == "latin_PP-OCRv5_mobile_rec"


def test_a_missing_or_broken_file_means_no_speech(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert sm.load_speech(None) is None
    assert sm.load_speech(str(tmp_path / "never.json")) is None
    assert sm.load_speech(str(broken)) is None
