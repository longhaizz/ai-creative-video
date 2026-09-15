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
    return {n: [(SUB, sub_text, 0.9), (LOGO, "SALE 50%", 0.95)] for n in frames}


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


def test_a_short_word_sharing_the_arabic_article_is_chance():
    """Seen on ara1.mp4: "التمن" shares only "الت" with "الترابي", 3 of 5."""
    spoken = "سوف نستخدم هذا الترابي لنرقف الأخطاقية"
    assert sm.contained("التمن", spoken) == 0.0


# -- learning the line -----------------------------------------------------


def test_the_logo_is_off_the_line():
    band = sm.subtitle_band(_reads("mình chia sẻ một mẹo"), CUES, FPS)
    assert band is not None
    assert sm.on_the_line(SUB, band)
    assert not sm.on_the_line(LOGO, band)


def test_two_lines_of_subtitles_are_both_kept():
    upper = (100, 900, 740, 800)
    reads = {
        n: [(upper, "Hôm nay mình", 0.9), (SUB, "chia sẻ một mẹo", 0.9),
            (LOGO, "SALE", 0.9)]
        for n in range(1, 30, 3)
    }
    band = sm.subtitle_band(reads, CUES, FPS)
    assert sm.on_the_line(upper, band) and sm.on_the_line(SUB, band)
    assert not sm.on_the_line(LOGO, band)


def test_three_lines_are_kept_when_only_the_middle_one_matches():
    """Seen on ara1.mp4: OCR read one line of three well enough to match."""
    upper = (100, 900, 740, 795)
    lower = (100, 900, 862, 915)
    reads = {
        n: [(upper, "xqzw vbnm", 0.9), (SUB, "chia sẻ một mẹo", 0.9),
            (lower, "kjhg fdsa", 0.9), (LOGO, "SALE", 0.9)]
        for n in range(1, 30, 3)
    }
    band = sm.subtitle_band(reads, CUES, FPS)
    assert all(sm.on_the_line(box, band) for box in (upper, SUB, lower)), band
    assert not sm.on_the_line(LOGO, band)


def test_a_line_next_to_the_subtitles_at_other_times_is_not_grown_into():
    below = (100, 900, 865, 920)
    reads = _reads("mình chia sẻ một mẹo")
    for n in (31, 34, 37):
        reads[n] = [(below, "SHOP NOW", 0.9)]
    band = sm.subtitle_band(reads, CUES, FPS)
    assert sm.on_the_line(SUB, band)
    assert not sm.on_the_line(below, band)


def test_a_subtitle_in_another_language_finds_no_line():
    """Nothing matches, so there is no line, and the caller removes nothing."""
    assert sm.subtitle_band(_reads("Today I share a small tip"), CUES, FPS) is None


def test_text_shown_when_nobody_speaks_does_not_match():
    quiet = [{"start": 50.0, "end": 55.0, "text": CUES[0]["text"]}]
    assert sm.subtitle_band(_reads("mình chia sẻ một mẹo"), quiet, FPS) is None


def test_too_few_matching_frames_is_not_enough():
    reads = _reads("mình chia sẻ một mẹo", frames=[1, 4])
    assert sm.subtitle_band(reads, CUES, FPS) is None


# -- the log ---------------------------------------------------------------


def test_the_log_gives_one_line_per_text_with_place_and_verdict():
    reads = _reads("mình chia sẻ một mẹo")
    band = sm.subtitle_band(reads, CUES, FPS)
    lines = sm.read_log(reads, CUES, FPS, band)
    assert len(lines) == 2, lines
    sub, logo = lines
    assert sub.startswith("OCR 0.00-2.70s y=800-860 x=100-900 ocr=0.90 match=1.00 KEEP")
    assert sub.endswith('"mình chia sẻ một mẹo"')
    assert "y=600-660" in logo and "DROP" in logo


def test_without_a_line_the_log_still_shows_the_reads():
    """So the log says why nothing matched."""
    reads = _reads("Today I share a small tip")
    lines = sm.read_log(reads, CUES, FPS, None)
    assert len(lines) == 2
    assert not any("KEEP" in line or "DROP" in line for line in lines)


def test_the_same_text_back_later_is_a_new_line():
    reads = {1: [(SUB, "mình chia sẻ", 0.9)], 51: [(SUB, "mình chia sẻ", 0.9)]}
    assert len(sm.read_log(reads, CUES, FPS, None)) == 2


# -- the file the server writes --------------------------------------------


def test_load_speech_reads_what_the_server_writes(tmp_path):
    path = tmp_path / "speech_cues.json"
    path.write_text(json.dumps({"language": "vi", "cues": CUES + [
        {"start": 11.0, "end": 12.0, "text": "  "}]}), encoding="utf-8")
    speech = sm.load_speech(str(path))
    assert speech == {"language": "vi", "cues": CUES, "error": ""}
    assert sm.REC_MODELS[speech["language"]] == "latin_PP-OCRv5_mobile_rec"


def test_croatian_is_read_with_the_latin_model():
    """Seen on a Bosnian ad: Whisper said hr, and no model meant no removal."""
    assert sm.REC_MODELS["hr"] == "latin_PP-OCRv5_mobile_rec"
    assert sm.REC_MODELS["bs"] == "latin_PP-OCRv5_mobile_rec"


def test_no_path_means_the_filter_was_not_asked_for():
    assert sm.load_speech(None) is None


def test_a_missing_or_broken_file_is_an_error_not_no_filter(tmp_path):
    """Asked for but unreadable: the caller must remove nothing, not everything."""
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    for path in (tmp_path / "never.json", broken):
        speech = sm.load_speech(str(path))
        assert speech["cues"] == [] and speech["error"], speech
