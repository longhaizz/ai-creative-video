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


def _band(reads, cues, frame=1):
    """The subtitle line learned for one frame, or None."""
    bands = sm.subtitle_bands(reads, cues, FPS)
    return bands and bands[frame]


def _erase(reads, cues=CUES):
    """frame -> {box: why it is painted over}."""
    return sm.boxes_to_erase(reads, sm.subtitle_bands(reads, cues, FPS), FPS)


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


def test_letters_without_their_marks_still_match():
    """Seen on a Bosnian ad: OCR read "MOZETE", Whisper wrote "možete"."""
    assert sm.contained("MOZETE", "da možete govoriti") == 1.0
    assert sm.contained("HOM NAY MINH CHIA SE", CUES[0]["text"]) >= sm.MIN_SCORE


def test_a_subtitle_in_dialect_still_matches_the_standard_words():
    """Seen on a Moroccan ad: the subtitle said "انستفرام", Whisper "انستجرام"."""
    assert sm.contained("منشورات انستفرام", "في صورة انستجرام") >= sm.MIN_SCORE


def test_a_short_word_sharing_the_arabic_article_is_chance():
    """Seen on ara1.mp4: "التمن" shares only "الت" with "الترابي", 3 of 5."""
    spoken = "سوف نستخدم هذا الترابي لنرقف الأخطاقية"
    assert sm.contained("التمن", spoken) == 0.0


# -- learning the line -----------------------------------------------------


def test_the_logo_is_off_the_line():
    band = _band(_reads("mình chia sẻ một mẹo"), CUES)
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
    band = _band(reads, CUES)
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
    band = _band(reads, CUES)
    assert all(sm.on_the_line(box, band) for box in (upper, SUB, lower)), band
    assert not sm.on_the_line(LOGO, band)


def test_a_line_next_to_the_subtitles_at_other_times_is_not_grown_into():
    below = (100, 900, 865, 920)
    reads = _reads("mình chia sẻ một mẹo")
    for n in (31, 34, 37):
        reads[n] = [(below, "SHOP NOW", 0.9)]
    band = _band(reads, CUES, 31)
    assert sm.on_the_line(SUB, band)
    assert not sm.on_the_line(below, band)


def test_app_text_in_a_screen_recording_is_not_grown_into():
    """Seen on a Bosnian ad: left-aligned Play Store lines stacked above the
    subtitle, and small buttons inside its band, were all painted out."""
    stack = [(20, 300, 745, 790), (20, 300, 690, 735), (20, 300, 635, 680)]
    button = (450, 550, 830, 855)   # centred, inside the band, but small
    reads = {
        n: [(SUB, "mình chia sẻ một mẹo", 0.9), (button, "Install", 1.0)]
        + [(box, "Events happening now", 1.0) for box in stack]
        for n in range(1, 30, 3)
    }
    band = _band(reads, CUES)
    assert sm.on_the_line(SUB, band)
    assert not any(sm.on_the_line(box, band) for box in stack), band
    assert not sm.on_the_line(button, band)


def test_a_subtitle_that_moves_is_followed():
    """Seen on a Moroccan ad: y=950 for 17 seconds, then near y=750."""
    high = (100, 900, 600, 660)
    far_logo = (200, 400, 200, 260)
    reads = {n: [(SUB, "mình chia sẻ một mẹo", 0.9), (far_logo, "SALE 50%", 0.9)]
             for n in range(1, 30, 3)}
    reads.update({n: [(high, "mình chia sẻ một mẹo", 0.9), (far_logo, "SALE 50%", 0.9)]
                  for n in range(31, 60, 3)})
    bands = sm.subtitle_bands(reads, CUES, FPS)
    assert sm.on_the_line(SUB, bands[1]) and not sm.on_the_line(high, bands[1])
    assert sm.on_the_line(high, bands[55]) and not sm.on_the_line(SUB, bands[55])
    assert not any(sm.on_the_line(far_logo, band) for band in bands.values())


def test_lines_that_do_not_match_between_two_that_do_are_kept():
    """Seen on a Moroccan ad: whole lines were said in other words."""
    grown = (100, 900, 790, 880)    # the same line, animated a little bigger
    reads = {n: [(SUB, "mình chia sẻ một mẹo", 0.9)] for n in (1, 4, 7, 10, 25, 28, 31, 34)}
    reads.update({n: [(grown, "zzzz qqqq", 0.9)] for n in (13, 16, 19, 22)})
    bands = sm.subtitle_bands(reads, CUES, FPS)
    assert all(sm.on_the_line(grown, bands[n]) for n in (13, 16, 19, 22))


def test_a_subtitle_in_another_language_finds_no_line():
    """Nothing matches, so there is no line, and the caller removes nothing."""
    assert _band(_reads("Today I share a small tip"), CUES) is None


def test_text_shown_when_nobody_speaks_does_not_match():
    quiet = [{"start": 50.0, "end": 55.0, "text": CUES[0]["text"]}]
    assert _band(_reads("mình chia sẻ một mẹo"), quiet) is None


def test_too_few_matching_frames_is_not_enough():
    reads = _reads("mình chia sẻ một mẹo", frames=[1, 4])
    assert _band(reads, CUES) is None


# -- text that lingers while the subtitle shows up or goes away ------------

# A word of the subtitle, too short for the line, 0.3s after the last match.
SMALL = (134, 209, 785, 821)


def _with_linger(frame=31, text="mình", box=SMALL):
    reads = _reads("mình chia sẻ một mẹo")        # frames 1..28, marks to 2.70s
    reads[frame] = [(box, text, 0.99)]
    return reads


def test_a_subtitle_shrinking_away_is_still_erased():
    """Seen on 16.mp4: "GET DOWN LOWER" broke into words 36px tall a frame later."""
    reads = _with_linger()
    assert _erase(reads)[31][SMALL] == "linger"


def test_the_nav_bar_item_that_repeats_the_subtitle_is_left():
    """Seen on 16.mp4: "Workout" at the foot of a screen recording, while the
    subtitles said "EVERY WORKOUT IS UNIQUE"."""
    nav = (87, 160, 1223, 1241)
    reads = {n: items + [(nav, "mình", 1.0)]
             for n, items in _reads("mình chia sẻ một mẹo").items()}
    assert all(nav not in frame for frame in _erase(reads).values())


def test_a_small_button_inside_the_band_is_left():
    """Seen on a Bosnian ad: it is in the band, but it says something else."""
    button = (450, 550, 835, 860)
    reads = {n: items + [(button, "Install", 1.0)]
             for n, items in _reads("mình chia sẻ một mẹo").items()}
    assert all(button not in frame for frame in _erase(reads).values())


def test_the_carry_over_does_not_chain():
    """One step from a box on the line, never a step from a step."""
    reads = _with_linger()                 # 3.00s, 0.30s after the last mark
    reads[40] = [(SMALL, "mình", 0.99)]    # 3.90s, 0.90s after that one only
    erase = _erase(reads)
    assert erase[31][SMALL] == "linger"
    assert SMALL not in erase.get(40, {})


def test_a_huge_box_reading_the_subtitle_text_is_left():
    """Or a box the size of half the picture would mask half the picture."""
    huge = (48, 456, 700, 960)             # centred on the line, 260px tall
    reads = _with_linger(box=huge)
    assert huge not in _erase(reads).get(31, {})


def test_a_read_of_two_letters_is_not_enough_to_carry_over():
    """The frame counters and "/18" of an app read as one or two letters."""
    reads = _with_linger(text="mì")
    assert SMALL not in _erase(reads).get(31, {})


# -- the log ---------------------------------------------------------------


def test_the_log_gives_one_line_per_text_with_place_and_verdict():
    reads = _reads("mình chia sẻ một mẹo")
    lines = sm.read_log(reads, CUES, FPS, _erase(reads))
    assert len(lines) == 2, lines
    sub, logo = lines
    assert sub.startswith("OCR 0.00-2.70s y=800-860 x=100-900 ocr=0.90 match=1.00 ERASE")
    assert sub.endswith('"mình chia sẻ một mẹo"')
    assert "y=600-660" in logo and "LEAVE" in logo


def test_without_a_line_the_log_still_shows_the_reads():
    """So the log says why nothing matched."""
    reads = _reads("Today I share a small tip")
    lines = sm.read_log(reads, CUES, FPS, None)
    assert len(lines) == 2
    assert not any("ERASE" in line or "LEAVE" in line for line in lines)


def test_the_log_tells_a_lingering_box_from_a_box_on_the_line():
    lines = [line for line in sm.read_log(_with_linger(), CUES, FPS,
                                          _erase(_with_linger()))
             if "ERASE" in line or "LEAVE" in line]
    assert any("ERASE linger" in line and '"mình"' in line for line in lines), lines
    assert any("ERASE " in line and "linger" not in line for line in lines), lines
    assert any("LEAVE" in line for line in lines), lines


def test_the_same_text_back_later_is_a_new_line():
    reads = {1: [(SUB, "mình chia sẻ", 0.9)], 51: [(SUB, "mình chia sẻ", 0.9)]}
    assert len(sm.read_log(reads, CUES, FPS, None)) == 2


def test_the_log_always_shows_what_was_kept(monkeypatch):
    """Seen on a Moroccan ad: app text used up the cap before the subtitles."""
    monkeypatch.setattr(sm, "MAX_LOG_LINES", 1)
    noise = [((20, 120, y, y + 30), f"menu {y}", 1.0) for y in (100, 200, 300)]
    reads = {n: noise + [(SUB, "mình chia sẻ một mẹo", 0.9)] for n in range(1, 30, 3)}
    lines = sm.read_log(reads, CUES, FPS, _erase(reads))
    assert any("ERASE" in line and "mình chia sẻ" in line for line in lines), lines
    assert lines[-1] == "OCR ... 3 more lines not shown", lines


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
