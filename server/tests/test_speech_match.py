"""Telling subtitles from other text by what was said.

The module lives in the subtitle remover's venv, but it imports nothing
from there, so it is loaded here straight from its file.
"""

import importlib.util
import itertools
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


def test_the_band_stops_after_three_lines():
    """Seen on 16.mp4: the band walked down a screen recording of an app,
    line by line, until it covered the workout list and the buttons."""
    app = [(430, 570, 880, 940), (430, 570, 960, 1020), (430, 570, 1040, 1100)]
    reads = {n: [(SUB, "mình chia sẻ một mẹo", 0.9)]
                + [(box, f"row {box[2]}", 1.0) for box in app]
             for n in range(1, 30, 3)}
    band = _band(reads, CUES)
    top, bottom, _ = band
    assert bottom - top <= 4 * 60, band          # three lines, plus half a line each side
    assert not sm.on_the_line(app[-1], band)


# Five lines said over ten seconds, and text that reads about half of the
# first one: "minhchiase" out of "minhchiasexyzwqqqq" is 0.56.
MANY_CUES = [{"start": t, "end": t + 2.0,
              "text": CUES[0]["text"] if t == 0 else f"câu thứ {t} không liên quan"}
             for t in (0.0, 2.0, 4.0, 6.0, 8.0)]
HALF_READ = "mình chia sẻ xyzw qqqq"


def test_app_text_that_echoes_one_line_is_not_a_subtitle_track():
    """Seen on a Hindi ad: the voice read the app out loud, so its title and
    buttons matched here and there, and the whole screen was painted out."""
    reads = {n: [(SUB, HALF_READ, 0.9)] for n in range(1, 30, 3)}   # 0.00-2.70s
    assert 0.4 <= sm.contained(HALF_READ, CUES[0]["text"]) < sm.STRONG_SCORE
    assert sm.subtitle_bands(reads, MANY_CUES, FPS) is None


def _follows_the_speech(cues, frames):
    """A subtitle track: on screen the words being said, read only half right."""
    reads = {}
    for n in frames:
        seconds = (n - 1) / FPS
        said = next(c["text"] for c in cues if c["start"] <= seconds < c["end"])
        reads[n] = [(SUB, said[:20] + " xyzw qqqq", 0.9)]
    return reads


def test_a_subtitle_that_follows_every_line_is_kept_even_when_read_badly():
    """Seen on a Moroccan ad: dialect on screen, standard Arabic heard, so
    the real subtitles only scored 0.44-0.67 -- but they were always there."""
    reads = _follows_the_speech(MANY_CUES, range(1, 100, 3))   # 0.00-9.90s
    for items in reads.values():
        _box, text, _score = items[0]
        assert 0.4 <= sm.contained(text, _said_at(text)) < sm.STRONG_SCORE
    band = sm.subtitle_bands(reads, MANY_CUES, FPS)[1]
    assert sm.on_the_line(SUB, band)


def _said_at(read_text):
    """The cue this read came from, for the score check above."""
    return next(c["text"] for c in MANY_CUES
                if sm._letters(c["text"]).startswith(sm._letters(read_text)[:8]))


def test_a_subtitle_read_word_for_word_needs_no_more_proof():
    """16.mp4 read at 1.00, on screen for only part of what was said."""
    reads = {n: [(SUB, CUES[0]["text"], 0.9)] for n in range(1, 30, 3)}
    band = sm.subtitle_bands(reads, MANY_CUES, FPS)[1]
    assert sm.on_the_line(SUB, band)


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


# -- the text that stays on screen, for translating -------------------------


def _screen(reads, cues=CUES):
    """The groups worth translating, as the remover hands them over."""
    bands = sm.subtitle_bands(reads, cues, FPS)
    erase = sm.boxes_to_erase(reads, bands, FPS)
    return sm.screen_text(reads, cues, FPS, erase, bands)


def _texts(groups):
    return sorted(g["text"] for g in groups)


def test_the_subtitle_is_not_screen_text():
    """It is painted out and written again from the speech."""
    assert _texts(_screen(_reads("Hôm nay mình chia sẻ"))) == ["SALE 50%"]


def test_a_logo_nobody_says_is_screen_text():
    groups = _screen(_reads("Hôm nay mình chia sẻ"))
    assert [g["box"] for g in groups] == [LOGO]


def test_an_unsure_read_is_dropped():
    """A white box over a stray mark is worse than leaving the mark."""
    reads = {n: [(SUB, "Hôm nay mình chia sẻ", 0.9), (LOGO, "√", 0.43)]
             for n in range(1, 30, 3)}
    assert _screen(reads) == []


def test_a_read_of_two_letters_is_dropped():
    reads = {n: [(SUB, "Hôm nay mình chia sẻ", 0.9), (LOGO, "OK", 0.95)]
             for n in range(1, 30, 3)}
    assert _screen(reads) == []


def test_text_off_the_line_that_repeats_the_speech_is_dropped():
    """A hook saying what is said would be written twice over."""
    reads = {n: [(SUB, "Hôm nay mình chia sẻ", 0.9),
                 (LOGO, "chia sẻ một mẹo nhỏ", 0.95)]
             for n in range(1, 30, 3)}
    assert _screen(reads) == []


def test_a_box_touching_the_subtitle_band_is_left_alone():
    near = (100, 900, 880, 910)   # just under the line at 800-860
    reads = {n: [(SUB, "Hôm nay mình chia sẻ", 0.9), (near, "DOWNLOAD NOW", 0.95)]
             for n in range(1, 30, 3)}
    assert _screen(reads) == []


def test_text_showing_up_word_by_word_is_one_group():
    """Two reads of a growing line must not stack two boxes."""
    short, long = (100, 480, 600, 660), (100, 590, 600, 660)
    reads = {}
    for n in range(1, 15, 3):
        reads[n] = [(SUB, "Hôm nay mình chia sẻ", 0.9), (short, "AND IT LEARNS", 0.93)]
    for n in range(16, 30, 3):
        reads[n] = [(SUB, "Hôm nay mình chia sẻ", 0.9),
                    (long, "AND IT LEARNS FROM EVERY", 0.97)]
    groups = _screen(reads)
    assert len(groups) == 1
    assert groups[0]["text"] == "AND IT LEARNS FROM EVERY"
    assert groups[0]["box"] == long
    assert groups[0]["first"] == 0.0


def test_screen_text_is_found_even_with_no_subtitle_line():
    """A video with no spoken subtitles still has text on it."""
    reads = {n: [(LOGO, "SALE 50%", 0.95)] for n in range(1, 30, 3)}
    assert sm.subtitle_bands(reads, CUES, FPS) is None
    assert _texts(sm.screen_text(reads, CUES, FPS, {}, None)) == ["SALE 50%"]


def test_a_group_carries_when_it_came_and_went():
    groups = _screen(_reads("Hôm nay mình chia sẻ"))
    assert groups[0]["first"] == 0.0
    assert groups[0]["last"] == 2.7


# -- small print and flashes, from a real Hindi creative --------------------

FRAME_H = 1280
HEADLINE = (126, 594, 121, 185)   # 64px: "गर्भाव्था को प्रबधित करे"
BODY_COPY = (65, 375, 203, 232)   # 29px: a line of the advert's own text
TINY = (152, 458, 244, 262)       # 18px: smaller than any line of an advert


def _with(box, text, frames=range(1, 30, 3), score=0.97):
    """Reads where the subtitle runs throughout and `box` carries `text`."""
    return {n: [(SUB, "Hôm nay mình chia sẻ", 0.9), (box, text, score)]
            for n in frames}


def _screen_h(reads, height=FRAME_H):
    bands = sm.subtitle_bands(reads, CUES, FPS)
    erase = sm.boxes_to_erase(reads, bands, FPS)
    return sm.screen_text(reads, CUES, FPS, erase, bands, frame_height=height)


def test_the_smallest_marks_are_left_alone():
    """Only a floor under the noise. It cannot tell an app store row from a
    line of body copy -- on one Hindi video those were 23-42px and 29-40px
    on the same frame -- and it must not try."""
    assert _screen_h(_with(TINY, "Pregnancy Tracker")) == []


def test_small_body_copy_is_still_translated():
    """A paragraph of the advert reads smaller than its headline. Held to
    the headline's height, a five line block at 17.7s was thrown away and
    the viewer got a video with Hindi still on it."""
    reads = _with(BODY_COPY, "सखत होने की प्क्रिया में हैं।", frames=range(1, 47, 3))
    assert _texts(_screen_h(reads)) == [
        "सखत होने की प्क्रिया में हैं।"]


def test_a_headline_of_the_same_advert_is_kept():
    assert _texts(_screen_h(_with(HEADLINE, "गर्भाव्था को प्रबधित करे"))) == [
        "गर्भाव्था को प्रबधित करे"]


def test_without_a_frame_height_nothing_is_judged_by_size():
    """The height is not always known, and a guess would drop real text."""
    assert _texts(_screen_h(_with(TINY, "Pregnancy Tracker"), height=0)) == [
        "Pregnancy Tracker"]


def test_text_seen_in_a_single_frame_is_a_flash_not_a_message():
    """A row caught mid-scroll was on screen for about a tenth of a second."""
    reads = _with(HEADLINE, "गर्भाव्था को प्रबधित करे")
    reads[16] = [(SUB, "Hôm nay mình chia sẻ", 0.9),
                 (LOGO, "memeriksa", 0.99)]
    assert _texts(_screen_h(reads)) == ["गर्भाव्था को प्रबधित करे"]


def test_a_line_that_shows_up_word_by_word_survives_the_flash_rule():
    """Each part is read once, but together they stayed on screen."""
    reads = {}
    for n, text in zip(range(1, 30, 3), (
            "गर्भ", "गर्भाव्था", "गर्भाव्था को", "गर्भाव्था को प्रब",
            "गर्भाव्था को प्रबधित", "गर्भाव्था को प्रबधित करे",
            "गर्भाव्था को प्रबधित करे", "गर्भाव्था को प्रबधित करे",
            "गर्भाव्था को प्रबधित करे", "गर्भाव्था को प्रबधित करे")):
        reads[n] = [(SUB, "Hôm nay mình chia sẻ", 0.9), (HEADLINE, text, 0.97)]
    groups = _screen_h(reads)
    assert [g["text"] for g in groups] == ["गर्भाव्था को प्रबधित करे"]
    assert groups[0]["last"] > groups[0]["first"]


# -- one paragraph replacing another in the same place ----------------------

PARA_A = (57, 381, 236, 272)    # "खोपड़ी अभी पूरी तरह कठोर नहीं"
PARA_B = (42, 132, 231, 265)    # "शरीर के", the next paragraph, same band


def test_a_paragraph_replaced_by_another_is_not_joined_to_it():
    """On a Hindi video the second paragraph began 0.2s after the first
    ended, in the same band. Joined, it took the whole of the second
    paragraph and one line of the first away with it."""
    reads = {}
    for n in range(1, 41, 3):       # 0.0 - 3.9s
        reads[n] = [(SUB, "Hôm nay mình chia sẻ", 0.9),
                    (PARA_A, "खोपड़ी अभी पूरी तरह कठोर नहीं", 0.97)]
    for n in range(44, 90, 3):      # 4.3 - 8.8s, a 0.4s gap
        reads[n] = [(SUB, "Hôm nay mình chia sẻ", 0.9),
                    (PARA_B, "शरीर के तापमान को", 0.99)]

    groups = _screen_h(reads)
    assert sorted(g["text"] for g in groups) == [
        "खोपड़ी अभी पूरी तरह कठोर नहीं", "शरीर के तापमान को"]


def test_the_same_word_read_several_ways_is_still_one_group():
    """OCR gives "बच्वा", "बबच्चा", "बख्वा" for one word on one video."""
    reads = {}
    for n, text in zip(range(1, 47, 3), itertools.cycle(
            ("बच्वा", "बबच्चा", "बख्वा", "बच्चा", "बच्वा"))):
        reads[n] = [(SUB, "Hôm nay mình chia sẻ", 0.9), (PARA_A, text, 0.85)]
    assert len(_screen_h(reads)) == 1


def test_alike_tells_a_growing_line_from_a_new_one():
    assert sm._alike("AND IT LEARNS", "AND IT LEARNS FROM EVERY")
    assert sm._alike("बच्वा", "बबच्चा")
    assert not sm._alike("खोपड़ी अभी पूरी तरह कठोर नहीं", "शरीर के")
    assert not sm._alike("Loading..", "Tap to continue")


# -- words read one by one, put back into lines -----------------------------
# Boxes are the real ones from the Hindi creative at 22.5-27.2s.

PARAGRAPH = [
    ((41, 120, 155, 188), "आपका"),
    ((124, 185, 153, 191), "शिशु"),
    ((176, 249, 151, 187), "अपनी"),
    ((253, 355, 151, 188), "साँस लेने"),
    ((44, 78, 196, 224), "की"),
    ((81, 172, 193, 227), "प्रक्रिया,"),
    ((171, 236, 195, 225), "पाचन"),
    ((240, 352, 192, 225), "क्रिया और"),
    ((42, 132, 231, 265), "शरीर के"),
    ((135, 221, 235, 264), "तापमान"),
    ((228, 262, 235, 263), "को"),
    ((262, 351, 232, 265), "नियंत्रित"),
    ((102, 189, 271, 305), "करने में"),
    ((188, 257, 272, 307), "सक्षम"),
    ((257, 289, 273, 304), "है।"),
]


def _words(words, frames):
    """Reads of these words in every one of these frames, under a subtitle."""
    return {n: [(SUB, "Hôm nay mình chia sẻ", 0.9)]
               + [(box, text, 1.0) for box, text in words]
            for n in frames}


def test_a_paragraph_read_word_by_word_comes_back_as_one_paragraph():
    """14 white boxes with one word each would read as nonsense, and four
    lines translated one by one come back as fragments."""
    groups = _screen_h(_words(PARAGRAPH, range(1, 47, 3)))
    assert [g["text"] for g in groups] == [
        "आपका शिशु अपनी साँस लेने की प्रक्रिया, पाचन क्रिया और "
        "शरीर के तापमान को नियंत्रित करने में सक्षम है।"
    ]
    assert groups[0]["lines"] == 4
    assert groups[0]["box"] == (41, 355, 151, 307)
    assert 30 <= groups[0]["line_height"] <= 38, "the type size, not the block"


def test_a_line_covers_every_word_it_was_made_of():
    groups = _screen_h(_words(PARAGRAPH[8:12], range(1, 47, 3)))
    assert len(groups) == 1
    assert groups[0]["box"] == (42, 351, 231, 265)


def test_a_two_letter_word_holds_its_line_together():
    """Dropped first for being short, "को" left a 41px hole in the middle of
    "शरीर के तापमान को नियंत्रित" and the line came back in two."""
    groups = _screen_h(_words(PARAGRAPH[8:12], range(1, 47, 3)))
    assert [g["text"] for g in groups] == ["शरीर के तापमान को नियंत्रित"]


def test_a_two_letter_word_alone_is_still_dropped():
    assert _screen_h(_words([((228, 262, 235, 263), "को")], range(1, 47, 3))) == []


def test_a_word_that_stays_does_not_join_two_sentences():
    """"है।" stayed from 22.5s to 32.0s while the words around it changed.
    It must not carry the first sentence into the second."""
    first = [((102, 189, 271, 305), "करने में"), ((188, 257, 272, 307), "सक्षम")]
    second = [((57, 152, 264, 304), "महसूस"), ((158, 212, 265, 296), "कर"),
              ((213, 299, 263, 297), "सकता")]
    reads = {}
    for n in range(1, 95, 3):       # 0.0 - 4.5s, then 4.8 - 9.3s
        reads[n] = [(SUB, "Hôm nay mình chia sẻ", 0.9),
                    ((257, 289, 273, 304), "है।", 1.0)]
        reads[n] += [(box, text, 1.0) for box, text in
                     (first if n < 48 else second)]

    texts = [g["text"] for g in _screen_h(reads)]
    assert not any("सक्षम" in t and "महसूस" in t for t in texts), texts
    assert any(t.startswith("करने में सक्षम") for t in texts), texts
    assert any(t.startswith("महसूस कर सकता") for t in texts), texts


def test_a_line_read_whole_and_word_by_word_is_said_once():
    """OCR gave "अभी" on its own and "अभी आज़माए" whole, lying over it."""
    reads = {}
    for n in range(1, 47, 3):
        reads[n] = [(SUB, "Hôm nay mình chia sẻ", 0.9),
                    ((232, 339, 264, 319), "अभी", 1.0),
                    ((227, 488, 262, 324), "अभी आज़माए", 0.89)]
    assert [g["text"] for g in _screen_h(reads)] == ["अभी आज़माए"]


# -- phone screens: small and quick -------------------------------------------
# Boxes and times from a phone screen recording (AI Photo 19, 720x1280).


def test_a_phone_button_on_screen_for_seconds_is_left_alone():
    """"Regenerate" was 33px tall and there for 2.6s."""
    reads = _with((292, 428, 1071, 1104), "إعادة التوليد", frames=range(1, 28, 3))
    assert _screen_h(reads) == []


def test_a_menu_that_flashes_past_is_left_alone():
    """"أسلوب الذكاء" was there under a second while the menu scrolled."""
    reads = _with((307, 412, 1228, 1256), "أسلوب الذكاء", frames=range(1, 10, 3))
    assert _screen_h(reads) == []


def test_small_text_that_stays_is_still_translated():
    """The Hindi body copy was as small, but stayed 4.7s."""
    reads = _with((65, 375, 164, 196), "आपके बच्े की हड़याँ अभी भी",
                  frames=range(1, 49, 3))
    assert _texts(_screen_h(reads)) == ["आपके बच्े की हड़याँ अभी भी"]


def test_big_text_needs_only_a_second():
    reads = _with(HEADLINE, "गर्भाव्था को प्रबधित करे", frames=range(1, 14, 3))  # 1.2s
    assert _texts(_screen_h(reads)) == ["गर्भाव्था को प्रबधित करे"]


# -- lines stacked into one paragraph ---------------------------------------


def _lines(rows, frames=range(1, 47, 3)):
    """Reads of whole lines: (box, text) each, in every frame."""
    return _words(rows, frames)


def test_lines_far_apart_are_two_blocks():
    """A headline at the top and a caption at the bottom are not one text."""
    groups = _screen_h(_lines([((65, 375, 164, 196), "आपके बच्े की हड़याँ"),
                               ((65, 375, 600, 632), "डाउनलोड करें अभी")]))
    assert len(groups) == 2


def test_side_by_side_columns_are_two_blocks():
    groups = _screen_h(_lines([((20, 300, 164, 196), "आपके बच्े की हड़याँ"),
                               ((420, 700, 200, 232), "डाउनलोड करें अभी")]))
    assert len(groups) == 2


def test_a_heading_much_bigger_than_the_lines_under_it_stays_apart():
    groups = _screen_h(_lines([((65, 600, 60, 160), "बड़ा शीर्षक यहाँ"),
                               ((65, 375, 164, 196), "आपके बच्े की हड़याँ")]))
    assert len(groups) == 2


def test_a_single_line_is_a_paragraph_of_one():
    groups = _screen_h(_lines([((65, 375, 164, 196), "आपके बच्े की हड़याँ")]))
    assert groups[0]["lines"] == 1
    assert groups[0]["line_height"] == 32


def test_a_paragraph_is_drawn_for_as_long_as_any_of_it_is_seen():
    """The middle of the lines' times joins lines; drawn by it, the top line
    of this paragraph would show from under the translation for 2.4s."""
    top = [w for w in PARAGRAPH if w[0][2] < 200]
    reads = _words(PARAGRAPH, range(1, 47, 3))
    for n, found in _words(top, range(49, 71, 3)).items():
        reads[n] = found
    groups = _screen_h(reads)
    assert len(groups) == 1
    assert groups[0]["show_first"] == 0.0
    assert groups[0]["show_last"] == 6.9
    assert groups[0]["last"] < groups[0]["show_last"]


def test_a_word_read_alone_inside_its_line_is_not_a_second_piece():
    """"तुम्हारे यहाँ बच्चा होगा!" in a decorated font read as the whole line,
    and in some frames as "बबच्चा" alone, too unlike to join. Drawn apart,
    the word's box sat on the line's and cut it short."""
    line = ((107, 605, 384, 503), "तुम्हारे यहाँ बच्चा होगा!")
    word = ((351, 476, 425, 486), "बबच्चा")
    reads = _words([line], range(1, 47, 3))
    for n in range(1, 20, 3):
        reads[n] = reads[n] + [(word[0], word[1], 1.0)]
    groups = _screen_h(reads)
    assert [g["text"] for g in groups] == [line[1]]
    assert groups[0]["show_first"] == 0.0


def test_a_different_sentence_inside_a_bigger_box_is_kept():
    """"शरीर के तापमान को नियंत्रित करने में सक्षम है।" lay inside the box of
    a piece joined from other sentences, and was thrown away as a reread."""
    big = ((44, 375, 192, 304), "की सखत होने की प्रक्रिया में हैं अंतर महसूस कर सकता")
    small = ((42, 351, 231, 300), "शरीर के तापमान को नियंत्रित करने में सक्षम है।")
    reads = _words([big], range(1, 47, 3))
    for n in range(1, 47, 3):
        reads[n] = reads[n] + [(small[0], small[1], 1.0)]
    texts = [g["text"] for g in _screen_h(reads)]
    assert small[1] in texts, texts


def test_the_words_behind_the_pieces_can_be_kept():
    words = []
    reads = _words(PARAGRAPH[8:12], range(1, 47, 3))
    bands = sm.subtitle_bands(reads, CUES, FPS)
    sm.screen_text(reads, CUES, FPS, sm.boxes_to_erase(reads, bands, FPS), bands,
                   frame_height=FRAME_H, words_out=words)
    assert sorted(w["text"] for w in words) == sorted(t for _b, t in PARAGRAPH[8:12])
