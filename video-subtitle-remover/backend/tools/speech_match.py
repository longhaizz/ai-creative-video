"""Tell the subtitles apart from other text by what is said at the time.

PATCH (dub server). The detector finds every piece of text in the scan area:
the subtitles, but also logos, prices and calls to action. Subtitles are the
text that is spoken. The dub server runs Whisper first and writes what it
heard to a JSON file. Here the boxes whose text was spoken show where the
subtitle line sits, and the boxes off that line are dropped.

When the filter was asked for but cannot say where the line is, nothing is
removed: a subtitle left on screen is easier to live with than a logo, a
product and a price painted out of the whole frame.

No paddle in this file, so it can be tested without a GPU.
"""

import bisect
import json
import re
import statistics
import unicodedata
from difflib import SequenceMatcher

# Whisper language code -> the PaddleOCR model that reads that script.
# A language missing here means the text cannot be read, so nothing is removed.
REC_MODELS = {
    "zh": "PP-OCRv5_server_rec",
    "ja": "PP-OCRv5_server_rec",
    "en": "PP-OCRv5_server_rec",
    "ko": "korean_PP-OCRv5_mobile_rec",
    "th": "th_PP-OCRv5_mobile_rec",
    "el": "el_PP-OCRv5_mobile_rec",
    "ru": "eslav_PP-OCRv5_mobile_rec",
    "uk": "eslav_PP-OCRv5_mobile_rec",
    "be": "eslav_PP-OCRv5_mobile_rec",
    "ar": "arabic_PP-OCRv5_mobile_rec",
    "hi": "devanagari_PP-OCRv5_mobile_rec",
    "ta": "ta_PP-OCRv5_mobile_rec",
    "te": "te_PP-OCRv5_mobile_rec",
    **{code: "latin_PP-OCRv5_mobile_rec" for code in (
        "vi", "id", "ms", "tl", "fr", "de", "es", "pt", "it", "nl", "pl",
        "tr", "ro", "cs", "sk", "hu", "sv", "da", "no", "fi",
        "hr", "bs", "sl", "sq", "et", "lv", "lt", "ca", "is", "ga", "cy",
        "mt", "sw", "af", "az", "uz", "eu", "gl",
    )},
    # Whisper writes Serbian in Cyrillic.
    **{code: "cyrillic_PP-OCRv5_mobile_rec" for code in (
        "sr", "bg", "mk", "mn", "kk",
    )},
}

# Seconds a subtitle may show before or after its words are said.
TIME_PAD = 0.75
# Share of the read text that must be found in the speech. Low, because the
# subtitle is often not word for word what Whisper wrote: a Moroccan ad
# showed dialect ("واش باقي كتقلب") where Whisper wrote standard Arabic, and
# its real subtitles scored 0.44-0.67. MIN_MATCHED_LETTERS keeps chance out.
MIN_SCORE = 0.4
# Read this well, and the box says the subtitle on its own: OCR read the
# subtitles of 16.mp4 at 1.00, on screen for only part of what was said.
STRONG_SCORE = 0.85
# ...as long as it is word for word for this many different lines. A
# headline repeats one line: an English loan ad showed "Clear payment plan"
# as the voice said it, read word for word in 27 frames, one line of six,
# and the whole band it sat in was painted out.
MIN_STRONG_CUES = 2
# Without such a reading, the text has to follow the speech through the
# video before it counts as a subtitle track. Real subtitles are there for
# nearly every line; a Hindi ad that narrated its own app matched two lines
# of ten, the loan ad one of six.
MIN_CUE_SHARE = 0.5
# Matches this many seconds apart, or closer, learn one line together.
LOCAL_SECONDS = 2.0
# Fewer frames than this that match the speech is too little to trust.
MIN_MATCHED_FRAMES = 3
# A read shorter than this says nothing ("OK", "5%").
MIN_CHARS = 3
# Shorter runs of shared letters are chance: any long sentence holds the
# letters of "SALE" somewhere, in order, one by one.
MIN_RUN = 3
# Fewer shared letters than this is chance, whatever the share. A short word
# that starts with the Arabic article (ال) shares three letters with half the
# words said, and 3 of 5 is already 0.6.
MIN_MATCHED_LETTERS = 6
# A box shorter than this share of the subtitle line is other text: on a
# Bosnian ad the line was 48px and the app buttons under it 25-30px.
MIN_HEIGHT_SHARE = 0.7
# A subtitle grows as it shows up and shrinks as it goes. Those frames are
# too short for the line, so they are erased on the word they still read,
# as long as a box on the line said the same thing this close in time.
LINGER_SECONDS = 1.0
# Subtitles run to three lines, no more. Without a ceiling the band walked
# from the subtitle down through a screen recording of an app on 16.mp4:
# a line 64px tall grew into a band 400px tall, and the workout list, the
# buttons and the timer under it were all painted out.
MAX_BAND_LINES = 3

# How sure OCR must be before a read is worth translating and covering with
# a box of its own. Junk scored 0.00-0.43 on two videos ("|", "√", ""); real
# Hindi words read with a slip still scored 0.61-0.79 ("चर्हली", "होगाd",
# "यहा"), and 0.6 lost some of them. Reads below this are dropped before
# lines are built, so a bad read cannot glue itself into a line.
# ponytail: lowered to 0.5 on request; raise it again if junk gets through
MIN_SCREEN_OCR = 0.5
# A floor on how tall a piece of text has to be before it is worth reading.
# It only keeps out the very smallest marks; it does NOT tell advertising
# copy from the chrome of an app store, and it never will. On one Hindi
# video the app store rows ran 23 to 42 pixels tall and a paragraph of the
# advert's own body copy ran 29 to 40, on the same 1280 tall frame. Set
# higher, to 0.035, this rule threw that paragraph away.
# ponytail: a noise floor, not a filter; the junk needs a different signal
MIN_SCREEN_HEIGHT_SHARE = 0.02
# How alike two reads must look before they count as the same piece of text
# rather than one piece replacing another in the same place. Measured on a
# Hindi creative: one word read several ways ("बच्वा"/"बबच्चा"/"बख्वा",
# "तुम्तरे"/"तुम्हारे") scored 0.57 to 0.80, and two different paragraphs
# sharing a band scored 0.17 to 0.32. Nothing landed in between.
ALIKE_RATIO = 0.5
# When two pieces of text count as words of one line. Measured on a Hindi
# creative: the words of a line sat 3 to 7 pixels apart at about 30 pixels
# tall, and the lines of a paragraph shared none of their height.
LINE_HEIGHT_RATIO = 1.5     # the taller no more than this times the shorter
LINE_SHARED_HEIGHT = 0.5    # share of the shorter one's height in common
LINE_GAP = 1.0              # widest gap between them, in line heights
LINE_SHARED_TIME = 0.5      # share of the shorter one's time on screen together
# How much of the smaller box must lie inside the other for two reads to be
# one piece of text. Words side by side on a line share a pixel or two.
COVER_SHARE = 0.5
# How much of a piece must lie inside a bigger one, on screen at the same
# time, for it to be a second read of part of that piece.
NESTED_SHARE = 0.9
# And how much of its letters must turn up, in order, in the bigger one.
# A word read again scored 0.75 against its line ("बबच्चा" in "तुम्हारे यहाँ
# बच्चा होगा!"); a different sentence in the same place scored 0.27-0.39.
NESTED_TEXT_SHARE = 0.6
# A moment the text on screen changes: at least this many words start, and
# as many words stopped just before. On a Hindi ad one paragraph gave way to
# the next in the same place at 22.5s and 27.3s, 14 and 9 words at once;
# a line read again after a gap brought one or two.
CUT_WORDS = 3
CUT_SECONDS = 0.35
# How long a piece of text must stay on screen before it is translated.
MIN_SCREEN_SECONDS = 1.0
# Text shorter than this share of the frame is phone-screen size, and must
# stay longer: SMALL_TEXT_MIN_SECONDS. See _stayed_on_screen.
# ponytail: 3s split one phone recording (<=2.6s) from body copy (4.7s)
SMALL_TEXT_SHARE = 0.035
SMALL_TEXT_MIN_SECONDS = 3.0
# When stacked lines are one paragraph: no further apart than a line is
# tall, and sharing most of the narrower line's width.
PARAGRAPH_GAP = 1.0
PARAGRAPH_SHARED_WIDTH = 0.5

# The log shows one line per piece of text while it stays on screen. Reads
# of the same text this close in place and time are one line.
MERGE_PX = 20
MERGE_SECONDS = 1.0
# A noisy video can still give hundreds of lines; the job log is not the
# place for all of them.
MAX_LOG_LINES = 200


def load_speech(path):
    """Read the file the dub server wrote.

    Returns None when no path was given: nobody asked for the filter, and
    every box is kept as upstream does. Otherwise a dict with "language",
    "cues" and "error". A file that cannot be read still returns a dict,
    with the reason in "error": the filter was asked for, so the caller
    must remove nothing rather than everything.
    """
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cues = [
            {"start": float(c["start"]), "end": float(c["end"]), "text": str(c["text"])}
            for c in data["cues"]
            if str(c.get("text") or "").strip()
        ]
        return {"language": str(data.get("language") or ""), "cues": cues, "error": ""}
    except Exception as e:
        return {"language": "", "cues": [], "error": f"could not read {path}: {e}"}


def _letters(text):
    # Lower case, no spaces or punctuation. Works the same for scripts
    # written without spaces between words. Marks on letters go too: OCR
    # reads "MOZETE" where Whisper writes "možete", and Vietnamese marks are
    # often read wrong.
    text = "".join(
        c for c in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(c)
    )
    return re.sub(r"[\W_]+", "", text)


def contained(read, spoken):
    """Share of the read text that appears, in order, in the speech (0..1).

    Not how alike the two are: a karaoke subtitle shows half a sentence,
    and half a sentence must still score high against the whole of it.
    """
    a, b = _letters(read), _letters(spoken)
    if len(a) < MIN_CHARS or not b:
        return 0.0
    blocks = SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    shared = sum(m.size for m in blocks if m.size >= MIN_RUN)
    if shared < MIN_MATCHED_LETTERS:
        return 0.0
    return shared / len(a)


def spoken_at(cues, seconds):
    """Everything said around this moment, as one string."""
    return " ".join(
        c["text"] for c in cues
        if c["start"] - TIME_PAD <= seconds <= c["end"] + TIME_PAD
    )


def speech_evidence(reads, cues, fps):
    """What says the text on screen is a subtitle track at all.

    Returns {"matched": frame -> the boxes whose text was said, "strong":
    how many frames read one of them almost word for word, "strong_cues":
    how many different lines those frames read, "cue_share": the share of
    the lines Whisper heard that had matching text on screen}.

    A line counts as matched by what it says, not by when: text on screen
    through the end of one line and the start of the next matches only
    the one whose words it carries.

    Text that is not a subtitle still matches now and then, above all when
    the voice reads out the app it is showing. Subtitles are either read
    word for word or they follow the speech through the whole video; app
    text does neither.
    """
    matched, strong, covered, strong_cues = {}, 0, set(), set()
    for frame_no, items in reads.items():
        seconds = (frame_no - 1) / fps
        now = [i for i, cue in enumerate(cues)
               if cue["start"] - TIME_PAD <= seconds <= cue["end"] + TIME_PAD]
        if not now:
            continue
        spoken = " ".join(cues[i]["text"] for i in now)
        scores = [(box, text, contained(text, spoken)) for box, text, _score in items]
        found = [(box, text) for box, text, score in scores if score >= MIN_SCORE]
        if not found:
            continue
        matched[frame_no] = [box for box, _text in found]
        if max(score for _box, _text, score in scores) >= STRONG_SCORE:
            strong += 1
        for _box, text in found:
            line = max(now, key=lambda i: contained(text, cues[i]["text"]))
            covered.add(line)
            if contained(text, spoken) >= STRONG_SCORE:
                strong_cues.add(line)
    return {"matched": matched, "strong": strong, "strong_cues": len(strong_cues),
            "cue_share": len(covered) / len(cues) if cues else 0.0}


def subtitle_bands(reads, cues, fps):
    """Where the subtitle line sits in each frame, learned from boxes whose text was said.

    reads: frame number (from 1) -> list of (box, text, ocr_score), where a
    box is (xmin, xmax, ymin, ymax). Returns frame number -> (top, bottom,
    line_height) in pixels, or None when too few frames match the speech.

    The line is learned again around every match, from the matches close to
    it in time, and each frame takes the line of the match nearest to it.
    Subtitles move: in one Arabic ad they sat at y=950 for 17 seconds and
    then near y=750, and one line for the whole video missed the first part.
    """
    if fps <= 0:
        return None
    found = speech_evidence(reads, cues, fps)
    matched = found["matched"]
    if len(matched) < MIN_MATCHED_FRAMES:
        return None
    word_for_word = (found["strong"] >= MIN_MATCHED_FRAMES
                     and found["strong_cues"] >= MIN_STRONG_CUES)
    if not word_for_word and found["cue_share"] < MIN_CUE_SHARE:
        return None

    # ponytail: compares every match with every other; fine for the few hundred sampled frames of an ad
    window = LOCAL_SECONDS * fps
    lines = {}
    for frame_no in matched:
        frames = [n for n in matched if abs(n - frame_no) <= window]
        lines[frame_no] = _line(reads, frames, [b for n in frames for b in matched[n]])
    order = sorted(n for n, line in lines.items() if line)
    if not order:
        return None
    bands = {}
    for frame_no in reads:
        i = bisect.bisect_left(order, frame_no)
        nearest = min(order[max(0, i - 1):i + 1], key=lambda n: abs(n - frame_no))
        bands[frame_no] = lines[nearest]
    return bands


def _line(reads, frames, matched):
    """One subtitle line from the boxes that matched in these frames, or None."""
    height = statistics.median(ymax - ymin for _, _, ymin, ymax in matched)
    # One stray match far from the rest -- a hook that repeats the speech --
    # must not stretch the band over the middle of the picture.
    middle = statistics.median((ymin + ymax) / 2 for _, _, ymin, ymax in matched)
    near = [b for b in matched if abs((b[2] + b[3]) / 2 - middle) <= 2 * height]
    if not near:
        return None
    top = min(b[2] for b in near)
    bottom = max(b[3] for b in near)

    # A subtitle of several lines often matches on one line only: OCR reads
    # the others badly. Grow the band over the lines right above and below,
    # as long as they are on screen in a frame where a line matched. A logo
    # that sits just under the subtitles but shows at other times stays out.
    # Subtitle lines are centred on one another; the text of an app shown in
    # a screen recording mostly is not, and without this check it climbed,
    # line by line, halfway up a Play Store page.
    center = statistics.median((xmin + xmax) / 2 for xmin, xmax, _, _ in matched)
    together = [box for n in frames for box, _text, _score in reads[n]]
    grown = True
    while grown:
        grown = False
        for xmin, xmax, ymin, ymax in together:
            if not _line_height(ymax - ymin, height):
                continue
            if abs((xmin + xmax) / 2 - center) > height:
                continue
            touches = ymin <= bottom + height / 2 and ymax >= top - height / 2
            if not touches or (ymin >= top and ymax <= bottom):
                continue
            new_top, new_bottom = min(top, ymin), max(bottom, ymax)
            if new_bottom - new_top > MAX_BAND_LINES * height:
                continue
            top, bottom, grown = new_top, new_bottom, True
    return top - height / 2, bottom + height / 2, height


def _line_height(box_height, height):
    """Is a box this tall a line of the subtitle font?

    Twice the height is still allowed: the detector sometimes draws one box
    around a subtitle of two lines. The floor keeps out the small buttons
    and labels of an app that sit inside the band.
    """
    return MIN_HEIGHT_SHARE * height <= box_height <= 2 * height


def on_the_line(box, band):
    """Is the middle of this box on the subtitle line, and is it about as tall?

    The middle, not the edges: an animated subtitle grows as it shows up
    (66px to 133px tall on one ad) around the same middle.
    """
    top, bottom, height = band
    _, _, ymin, ymax = box
    return top <= (ymin + ymax) / 2 <= bottom and _line_height(ymax - ymin, height)


def boxes_to_erase(reads, bands, fps):
    """frame number -> {box: "line" or "linger"}: everything to paint over.

    Two passes. First the boxes on the subtitle line. Then the boxes that
    read the same words in the same place moments before or after one of
    those: a subtitle grows as it shows up and shrinks as it goes, and OCR
    on 16.mp4 read "GET" 36px tall one tenth of a second after reading
    "GET DOWN LOWER" at 55px.

    Only the height of a line is waived, and only for a box that repeats
    what a box on the line said. The nav bar of an app in a screen
    recording read "Workout" while the subtitles said "EVERY WORKOUT IS
    UNIQUE"; it stays because it sits nowhere near the line.
    """
    if not bands or fps <= 0:
        return {}
    erase = {}
    for frame_no, items in reads.items():
        band = bands.get(frame_no)
        if band is None:
            continue
        on_line = {box: "line" for box, _text, _score in items if on_the_line(box, band)}
        if on_line:
            erase[frame_no] = on_line

    marks = []
    for frame_no, items in reads.items():
        for box, text, _score in items:
            letters = _letters(text)
            if erase.get(frame_no, {}).get(box) == "line" and len(letters) >= MIN_CHARS:
                marks.append((frame_no, letters))
    # ponytail: every candidate against every mark; a few hundred sampled frames of an ad
    window = LINGER_SECONDS * fps
    for frame_no, items in reads.items():
        band = bands.get(frame_no)
        if band is None:
            continue
        top, bottom, height = band
        for box, text, _score in items:
            if erase.get(frame_no, {}).get(box):
                continue
            _, _, ymin, ymax = box
            # The upper bound stays: a box the size of half the picture that
            # happens to read the subtitle would mask half the picture.
            if not top <= (ymin + ymax) / 2 <= bottom or ymax - ymin > 2 * height:
                continue
            letters = _letters(text)
            if len(letters) < MIN_CHARS:
                continue
            if any(abs(n - frame_no) <= window and (letters in mark or mark in letters)
                   for n, mark in marks):
                erase.setdefault(frame_no, {})[box] = "linger"
    return erase


def text_groups(reads, cues, fps):
    """One entry per piece of text, for as long as it stays on screen.

    reads is what OCR read, frame by frame. Reads of the same text in about
    the same place, one after another, are one group. A group carries the
    box and the frame number of its first read, when the text came and
    went in seconds, the best OCR score it got, and how much of it was
    spoken at the time.
    """
    groups, open_groups = [], {}
    for frame_no in sorted(reads):
        seconds = (frame_no - 1) / fps if fps > 0 else 0.0
        spoken = spoken_at(cues, seconds)
        for box, text, score in reads[frame_no]:
            key = _letters(text)
            match = contained(text, spoken) if spoken else 0.0
            group = open_groups.get(key)
            if (group and abs(box[2] - group["box"][2]) <= MERGE_PX
                    and seconds - group["last"] <= MERGE_SECONDS):
                group["last"] = seconds
                group["ocr"] = max(group["ocr"], score)
                group["match"] = max(group["match"], match)
                continue
            group = {"box": box, "text": text, "first": seconds, "last": seconds,
                     "ocr": score, "match": match, "frame": frame_no}
            open_groups[key] = group
            groups.append(group)
    return groups


def screen_text(reads, cues, fps, erase, bands, frame_height=0, words_out=None):
    """The text that stays on screen, ready to be translated.

    Everything the video paints out is left out of this: that is the
    subtitle, and the dub server writes it again from the speech. What is
    left is the text nobody says -- a headline, a price, a call to action.

    OCR reads a paragraph one word at a time, so the words are put back
    into lines before anything is judged or translated. Two things follow
    from that order. A word of two letters -- "को", "की" -- is the glue
    between the words around it, so the length rule waits for the finished
    line; dropped first, it split "शरीर के तापमान को नियंत्रित" in two. And
    what goes to the translator is a line, not "temperature", "control"
    and "digestion" scattered over the picture.

    words_out, when given, is filled with the words the lines were made
    of, so a line that came out wrong can be traced back to them.
    """
    words = []
    for g in text_groups(reads, cues, fps):
        if erase.get(g["frame"], {}).get(g["box"]):
            continue
        if not _worth_reading(g["box"], frame_height):
            continue
        if g["ocr"] < MIN_SCREEN_OCR or not _letters(g["text"]):
            continue
        # Judged per word, not per line: one read of a line of body copy
        # matched the speech at 0.50 by chance, and judged on the line that
        # one read would have taken the whole line away.
        if g["match"] >= MIN_SCORE:
            continue
        words.append(g)
    if words_out is not None:
        words_out.extend(words)
    return pieces_from_words(words, bands, frame_height)


def pieces_from_words(words, bands=None, frame_height=0):
    """Put the words OCR kept back into lines and paragraphs."""
    cuts = _text_cuts(words)
    lines = _join_into_lines(_join_covering(words, cuts), cuts)
    lines = [line for line in lines
             if _worth_translating(line, bands, frame_height)]
    return _drop_nested(_join_into_paragraphs(lines, cuts))


def _text_cuts(words):
    """The moments the text on screen changed all at once, in seconds.

    A paragraph replaced by another in the same place looks, word by word,
    like one text that stayed: the words overlap, their times touch, and
    short Hindi words look alike once their vowel signs are gone ("शिशु
    अपनी" and "शिशु अब" scored 0.67, more than a line read twice). What
    tells them apart is that the whole screen changes at one moment. Words
    on the two sides of such a moment are never joined.
    """
    readable = [w for w in words if len(_letters(w["text"])) >= MIN_CHARS]
    cuts = []
    for t in sorted({w["first"] for w in readable}):
        came = sum(1 for w in readable if w["first"] == t)
        went = sum(1 for w in readable if t - CUT_SECONDS <= w["last"] < t)
        if came >= CUT_WORDS and went >= CUT_WORDS:
            cuts.append(t)
    return cuts


def _apart(a, b, cuts):
    """Is there a cut between these two: one gone before it, one after?"""
    return any(a["last"] < c <= b["first"] or b["last"] < c <= a["first"]
               for c in cuts)


def _letters_in(small, big):
    """Share of small's letters found, in order, in big (0..1).

    Unlike contained(), no floor on how many letters: a short word read
    again is exactly the case this is for.
    """
    a, b = _letters(small), _letters(big)
    if not a:
        return 0.0
    blocks = SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return sum(m.size for m in blocks) / len(a)


def _drop_nested(blocks):
    """Leave out a piece that lies inside a bigger one at the same time.

    OCR reads a line whole in some frames and one word of it in others,
    and a decorated font can read so differently that the two never join:
    "तुम्हारे यहाँ बच्चा होगा!" came back as the whole line and as "बबच्चा"
    alone. Drawn as two pieces, the word's box sat on top of the line's
    and cut the line short. The bigger piece already covers it.
    """
    def area(b):
        xmin, xmax, ymin, ymax = b["box"]
        return (xmax - xmin) * (ymax - ymin)

    kept = []
    for small in blocks:
        inside = [big for big in blocks
                  if big is not small and area(big) > area(small)
                  and _covers(big["box"], small["box"], NESTED_SHARE)
                  and _letters_in(small["text"], big["text"]) >= NESTED_TEXT_SHARE
                  # The middle times, as lines are joined by: a word that
                  # stays on stretches show_first back into the sentence
                  # before, and that sentence would look nested.
                  and big["first"] < small["last"]
                  and small["first"] < big["last"]]
        if not inside:
            kept.append(small)
            continue
        big = max(inside, key=area)
        big["show_first"] = min(big["show_first"], small["show_first"])
        big["show_last"] = max(big["show_last"], small["show_last"])
    return kept


def _worth_translating(line, bands, frame_height=0):
    """The checks that only mean something once the words are a line."""
    if len(_letters(line["text"])) < MIN_CHARS:
        return False
    if _near_the_band(line["box"], bands.get(line["frame"]) if bands else None):
        return False
    return _stayed_on_screen(line, frame_height)


def _join_into_lines(groups, cuts=()):
    """Put the words that OCR read one by one back into their lines.

    Two pieces are one line when they sit on the same row, at about the
    same height, next to each other, at the same time. Lines stacked one
    over another stay apart: each gets its own box, where the old line was.

    Merged until nothing more joins, so the order the words come in does
    not decide which of them end up together.
    """
    # ponytail: pairs of lines until nothing joins; a few hundred words at most
    lines = [_line_of([g]) for g in groups]
    i = 0
    while i < len(lines):
        j = i + 1
        while j < len(lines):
            if (_same_row(lines[i]["box"], lines[j]["box"])
                    and _share_time(lines[i], lines[j], cuts)):
                lines[i] = _line_of(lines[i]["parts"] + lines.pop(j)["parts"])
                j = i + 1   # the line grew, so look at the rest again
            else:
                j += 1
        i += 1
    return sorted(lines, key=lambda l: (l["first"], l["box"][2], l["box"][0]))


def _line_of(parts):
    """One line made of these words, read left to right.

    The time is the middle of the words' own times, not their widest span.
    Some words stay on screen while the rest of a paragraph changes around
    them; taken as the span, one of those carried a line into the next
    sentence's time and joined the two.
    """
    parts = sorted(parts, key=lambda p: p["box"][0])
    readable = _readable_parts(parts)
    # One time per spot, not per read: a spot OCR read five ways, once each,
    # would otherwise outvote the spot's own span and shorten the line.
    spots = [[p for p in parts if _covers(p["box"], r["box"])] for r in readable]
    return {
        "parts": parts,
        "text": " ".join(p["text"] for p in readable),
        "box": (min(p["box"][0] for p in parts), max(p["box"][1] for p in parts),
                min(p["box"][2] for p in parts), max(p["box"][3] for p in parts)),
        "first": statistics.median(min(p["first"] for p in spot) for spot in spots),
        "last": statistics.median(max(p["last"] for p in spot) for spot in spots),
        # When to draw the translation. The middle of the words, like first
        # and last: one word that stays on, or one read by mistake, would
        # otherwise stretch the line over the next sentence.
        "show_first": statistics.median(min(p["first"] for p in spot) for spot in spots),
        "show_last": statistics.median(max(p["last"] for p in spot) for spot in spots),
        "ocr": min(p["ocr"] for p in parts),
        "match": max(p["match"] for p in parts),
        "frame": min(parts, key=lambda p: p["first"])["frame"],
    }


def _join_into_paragraphs(lines, cuts=()):
    """Put lines stacked into one block back together as one paragraph.

    A paragraph sent line by line comes back as fragments -- "Your baby
    its breathing" -- because the sentence runs across the lines. Joined,
    it is translated as one sentence and drawn as one box over the block.

    Lines belong together when they were on screen at the same time, are
    about as tall, sit one right under another, and share most of their
    width. On a Hindi video the lines of a paragraph sat 1 to 7 pixels
    apart at about 30 pixels tall.
    """
    # ponytail: pairs of paragraphs until nothing joins, like _join_into_lines
    blocks = [_paragraph_of([line]) for line in lines]
    i = 0
    while i < len(blocks):
        j = i + 1
        while j < len(blocks):
            if (_stacked(blocks[i], blocks[j])
                    and _share_time(blocks[i], blocks[j], cuts)):
                blocks[i] = _paragraph_of(blocks[i]["parts"] + blocks.pop(j)["parts"])
                j = i + 1
            else:
                j += 1
        i += 1
    return sorted(blocks, key=lambda b: (b["first"], b["box"][2], b["box"][0]))


def _paragraph_of(lines):
    """One paragraph made of these lines, read top to bottom."""
    lines = sorted(lines, key=lambda l: l["box"][2])
    return {
        "parts": lines,
        "text": " ".join(l["text"] for l in lines),
        "box": (min(l["box"][0] for l in lines), max(l["box"][1] for l in lines),
                min(l["box"][2] for l in lines), max(l["box"][3] for l in lines)),
        "first": statistics.median(l["first"] for l in lines),
        "last": statistics.median(l["last"] for l in lines),
        # A paragraph is drawn while any of its lines is on screen.
        "show_first": min(l["show_first"] for l in lines),
        "show_last": max(l["show_last"] for l in lines),
        "ocr": min(l["ocr"] for l in lines),
        "match": max(l["match"] for l in lines),
        "frame": min(lines, key=lambda l: l["first"])["frame"],
        # The size of the type, which the box of a whole block no longer says.
        "line_height": statistics.median(l["box"][3] - l["box"][2] for l in lines),
        "lines": len(lines),
    }


def _stacked(a, b):
    """Does one of these blocks sit right under the other, in the same column?"""
    axmin, axmax, aymin, aymax = a["box"]
    bxmin, bxmax, bymin, bymax = b["box"]
    ha, hb = a["line_height"], b["line_height"]
    if min(ha, hb) <= 0 or max(ha, hb) > LINE_HEIGHT_RATIO * min(ha, hb):
        return False
    gap = max(aymin, bymin) - min(aymax, bymax)    # below zero when they overlap
    if gap > PARAGRAPH_GAP * max(ha, hb):
        return False
    shared = min(axmax, bxmax) - max(axmin, bxmin)
    narrower = min(axmax - axmin, bxmax - bxmin)
    return narrower > 0 and shared >= PARAGRAPH_SHARED_WIDTH * narrower


def _readable_parts(parts):
    """The parts of a line to read out, one per spot, left to right.

    OCR reads a line whole now and then and word by word the rest of the
    time, so a line holds "अभी" and "अभी आज़माए" lying over each other.
    Read out as they are, that is "अभी आज़माए अभी". The longest read of a
    spot stands for it; the box still covers every read, because every
    read was ink on the picture.
    """
    ranked = sorted(parts, key=lambda p: (len(_letters(p["text"])), p["ocr"]),
                    reverse=True)
    kept = []
    for part in ranked:
        if not any(_covers(part["box"], k["box"]) for k in kept):
            kept.append(part)
    return sorted(kept, key=lambda p: p["box"][0])


def _same_row(a, b):
    """Are these two boxes words of one line?

    About as tall, most of their height shared, and no wider apart than a
    line is tall. On a Hindi video the words of one line sat 3 to 7 pixels
    apart at 30 pixels tall, and the lines of a paragraph did not share
    their height at all.
    """
    axmin, axmax, aymin, aymax = a
    bxmin, bxmax, bymin, bymax = b
    ha, hb = aymax - aymin, bymax - bymin
    if min(ha, hb) <= 0 or max(ha, hb) > LINE_HEIGHT_RATIO * min(ha, hb):
        return False
    shared = min(aymax, bymax) - max(aymin, bymin)
    if shared < LINE_SHARED_HEIGHT * min(ha, hb):
        return False
    gap = max(axmin, bxmin) - min(axmax, bxmax)    # below zero when they overlap
    return gap <= LINE_GAP * max(ha, hb)


def _share_time(a, b, cuts=()):
    """Were these two on screen together for most of the shorter one's time?"""
    if _apart(a, b, cuts):
        return False
    shared = min(a["last"], b["last"]) - max(a["first"], b["first"])
    shorter = min(a["last"] - a["first"], b["last"] - b["first"])
    if shorter <= 0:
        return shared >= 0
    return shared >= LINE_SHARED_TIME * shorter


def _worth_reading(box, frame_height):
    """Is this text big enough to be a message rather than small print?

    A Hindi creative ended on a screen recording of an app store, and every
    row of it -- "Contains ads", "14 MB", a developer name -- is text on
    screen by any reading, so all of it came back to be translated and
    covered with a white box.

    Height does not tell the two apart: on that same video the app store
    rows ran 23 to 42 pixels tall and a paragraph of the advert's own body
    copy ran 29 to 40. The rule here is only a floor under the smallest
    marks, because dropping a line the advert meant to be read is the worse
    of the two mistakes -- the viewer is left with a language they cannot
    read, and nothing in the output says why.
    """
    if frame_height <= 0:
        return True
    _, _, ymin, ymax = box
    return (ymax - ymin) >= MIN_SCREEN_HEIGHT_SHARE * frame_height


def _stayed_on_screen(group, frame_height=0):
    """Was this text there long enough to be worth covering over?

    A piece gone in a moment is a row caught mid-scroll or a frame of a
    transition; covering it with a white box draws the eye to a flash that
    nobody was reading.

    Small text has to stay longer. The buttons and labels of a phone
    screen recording ("Save", "Regenerate", "World Cup") were 26 to 36
    pixels tall and on screen 0.1 to 2.6s. Height alone cannot drop them:
    a paragraph of body copy on another video was 29 to 40 pixels tall.
    That paragraph stayed 4.7s, though, and the phone screen never did.
    """
    seen = group["last"] - group["first"]
    _, _, ymin, ymax = group["box"]
    small = frame_height > 0 and ymax - ymin < SMALL_TEXT_SHARE * frame_height
    return seen >= (SMALL_TEXT_MIN_SECONDS if small else MIN_SCREEN_SECONDS)

def _near_the_band(box, band):
    """Does this box touch the band the subtitles sit on?

    Not on_the_line, which asks whether the box IS a subtitle line. Here we
    only need to keep away from where the new subtitles will be drawn, so
    touching the band at all is reason enough to leave the box alone.
    """
    if band is None:
        return False
    top, bottom, _height = band
    _, _, ymin, ymax = box
    return ymin <= bottom and ymax >= top


def _join_covering(groups, cuts=()):
    """Join the groups that sit on each other at the same time.

    Text that shows up word by word reads as several different strings in
    one place, one after another: "AND IT LEARNS FROM", then "AND IT LEARNS
    FROM EVERY". Left alone, each would get its own translation and its own
    white box, drawn on top of the last. The longest read is the one where
    the text had finished showing up, so it is the only one worth keeping.

    Looking alike is what tells that apart from the other thing that happens
    in one place: a paragraph going away and another taking its spot. Those
    two are as close in time and place as a line growing -- 0.2s apart on
    one Hindi video -- and joining them threw a whole paragraph of the
    advert away, and a line of the paragraph before it.
    """
    # ponytail: every group against every kept one; an ad gives a few dozen
    out = []
    for g in sorted(groups, key=lambda g: g["first"]):
        for kept in out:
            if _one_piece(g, kept, cuts):
                if len(_letters(g["text"])) > len(_letters(kept["text"])):
                    kept["text"], kept["box"] = g["text"], g["box"]
                kept["first"] = min(kept["first"], g["first"])
                kept["last"] = max(kept["last"], g["last"])
                break
        else:
            out.append(dict(g))
    return out


def _one_piece(a, b, cuts=()):
    """Are these two groups the same piece of text, read twice?

    They have to lie over each other, not just touch: two words side by
    side on a line touch at the edge and are two words. And the reads have
    to look alike -- a line still showing up, a word read badly. A
    paragraph and the one that replaces it lie over each other too, and do
    not look alike.

    Being on screen at the same time is not enough on its own. A short word
    that stays while the sentence around it changes lies over a word of the
    next sentence, and joined to it, pulled that word back into the
    sentence before.
    """
    return (_covers(a["box"], b["box"]) and _times_touch(a, b)
            and not _apart(a, b, cuts)
            and _alike(a["text"], b["text"]))


def _covers(a, b, share=COVER_SHARE):
    """Does most of the smaller of these two boxes lie inside the other?"""
    axmin, axmax, aymin, aymax = a
    bxmin, bxmax, bymin, bymax = b
    width = min(axmax, bxmax) - max(axmin, bxmin)
    height = min(aymax, bymax) - max(aymin, bymin)
    if width <= 0 or height <= 0:
        return False
    smaller = min((axmax - axmin) * (aymax - aymin), (bxmax - bxmin) * (bymax - bymin))
    return smaller > 0 and width * height >= share * smaller


def _alike(a, b):
    """Do these two reads look like the same words?

    Not whether they are equal: OCR reads one word several ways as it comes
    and goes. One read inside the other is a line still showing up; a high
    share of letters in common is the same words read badly. Anything less
    is another piece of text that happens to sit in the same place.
    """
    x, y = _letters(a), _letters(b)
    if not x or not y:
        return False
    # Only a read long enough to mean something. "को" is "क" once its vowel
    # sign is gone, and "क" is inside "शरीर के" and half the words around it.
    if min(len(x), len(y)) >= MIN_CHARS and (x in y or y in x):
        return True
    return SequenceMatcher(None, x, y, autojunk=False).ratio() >= ALIKE_RATIO


def _times_touch(a, b):
    """Are these two groups on screen at the same time, or near enough?"""
    return (a["first"] <= b["last"] + MERGE_SECONDS
            and b["first"] <= a["last"] + MERGE_SECONDS)


def read_log(reads, cues, fps, erase):
    """Log lines: what OCR read, where, how sure, and what became of it.

    Reads of the same text in about the same place, one after another, are
    one line with the time they stayed on screen. ERASE says the text is
    painted out of the video, LEAVE says it stays on screen; with no
    decision to show, neither word is printed. Lines that are erased or
    matched the speech are always shown; the cap only cuts the rest, which
    on a screen recording is mostly app text.
    """
    groups = text_groups(reads, cues, fps)

    for g in groups:
        reason = erase.get(g["frame"], {}).get(g["box"]) if erase else None
        g["verdict"] = "" if erase is None else (
            f" ERASE {reason}" if reason == "linger" else
            " ERASE" if reason else " LEAVE")
        g["important"] = g["match"] > 0 or g["verdict"].startswith(" ERASE")

    room = MAX_LOG_LINES - sum(g["important"] for g in groups)
    lines = []
    for g in groups:
        if not g["important"]:
            if room <= 0:
                continue
            room -= 1
        xmin, xmax, ymin, ymax = g["box"]
        lines.append(
            f"OCR {g['first']:.2f}-{g['last']:.2f}s y={ymin}-{ymax} x={xmin}-{xmax} "
            f"ocr={g['ocr']:.2f} match={g['match']:.2f}{g['verdict']} \"{g['text']}\""
        )
    if len(groups) > len(lines):
        lines.append(f"OCR ... {len(groups) - len(lines)} more lines not shown")
    return lines
