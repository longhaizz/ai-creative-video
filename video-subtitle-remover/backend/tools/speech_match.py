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
    matched = {}
    for frame_no, items in reads.items():
        spoken = spoken_at(cues, (frame_no - 1) / fps)
        if not spoken:
            continue
        boxes = [box for box, text, _score in items if contained(text, spoken) >= MIN_SCORE]
        if boxes:
            matched[frame_no] = boxes
    if len(matched) < MIN_MATCHED_FRAMES:
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


def read_log(reads, cues, fps, erase):
    """Log lines: what OCR read, where, how sure, and what became of it.

    Reads of the same text in about the same place, one after another, are
    one line with the time they stayed on screen. ERASE says the text is
    painted out of the video, LEAVE says it stays on screen; with no
    decision to show, neither word is printed. Lines that are erased or
    matched the speech are always shown; the cap only cuts the rest, which
    on a screen recording is mostly app text.
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
