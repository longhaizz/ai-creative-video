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

import json
import re
import statistics
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
    )},
}

# Seconds a subtitle may show before or after its words are said.
TIME_PAD = 0.75
# Share of the read text that must be found in the speech.
MIN_SCORE = 0.6
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
    # written without spaces between words.
    return re.sub(r"[\W_]+", "", text.lower())


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


def subtitle_band(reads, cues, fps):
    """Where the subtitle line sits, learned from boxes whose text was said.

    reads: frame number (from 1) -> list of (box, text, ocr_score), where a
    box is (xmin, xmax, ymin, ymax). Returns (top, bottom, line_height) in
    pixels, or None when too few frames match the speech to say.
    """
    if fps <= 0:
        return None
    matched, frames = [], set()
    for frame_no, items in reads.items():
        spoken = spoken_at(cues, (frame_no - 1) / fps)
        if not spoken:
            continue
        for box, text, _score in items:
            if contained(text, spoken) >= MIN_SCORE:
                matched.append(box)
                frames.add(frame_no)
    if len(frames) < MIN_MATCHED_FRAMES:
        return None

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
    together = [box for n in frames for box, _text, _score in reads[n]]
    grown = True
    while grown:
        grown = False
        for _, _, ymin, ymax in together:
            if not height / 2 <= ymax - ymin <= height * 2:
                continue
            touches = ymin <= bottom + height / 2 and ymax >= top - height / 2
            if touches and (ymin < top or ymax > bottom):
                top, bottom = min(top, ymin), max(bottom, ymax)
                grown = True
    return top - height / 2, bottom + height / 2, height


def on_the_line(box, band):
    """Does this box sit on the subtitle line, at about its height?

    Twice the height is still allowed: the detector sometimes draws one box
    around a subtitle of two lines.
    """
    top, bottom, height = band
    _, _, ymin, ymax = box
    return top <= ymin and ymax <= bottom and height / 2 <= ymax - ymin <= height * 2


def read_log(reads, cues, fps, band):
    """Log lines: what OCR read, where, how sure, and what became of it.

    Reads of the same text in about the same place, one after another, are
    one line with the time they stayed on screen. With a band, each line
    ends in KEEP or DROP; without one there is nothing to decide.
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
                     "ocr": score, "match": match}
            open_groups[key] = group
            groups.append(group)

    lines = []
    for g in groups[:MAX_LOG_LINES]:
        xmin, xmax, ymin, ymax = g["box"]
        verdict = ""
        if band is not None:
            verdict = " KEEP" if on_the_line(g["box"], band) else " DROP"
        lines.append(
            f"OCR {g['first']:.2f}-{g['last']:.2f}s y={ymin}-{ymax} x={xmin}-{xmax} "
            f"ocr={g['ocr']:.2f} match={g['match']:.2f}{verdict} \"{g['text']}\""
        )
    if len(groups) > MAX_LOG_LINES:
        lines.append(f"OCR ... {len(groups) - MAX_LOG_LINES} more lines not shown")
    return lines
