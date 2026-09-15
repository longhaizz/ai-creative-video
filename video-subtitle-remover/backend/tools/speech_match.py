"""Tell the subtitles apart from other text by what is said at the time.

PATCH (dub server). The detector finds every piece of text in the scan area:
the subtitles, but also logos, prices and calls to action. Subtitles are the
text that is spoken. The dub server runs Whisper first and writes what it
heard to a JSON file. Here the boxes whose text was spoken show where the
subtitle line sits, and the boxes off that line are dropped.

No paddle in this file, so it can be tested without a GPU.
"""

import json
import re
import statistics
from difflib import SequenceMatcher

# Whisper language code -> the PaddleOCR model that reads that script.
# A language missing here turns the filter off, and every box is kept.
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


def load_speech(path):
    """Return {"language": str, "cues": [...]} from the file, or None.

    A broken or empty file must not sink a run that would finish without
    it, so every failure means "no speech" and the filter stays off.
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
        if not cues:
            return None
        return {"language": str(data.get("language") or ""), "cues": cues}
    except Exception:
        return None


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
    return sum(m.size for m in blocks if m.size >= MIN_RUN) / len(a)


def spoken_at(cues, seconds):
    """Everything said around this moment, as one string."""
    return " ".join(
        c["text"] for c in cues
        if c["start"] - TIME_PAD <= seconds <= c["end"] + TIME_PAD
    )


def subtitle_band(reads, cues, fps):
    """Where the subtitle line sits, learned from boxes whose text was said.

    reads: frame number (from 1) -> list of (box, text), where a box is
    (xmin, xmax, ymin, ymax). Returns (top, bottom, line_height) in pixels,
    or None when too few frames match the speech to say.
    """
    if fps <= 0:
        return None
    matched, frames = [], set()
    for frame_no, items in reads.items():
        spoken = spoken_at(cues, (frame_no - 1) / fps)
        if not spoken:
            continue
        for box, text in items:
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
    top = min(b[2] for b in near) - height / 2
    bottom = max(b[3] for b in near) + height / 2
    return top, bottom, height


def on_the_line(box, band):
    """Does this box sit on the subtitle line, at about its height?

    Twice the height is still allowed: the detector sometimes draws one box
    around a subtitle of two lines.
    """
    top, bottom, height = band
    _, _, ymin, ymax = box
    return top <= ymin and ymax <= bottom and height / 2 <= ymax - ymin <= height * 2
