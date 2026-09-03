"""Guess how long a line will take to say, before saying it.

VoxCPM has no duration predictor to borrow. The length of a take is decided
by the same autoregressive loop that makes the audio, so asking it what it
would produce costs exactly as much as producing it.

So this is the cheap stand-in: a straight line through what past takes
really measured.

    seconds = a*syllables + b*commas + c*full_stops + d*digits + e

The features are the ones that hold across voices — how much there is to
say, and how many places a speaker stops. How fast a given voice runs is
deliberately NOT in here: a cloned voice is different in every video, and
one video is not enough to learn a new set of weights from. That part is a
single multiplier the caller measures while the job runs (see Speed).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# Fitted from nothing: a first guess that is close enough to work on the
# very first video, before any history exists.
DEFAULT_COEF = (0.21, 0.12, 0.28, 0.30, 0.20)
# Below this many rows a fit says more about the noise than about the voice.
MIN_SAMPLES = 30
# The columns of the history file. Everything after `seconds` is there to
# train something better later, not for the fit below.
HEADER = ["lang", "syllables", "commas", "stops", "digits", "seconds",
          "speed", "text"]
# Languages that write one syllable per space-separated word.
SYLLABIC = ("vi", "zh", "ja", "th")
# Everywhere else, a word is about this many syllables.
SYLLABLES_PER_WORD = 1.5

_VOWELS = re.compile(r"[aeiouyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩị"
                     r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+", re.IGNORECASE)
_COMMA = re.compile(r"[,;:—–]")
_STOP = re.compile(r"[.!?…。！？]")
_DIGIT = re.compile(r"\d")


def syllables(text: str, lang: str = "") -> int:
    """How many syllables the line has, near enough.

    Vietnamese writes every syllable as its own word, so words are the
    right unit there. Elsewhere a syllable is a run of vowels.
    """
    words = (text or "").split()
    if not words:
        return 0
    if (lang or "").lower()[:2] in SYLLABIC:
        return len(words)
    return sum(max(len(_VOWELS.findall(word)), 1) for word in words)


def _syllables_per_word(lang: str) -> float:
    return 1.0 if (lang or "").lower()[:2] in SYLLABIC else SYLLABLES_PER_WORD


def features(text: str, lang: str = "") -> tuple:
    """The row this text turns into: (syllables, commas, stops, digits)."""
    text = text or ""
    return (
        syllables(text, lang),
        len(_COMMA.findall(text)),
        len(_STOP.findall(text)),
        len(_DIGIT.findall(text)),
    )


def predict(text: str, lang: str = "", coef=DEFAULT_COEF) -> float:
    """Seconds this line should take, for an average voice."""
    row = features(text, lang)
    seconds = coef[4] + sum(value * weight for value, weight in zip(row, coef))
    return max(seconds, 0.2)


def fit(rows) -> tuple | None:
    """Least squares over (features..., seconds) rows. None if too few."""
    rows = [row for row in rows if row and row[-1] > 0.2]
    if len(rows) < MIN_SAMPLES:
        return None
    import numpy

    matrix = numpy.array([[*row[:4], 1.0] for row in rows], dtype=float)
    target = numpy.array([row[4] for row in rows], dtype=float)
    solution, *_ = numpy.linalg.lstsq(matrix, target, rcond=None)
    coef = tuple(float(value) for value in solution)
    if coef[0] <= 0 or not all(abs(value) < 10 for value in coef):
        return None  # a fit this odd is a broken history, not a voice
    return coef


class Model:
    """The fitted line, plus the history file it came from.

    Nothing here may raise: a missing, empty or corrupt history file only
    means the default weights are used. A dub job must never fail because
    of a statistics file.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.rows: list[tuple] = []
        self.coef = DEFAULT_COEF
        self.fitted = False
        self._read()
        self.refit()

    def seconds(self, text: str, lang: str = "") -> float:
        return predict(text, lang, self.coef)

    def words_for(self, seconds: float, lang: str = "",
                  speed: float = 1.0) -> int:
        """How many words fit in `seconds`. This is the prompt's budget.

        The fitted line read backwards: take off what a sentence costs
        before its first word, then divide by what one word costs.
        """
        rate = max(speed, 0.1)
        per_word = max(self.coef[0], 0.01) * rate * _syllables_per_word(lang)
        overhead = (self.coef[4] + self.coef[2]) * rate
        return max(int((float(seconds) - overhead) / per_word), 2)

    def record(self, lang: str, text: str, measured: float,
               speed: float = 1.0) -> None:
        """Keep one (line, how long it really took) pair for the next fit.

        The line itself is written down, not only the four numbers this fit
        happens to use. Without it the history can never train anything but
        this one model: a better feature thought of next month cannot be
        computed backwards from "24 syllables, 3 commas". The text is the
        data; the features are one reading of it.

        `speed` is how fast the voice of that job ran. It is what separates
        "this line is long" from "this voice is slow", and the two look
        identical in a column of seconds.
        """
        if measured <= 0.2:
            return
        row = (*features(text, lang), float(measured))
        self.rows.append(row)
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(HEADER)
                writer.writerow([lang, *row, round(float(speed), 3), text])
        except OSError:
            self.path = None  # stop trying; the job carries on regardless

    def refit(self) -> bool:
        coef = fit(self.rows)
        if coef is None:
            return False
        self.coef = coef
        self.fitted = True
        return True

    def _read(self) -> None:
        """Load the history. Rows written before the text column still count.

        Only the first six fields are read, so a row from any version fits:
        the older ones simply stop there.
        """
        if self.path is None or not self.path.is_file():
            return
        try:
            with self.path.open(newline="", encoding="utf-8") as handle:
                for line in csv.reader(handle):
                    if len(line) < 6 or line[0] == "lang":
                        continue
                    try:
                        self.rows.append(tuple(float(v) for v in line[1:6]))
                    except ValueError:
                        continue
        except OSError:
            return
        self._fix_header()

    def _fix_header(self) -> None:
        """Bring an old file's header line up to date, once.

        The rows below it grew two columns. Leaving the header short is how
        a spreadsheet ends up showing seconds under the heading for digits.
        """
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines(True)
            if not lines or not lines[0].startswith("lang,"):
                return
            wanted = ",".join(HEADER)
            if lines[0].strip() == wanted:
                return
            lines[0] = wanted + "\n"
            self.path.write_text("".join(lines), encoding="utf-8")
        except OSError:
            return


class Speed:
    """How fast this one voice runs against the model. One number.

    A cloned voice can be half again as fast as the average. The shape of
    the line — where it pauses, how much there is — comes from the Model;
    this is the only thing measured per job.
    """

    def __init__(self, value: float = 1.0):
        self.value = float(value)
        self.samples = 0

    def observe(self, predicted: float, measured: float) -> None:
        if predicted <= 0.2 or measured <= 0.2:
            return
        ratio = measured / predicted
        if not 0.25 < ratio < 4.0:
            return  # a babbling take says nothing about the voice
        self.samples += 1
        weight = 1.0 / min(self.samples + 1, 8)
        self.value += (ratio - self.value) * weight

    def seconds(self, model: Model, text: str, lang: str = "") -> float:
        return model.seconds(text, lang) * self.value


def load(path=None) -> Model:
    return Model(path)


def _selfcheck():
    import tempfile

    assert syllables("một hai ba", "vi") == 3
    assert syllables("hello there", "en") == 4, "vowel runs, near enough"
    assert features("Chào bạn, khỏe không?", "vi") == (4, 1, 1, 0)

    # A longer line is predicted to take longer. That is the whole job.
    assert predict("một hai ba", "vi") < predict("một hai ba bốn năm", "vi")

    # Fitting recovers weights from clean data.
    rows = []
    for n in range(5, 45):
        rows.append((n, 0, 1, 0, 0.2 * n + 0.3 + 0.5))
    coef = fit(rows)
    assert coef is not None
    assert abs(coef[0] - 0.2) < 0.01, coef

    # Too few rows: keep the default rather than trust noise.
    assert fit(rows[:5]) is None

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d" / "duration.csv"
        model = Model(path)
        assert model.coef == DEFAULT_COEF and not model.fitted
        for n in range(5, 45):
            model.record("vi", " ".join(["từ"] * n) + ".", 0.2 * n + 0.8)
        assert model.refit(), "enough history must give a fit"
        assert abs(model.seconds(" ".join(["từ"] * 20) + ".", "vi")
                   - (0.2 * 20 + 0.8)) < 0.2

        # A second model reads the same history back.
        again = Model(path)
        assert again.fitted, "history must survive a restart"

    # A broken file is not an error, only a reason to use the defaults.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "duration.csv"
        path.write_text("nonsense\n1,2\n", encoding="utf-8")
        assert Model(path).coef == DEFAULT_COEF

    # The word budget is the prediction read backwards: put that many words
    # in and the model says about the seconds we asked for.
    model = Model()
    for seconds in (1.5, 3.0, 6.0):
        words = model.words_for(seconds, "vi")
        guess = model.seconds(" ".join(["từ"] * words) + ".", "vi")
        assert abs(guess - seconds) < 0.35, (seconds, words, guess)
    assert model.words_for(3.0, "en") < model.words_for(3.0, "vi"),         "English words hold more syllables, so fewer of them fit"

    # The per-voice multiplier walks towards the truth and ignores babble.
    speed = Speed()
    for _ in range(10):
        speed.observe(2.0, 3.0)
    assert 1.35 < speed.value < 1.55, speed.value
    before = speed.value
    speed.observe(2.0, 40.0)
    assert speed.value == before, "a runaway take must not move the speed"

    print("duration.py self-check OK")


if __name__ == "__main__":
    _selfcheck()
