"""Guessing how long a line takes before speaking it.

The guess decides which of the three translated lines a block gets, so a
bad guess is heard as either silence over a talking head or a line that
runs into the next one. It also has to survive a missing or broken history
file without taking a dub job down with it.
"""

import csv

from server.steps import duration


def test_vietnamese_counts_one_syllable_per_word():
    assert duration.syllables("một hai ba bốn", "vi") == 4


def test_other_languages_count_vowel_runs():
    assert duration.syllables("hello", "en") == 2, "hel-lo"


def test_pauses_are_counted_separately_from_words():
    assert duration.features("Chào bạn, khỏe không?", "vi") == (4, 1, 1, 0)


def test_a_longer_line_is_guessed_longer():
    short = duration.predict("một hai ba", "vi")
    long = duration.predict("một hai ba bốn năm sáu bảy", "vi")
    assert long > short


def test_a_fit_recovers_the_weights_it_was_given():
    rows = [(n, 0, 1, 0, 0.2 * n + 0.8) for n in range(5, 45)]
    coef = duration.fit(rows)
    assert coef is not None
    assert abs(coef[0] - 0.2) < 0.01, coef


def test_too_little_history_is_not_fitted():
    """Five takes say more about the lines than about the voice."""
    rows = [(n, 0, 1, 0, 0.2 * n + 0.8) for n in range(5, 10)]
    assert duration.fit(rows) is None


def test_history_survives_a_restart(tmp_path):
    path = tmp_path / "nested" / "duration.csv"
    model = duration.Model(path)
    assert not model.fitted, "a fresh install starts on the default weights"
    for n in range(5, 45):
        model.record("vi", " ".join(["từ"] * n) + ".", 0.2 * n + 0.8)
    assert model.refit()

    again = duration.Model(path)
    assert again.fitted
    assert again.coef == model.coef


def test_a_broken_history_file_is_ignored(tmp_path):
    """A statistics file must never be able to fail a dub job."""
    path = tmp_path / "duration.csv"
    path.write_text("nonsense\n1,2\n", encoding="utf-8")
    model = duration.Model(path)
    assert model.coef == duration.DEFAULT_COEF
    model.record("vi", "một hai ba.", 1.0)  # still usable


def test_the_word_budget_is_the_guess_read_backwards():
    model = duration.Model()
    for seconds in (1.5, 3.0, 6.0):
        words = model.words_for(seconds, "vi")
        guess = model.seconds(" ".join(["từ"] * words) + ".", "vi")
        assert abs(guess - seconds) < 0.35, (seconds, words, guess)


def test_fewer_english_words_fit_in_the_same_room():
    model = duration.Model()
    assert model.words_for(3.0, "en") < model.words_for(3.0, "vi")


def test_the_voice_speed_walks_towards_the_measurements():
    speed = duration.Speed()
    for _ in range(10):
        speed.observe(2.0, 3.0)  # this voice runs half again as fast
    assert 1.35 < speed.value < 1.55


def test_a_runaway_take_does_not_move_the_voice_speed():
    speed = duration.Speed()
    for _ in range(10):
        speed.observe(2.0, 3.0)
    before = speed.value
    speed.observe(2.0, 40.0)
    assert speed.value == before


def test_a_row_written_before_the_text_column_still_counts(tmp_path):
    """290 takes were recorded before the line itself was kept.

    Dropping them on the day a column is added would throw away the only
    history there is.
    """
    path = tmp_path / "duration.csv"
    path.write_text(
        "lang,syllables,commas,stops,digits,seconds\n"
        "vi,24,3,1,0,6.24\n",
        encoding="utf-8",
    )
    model = duration.Model(path)
    assert model.rows == [(24.0, 3.0, 1.0, 0.0, 6.24)]


def test_the_line_and_the_voice_speed_are_written_down(tmp_path):
    """The text is the data. The four features are one reading of it."""
    path = tmp_path / "duration.csv"
    model = duration.Model(path)
    model.record("vi", "câu này dài vừa phải.", 2.5, speed=1.22)

    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == duration.HEADER
    assert rows[1][-1] == "câu này dài vừa phải."
    assert rows[1][-2] == "1.22"


def test_an_old_header_is_brought_up_to_date(tmp_path):
    """Rows grew two columns; a short header misreads the whole file."""
    path = tmp_path / "duration.csv"
    path.write_text(
        "lang,syllables,commas,stops,digits,seconds\n"
        "vi,24,3,1,0,6.24\n",
        encoding="utf-8",
    )
    duration.Model(path)
    first = path.read_text(encoding="utf-8").splitlines()[0]
    assert first == ",".join(duration.HEADER)
    assert "vi,24,3,1,0,6.24" in path.read_text(encoding="utf-8"), "rows kept"
