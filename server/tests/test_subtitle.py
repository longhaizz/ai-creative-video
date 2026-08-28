"""Subtitle text and placement, checked without ffmpeg.

Everything up to the ffmpeg call is plain text work, and that is where the
mistakes are: a line too long to read, a cue that ends before it starts, or
text placed off the bottom of a 720p frame because the position was written
in pixels somewhere.
"""

import shutil
import subprocess

import pytest

from pydantic import ValidationError

from server.schemas import DubParams
from server.steps.subtitle import (
    AUTO_SIZE,
    MAX_CHARS_PER_LINE,
    MAX_LINES_PER_CUE,
    _ass_time,
    burn,
    chars_per_line,
    normalize_cues,
    resolve_font_size,
    split_cue,
    wrap_text_lines,
    write_ass,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not on PATH",
)


# -- wrapping ---------------------------------------------------------------


def test_short_text_stays_on_one_line():
    assert wrap_text_lines("hello there") == ["hello there"]


def test_lines_never_pass_the_limit():
    text = "mot hai ba bon nam sau bay tam chin muoi " * 4
    for line in wrap_text_lines(text):
        assert len(line) <= MAX_CHARS_PER_LINE


def test_it_breaks_on_spaces_not_inside_words():
    lines = wrap_text_lines("aaaa bbbb cccc dddd eeee ffff gggg", max_chars=10)
    for line in lines:
        assert not line.startswith(" ") and not line.endswith(" ")
        for word in line.split():
            assert word in "aaaa bbbb cccc dddd eeee ffff gggg"


def test_one_very_long_word_is_cut():
    """Nothing can wrap it, so it has to be cut rather than overflow."""
    lines = wrap_text_lines("x" * 100, max_chars=10)
    assert lines == ["x" * 10] * 10


def test_empty_text_gives_no_lines():
    assert wrap_text_lines("") == []
    assert wrap_text_lines("   \n  ") == []


# -- splitting one cue into several -----------------------------------------


def test_a_short_cue_keeps_its_own_times():
    out = split_cue({"start": 1.0, "end": 3.0, "text": "hello there"})
    assert out == [{"start": 1.0, "end": 3.0, "text": "hello there"}]


def test_a_long_cue_becomes_several_and_fills_the_same_span():
    cue = {"start": 10.0, "end": 20.0, "text": "mot hai ba bon nam sau bay " * 6}
    out = split_cue(cue)

    assert len(out) > 1
    assert out[0]["start"] == 10.0
    assert out[-1]["end"] == 20.0, "the last part must keep the original end"
    for part in out:
        assert part["end"] > part["start"], "a cue must last some time"
        assert part["text"].count("\n") + 1 <= MAX_LINES_PER_CUE


def test_the_parts_follow_each_other_without_gaps():
    cue = {"start": 0.0, "end": 9.0, "text": "mot hai ba bon nam sau bay " * 6}
    out = split_cue(cue)
    for earlier, later in zip(out, out[1:]):
        assert abs(earlier["end"] - later["start"]) < 0.001


def test_a_longer_part_stays_on_screen_longer():
    cue = {"start": 0.0, "end": 10.0, "text": "a " + "word " * 30}
    out = split_cue(cue)
    lengths = [len(part["text"]) for part in out]
    spans = [part["end"] - part["start"] for part in out]
    assert (lengths[0] > lengths[-1]) == (spans[0] > spans[-1])


def test_empty_cues_disappear():
    assert normalize_cues([{"start": 0, "end": 1, "text": "  "}]) == []


def test_cues_are_one_line():
    cue = {"start": 0.0, "end": 8.0, "text": "mot hai ba bon nam sau bay " * 6}
    for part in split_cue(cue):
        assert "\n" not in part["text"]
        assert part["text"].count("\n") + 1 <= MAX_LINES_PER_CUE


# -- auto size and wrap width -----------------------------------------------


def test_omitted_size_is_56_on_9_16():
    assert resolve_font_size(None, 1920) == AUTO_SIZE


def test_omitted_size_scales_with_height():
    assert resolve_font_size(None, 1080) == round(56 * 1080 / 1920)


def test_a_typed_size_is_not_scaled():
    assert resolve_font_size(56, 1080) == 56
    assert resolve_font_size(40, 1920) == 40


def test_chars_per_line_shrinks_on_a_tall_narrow_frame():
    portrait = chars_per_line(1080, 56)
    landscape = chars_per_line(1920, 32)
    assert portrait < landscape
    assert 20 <= portrait <= 32


def test_schema_omits_size_for_auto():
    assert DubParams().subtitle_size is None
    assert DubParams(subtitle_size="").subtitle_size is None
    assert DubParams(subtitle_size=42).subtitle_size == 42


def test_schema_rejects_a_size_below_eight():
    with pytest.raises(ValidationError):
        DubParams(subtitle_size=7)


def test_schema_default_position_is_three_quarters_down():
    assert DubParams().subtitle_position == 0.75


# -- time format ------------------------------------------------------------


def test_ass_time_uses_centiseconds():
    assert _ass_time(0) == "0:00:00.00"
    assert _ass_time(61.25) == "0:01:01.25"
    assert _ass_time(3661.5) == "1:01:01.50"


def test_a_negative_time_becomes_zero():
    assert _ass_time(-3) == "0:00:00.00"


def test_rounding_up_carries_into_the_seconds():
    assert _ass_time(1.999) == "0:00:02.00"


# -- the .ass file ----------------------------------------------------------


def read_ass(tmp_path, cues, width=1920, height=1080, **kwargs):
    settings = {"font": "Arial", "size": 28, "position": 0.75, **kwargs}
    path = write_ass(cues, tmp_path / "s.ass", width, height, **settings)
    return path.read_text(encoding="utf-8")


def test_the_frame_size_is_written_down(tmp_path):
    """Without PlayRes the player guesses, and the text lands somewhere else."""
    text = read_ass(tmp_path, [{"start": 0, "end": 1, "text": "hi"}], 1280, 720)
    assert "PlayResX: 1280" in text
    assert "PlayResY: 720" in text


def test_position_is_a_share_of_the_height(tmp_path):
    """The same setting must sit in the same place on 720p and on 1080p."""
    small = read_ass(tmp_path, [{"start": 0, "end": 1, "text": "hi"}], 1280, 720)
    large = read_ass(tmp_path, [{"start": 0, "end": 1, "text": "hi"}], 1920, 1080)
    assert "\\pos(640,540)" in small
    assert "\\pos(960,810)" in large


def test_the_box_is_black_text_on_white_with_a_border(tmp_path):
    text = read_ass(tmp_path, [{"start": 0, "end": 1, "text": "hi"}])
    styles = [line for line in text.splitlines() if line.startswith("Style:")]
    assert len(styles) == 2
    box, fill = styles
    assert ",3,10,0,5," in box, "outer black box: BorderStyle 3, outline 10, no shadow"
    assert ",3,8,0,5," in fill, "inner white box: padding 8"
    assert box.split(",")[3] == "&H00000000", "text is black"
    assert "&H00FFFFFF" in fill
    assert text.count("Dialogue:") == 2


def test_two_lines_use_the_ass_line_break(tmp_path):
    """A real newline would end the Dialogue line and lose the second half."""
    text = read_ass(tmp_path, [{"start": 0, "end": 1, "text": "first\nsecond"}])
    assert "first\\Nsecond" in text
    assert "Dialogue: 0,0:00:00.00,0:00:01.00" in text


def test_a_font_name_cannot_break_the_style_line(tmp_path):
    """Commas and colons separate fields in ASS, so they must not survive."""
    text = read_ass(tmp_path, [{"start": 0, "end": 1, "text": "hi"}], font="Ba,d:'font")
    styles = [line for line in text.splitlines() if line.startswith("Style:")]
    assert styles, "the file must name at least one style"
    for style in styles:
        assert style.count(",") == 22, "one comma per field, none from the font name"


def test_an_empty_font_falls_back_to_arial(tmp_path):
    text = read_ass(tmp_path, [{"start": 0, "end": 1, "text": "hi"}], font="")
    assert "Style: Default,Arial," in text


# -- the real ffmpeg call ---------------------------------------------------


def duration_of(path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def audio_codec_of(path) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


@needs_ffmpeg
def test_burning_keeps_the_video_whole(tmp_path):
    """The one test that runs the ffmpeg filter for real.

    The .ass path is pasted into a -vf string, where a colon separates
    options and a quote ends the value. Escaping it wrong breaks the filter,
    and only a real call shows that.
    """
    source = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(source)],
        check=True, capture_output=True,
    )

    out = burn(
        source,
        [{"start": 0.0, "end": 1.5, "text": "dong mot"},
         {"start": 1.5, "end": 3.0, "text": "dong hai"}],
        tmp_path / "out.mp4",
        width=1280, height=720,
    )

    assert out.stat().st_size > 0
    assert abs(duration_of(out) - duration_of(source)) < 0.1
    assert audio_codec_of(out) == "aac", "the audio must be copied, not re-encoded"


def test_subtitles_follow_the_spoken_take_not_the_asr_window(tmp_path):
    import json

    from server.pipeline import _subtitle_cues

    (tmp_path / "spoken_cues.json").write_text(
        json.dumps([{"start": 1.9, "end": 4.0, "text": "I bet you cannot."}]),
        encoding="utf-8",
    )
    asr = [{
        "start": 1.9, "end": 4.4,
        "speech_start": 1.9, "speech_end": 4.4, "text": "ar",
    }]
    out = _subtitle_cues(tmp_path, asr)
    assert out == [{"start": 1.9, "end": 4.0, "text": "I bet you cannot."}]
