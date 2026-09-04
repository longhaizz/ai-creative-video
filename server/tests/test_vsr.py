"""Subtitle removal, checked without paddle and without a GPU.

The heavy part runs in another venv, so it cannot be tested here. What can
be tested is everything around it, and that is where the mistakes live: the
share-to-pixel maths and the command line. A wrong scan area paints over the
wrong part of the picture and nothing crashes to tell you.
"""

import io
import json

import pytest

from server.jobs import PipelineError
from server.steps.vsr import (
    NO_SUBTITLE_EXIT_CODE,
    area_to_pixels,
    build_command,
    remove_subtitles,
    subtitle_position,
)


# -- share of the frame to pixels -------------------------------------------


def test_default_area_is_the_lower_band():
    """The defaults must cover where ad subtitles sit, on any size."""
    ymin, ymax, xmin, xmax = area_to_pixels(1920, 1080, 0.60, 0.96, 0.03, 0.97)
    assert (ymin, ymax) == (648, 1036)
    assert (xmin, xmax) == (57, 1862)


def test_the_same_shares_follow_the_video_size():
    """The box must cover the same part of the picture at any size.

    Not exactly double: int() cuts the decimals, so 0.96 of 360 gives 345
    and 0.96 of 720 gives 691, not 690. One pixel of drift is fine here;
    what matters is that the band does not move.
    """
    small = area_to_pixels(640, 360, 0.60, 0.96, 0.03, 0.97)
    large = area_to_pixels(1280, 720, 0.60, 0.96, 0.03, 0.97)
    for near, far in zip(small, large):
        assert abs(near * 2 - far) <= 1


def test_the_whole_frame():
    assert area_to_pixels(100, 50, 0.0, 1.0, 0.0, 1.0) == (0, 50, 0, 100)


def test_a_flat_area_still_has_one_row_and_one_column():
    """Rounding can flatten a thin band, and the tool needs something to scan."""
    ymin, ymax, xmin, xmax = area_to_pixels(100, 100, 0.5, 0.5001, 0.5, 0.5001)
    assert ymax > ymin
    assert xmax > xmin


# -- the command line -------------------------------------------------------


def test_command_matches_the_tool(monkeypatch, tmp_path):
    monkeypatch.setattr("server.config.VSR_PYTHON", "/opt/venv-vsr/bin/python")
    command = build_command(
        tmp_path / "in.mp4", tmp_path / "out.mp4", "sttn-det",
        (648, 1036, 57, 1862), tmp_path / "boxes.json",
    )
    assert command[0] == "/opt/venv-vsr/bin/python"
    assert command[1] == "backend/main.py"
    assert "--inpaint-mode" in command
    assert command[command.index("--inpaint-mode") + 1] == "sttn-det"


def test_the_area_is_given_in_the_order_the_tool_wants():
    """The tool reads YMIN YMAX XMIN XMAX. Any other order paints the wrong box."""
    command = build_command(
        "in.mp4", "out.mp4", "sttn-det", (10, 20, 30, 40), "boxes.json"
    )
    start = command.index("--subtitle-area-coords")
    assert command[start + 1 : start + 5] == ["10", "20", "30", "40"]


@pytest.mark.parametrize("mode", ["sttn-det", "sttn-auto", "lama", "propainter"])
def test_every_mode_the_client_may_pick_is_passed_through(mode):
    command = build_command("in.mp4", "out.mp4", mode, (1, 2, 3, 4), "boxes.json")
    assert command[command.index("--inpaint-mode") + 1] == mode


# -- a video with no subtitles in it ----------------------------------------


class _FakeProcess:
    """A finished subprocess that printed one line and left with `code`."""

    def __init__(self, code):
        self.returncode = code
        self.stdout = io.StringIO("Subtitle Finding: 100%\n")

    def wait(self):
        return self.returncode


def _run_with_exit_code(code, monkeypatch, tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"not really a video")
    monkeypatch.setattr("server.steps.vsr.probe_size", lambda path: (640, 360))
    monkeypatch.setattr(
        "server.steps.vsr.subprocess.Popen", lambda *a, **k: _FakeProcess(code)
    )
    return remove_subtitles(
        video, tmp_path / "no_subs.mp4", "sttn-det", 0.6, 0.96, 0.03, 0.97
    )


def test_no_subtitles_is_not_a_failure(monkeypatch, tmp_path):
    """A clean video must not turn the whole job red.

    The tool leaves with NO_SUBTITLE_EXIT_CODE and writes no file. The step
    hands back the video it was given, so the pipeline carries on with it.
    """
    video, position = _run_with_exit_code(NO_SUBTITLE_EXIT_CODE, monkeypatch, tmp_path)
    assert video == (tmp_path / "video.mp4").resolve()
    assert position is None


def test_a_real_crash_still_fails(monkeypatch, tmp_path):
    with pytest.raises(PipelineError):
        _run_with_exit_code(1, monkeypatch, tmp_path)


# -- where the old subtitles sat --------------------------------------------


def _boxes(tmp_path, frames):
    """Write a boxes file the way the tool writes it: frame no -> boxes."""
    path = tmp_path / "sub_boxes.json"
    path.write_text(json.dumps(frames), encoding="utf-8")
    return path


def test_one_block_of_text_gives_its_middle(tmp_path):
    """Boxes from y=800 to y=860 sit at 830, which is 0.83 of a 1000 frame."""
    frames = {str(n): [[100, 900, 800, 860]] for n in range(1, 50)}
    assert subtitle_position(_boxes(tmp_path, frames), 1000) == pytest.approx(0.83)


def test_a_logo_in_the_band_does_not_drag_the_answer(tmp_path):
    """A second block far above must not pull the result into the gap.

    Every frame holds the subtitles at y=830 and a logo at y=630. Their
    plain middle is 730 -- empty picture between the two. Dropping what
    sits far from that middle leaves the block with more boxes in it.
    """
    frames = {
        str(n): [[100, 900, 800, 860], [100, 900, 800, 860], [200, 400, 600, 660]]
        for n in range(1, 50)
    }
    assert subtitle_position(_boxes(tmp_path, frames), 1000) == pytest.approx(0.83)


def test_one_bad_frame_does_not_move_the_answer(tmp_path):
    """A single frame where the detector caught the whole band is outvoted."""
    frames = {str(n): [[100, 900, 800, 860]] for n in range(1, 50)}
    frames["50"] = [[0, 1000, 100, 900]]
    assert subtitle_position(_boxes(tmp_path, frames), 1000) == pytest.approx(0.83)


def test_no_boxes_means_no_position(tmp_path):
    assert subtitle_position(_boxes(tmp_path, {}), 1000) is None
    assert subtitle_position(_boxes(tmp_path, {"1": []}), 1000) is None


def test_no_file_means_no_position(tmp_path):
    """The mode ran no detection, or the run stopped before writing."""
    assert subtitle_position(tmp_path / "never_written.json", 1000) is None
