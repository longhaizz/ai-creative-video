"""Subtitle removal, checked without paddle and without a GPU.

The heavy part runs in another venv, so it cannot be tested here. What can
be tested is everything around it, and that is where the mistakes live: the
share-to-pixel maths and the command line. A wrong scan area paints over the
wrong part of the picture and nothing crashes to tell you.
"""

import pytest

from server.steps.vsr import area_to_pixels, build_command


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
        tmp_path / "in.mp4", tmp_path / "out.mp4", "sttn-det", (648, 1036, 57, 1862)
    )
    assert command[0] == "/opt/venv-vsr/bin/python"
    assert command[1] == "backend/main.py"
    assert "--inpaint-mode" in command
    assert command[command.index("--inpaint-mode") + 1] == "sttn-det"


def test_the_area_is_given_in_the_order_the_tool_wants():
    """The tool reads YMIN YMAX XMIN XMAX. Any other order paints the wrong box."""
    command = build_command("in.mp4", "out.mp4", "sttn-det", (10, 20, 30, 40))
    start = command.index("--subtitle-area-coords")
    assert command[start + 1 : start + 5] == ["10", "20", "30", "40"]


@pytest.mark.parametrize("mode", ["sttn-det", "sttn-auto", "lama", "propainter"])
def test_every_mode_the_client_may_pick_is_passed_through(mode):
    command = build_command("in.mp4", "out.mp4", mode, (1, 2, 3, 4))
    assert command[command.index("--inpaint-mode") + 1] == mode
