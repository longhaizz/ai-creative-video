"""Subtitle removal, checked without paddle and without a GPU.

The heavy part runs in another venv, so it cannot be tested here. What can
be tested is everything around it, and that is where the mistakes live: the
share-to-pixel maths and the command line. A wrong scan area paints over the
wrong part of the picture and nothing crashes to tell you.
"""

import io
import json
from pathlib import Path

import pytest

from server.jobs import PipelineError
from server.steps.vsr import (
    NO_SUBTITLE_EXIT_CODE,
    area_to_pixels,
    boxes_from_pieces,
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


def test_speech_cues_are_passed_only_when_given():
    """The hook pass sends none: nobody says the hook out loud."""
    plain = build_command("in.mp4", "out.mp4", "sttn-det", (1, 2, 3, 4), "boxes.json")
    assert "--speech-cues" not in plain
    heard = build_command(
        "in.mp4", "out.mp4", "sttn-det", (1, 2, 3, 4), "boxes.json", "cues.json"
    )
    assert heard[heard.index("--speech-cues") + 1] == "cues.json"


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


def test_a_filter_line_stuck_to_the_progress_bar_still_reaches_the_log(
        monkeypatch, tmp_path):
    """Seen on a Croatian video: the only line saying why nothing was removed
    came right after a tqdm bar, shared its line, and was thrown away."""
    (tmp_path / "video.mp4").write_bytes(b"v")
    monkeypatch.setattr("server.steps.vsr.probe_size", lambda path: (640, 360))
    process = _FakeProcess(NO_SUBTITLE_EXIT_CODE)
    process.stdout = io.StringIO(
        "Subtitle Finding: 85%|###| 710/833Speech filter: no text model "
        "for language 'xx', nothing removed\n"
    )
    monkeypatch.setattr("server.steps.vsr.subprocess.Popen", lambda *a, **k: process)

    class Ctx:
        logs = []

        def log(self, message):
            self.logs.append(message)

        def check_cancel(self):
            pass

    ctx = Ctx()
    remove_subtitles(tmp_path / "video.mp4", tmp_path / "no_subs.mp4",
                     "sttn-det", 0.6, 0.96, 0.03, 0.97, ctx=ctx)
    assert "Speech filter: no text model for language 'xx', nothing removed" in ctx.logs, ctx.logs


def test_speech_cues_reach_the_tool_as_an_absolute_path(monkeypatch, tmp_path):
    """The tool runs from its own folder, where jobs/... points at nothing.

    This went wrong once: the filter found no file, said nothing, and every
    piece of text in the frame was painted out.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "video.mp4").write_bytes(b"v")
    monkeypatch.setattr("server.steps.vsr.probe_size", lambda path: (640, 360))
    seen = {}

    def fake_popen(command, **kwargs):
        seen["command"] = command
        return _FakeProcess(NO_SUBTITLE_EXIT_CODE)

    monkeypatch.setattr("server.steps.vsr.subprocess.Popen", fake_popen)
    remove_subtitles(
        "video.mp4", "no_subs.mp4", "sttn-det", 0.6, 0.96, 0.03, 0.97,
        speech_cues=Path("jobs/abc/speech_cues.json"),
    )
    command = seen["command"]
    given = Path(command[command.index("--speech-cues") + 1])
    assert given.is_absolute()
    assert given == tmp_path / "jobs" / "abc" / "speech_cues.json"


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


# -- the text that stays on screen ------------------------------------------

from server.steps.vsr import read_screen_text        # noqa: E402


def test_asking_for_screen_text_reads_the_whole_frame():
    """Text to translate sits anywhere, not only where the subtitles are."""
    command = build_command("in.mp4", "out.mp4", "sttn-det", (1, 2, 3, 4),
                            "boxes.json", screen_text="screen.json")
    assert command[command.index("--dump-screen-text") + 1] == "screen.json"
    assert "--scan-all-text" in command
    assert "--detect-only" not in command


def test_detect_only_is_only_sent_when_asked():
    plain = build_command("in.mp4", "out.mp4", "sttn-det", (1, 2, 3, 4),
                          "boxes.json")
    assert "--dump-screen-text" not in plain
    assert "--scan-all-text" not in plain
    looking = build_command("in.mp4", "out.mp4", "sttn-det", (1, 2, 3, 4),
                            "boxes.json", screen_text="s.json", detect_only=True)
    assert "--detect-only" in looking


def test_how_long_text_must_stay_reaches_the_tool():
    command = build_command(
        "in.mp4", "out.mp4", "sttn-det", (1, 2, 3, 4), "boxes.json",
        screen_text="screen.json",
        screen_text_min_seconds=0.0, screen_text_small_min_seconds=0.5)
    assert command[command.index("--screen-text-min-seconds") + 1] == "0.0"
    assert command[command.index("--screen-text-small-min-seconds") + 1] == "0.5"
    plain = build_command("in.mp4", "out.mp4", "sttn-det", (1, 2, 3, 4),
                          "boxes.json")
    assert "--screen-text-min-seconds" not in plain


def test_detect_only_hands_back_the_video_it_was_given(monkeypatch, tmp_path):
    """Nothing was painted, so there is no new file to hand on."""
    video = tmp_path / "video.mp4"
    video.write_bytes(b"v")
    monkeypatch.setattr("server.steps.vsr.probe_size", lambda path: (640, 360))
    monkeypatch.setattr("server.steps.vsr.subprocess.Popen",
                        lambda *a, **k: _FakeProcess(0))
    out, position = remove_subtitles(
        video, tmp_path / "unused.mp4", "sttn-det", 0.6, 0.96, 0.03, 0.97,
        screen_text=tmp_path / "screen.json", detect_only=True,
    )
    assert out == video.resolve()
    assert position is None
    assert not (tmp_path / "unused.mp4").exists()


def test_a_detect_only_crash_still_fails(monkeypatch, tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"v")
    monkeypatch.setattr("server.steps.vsr.probe_size", lambda path: (640, 360))
    monkeypatch.setattr("server.steps.vsr.subprocess.Popen",
                        lambda *a, **k: _FakeProcess(1))
    with pytest.raises(PipelineError):
        remove_subtitles(
            tmp_path / "video.mp4", tmp_path / "unused.mp4", "sttn-det",
            0.6, 0.96, 0.03, 0.97,
            screen_text=tmp_path / "screen.json", detect_only=True,
        )


def test_the_screen_text_is_read_back_from_the_file(tmp_path):
    path = tmp_path / "screen.json"
    path.write_text(
        '[{"text": "SALE", "box": [1, 2, 3, 4], "start": 0.0, "end": 1.0}]',
        encoding="utf-8")
    assert read_screen_text(path)[0]["text"] == "SALE"


def test_a_missing_or_broken_screen_text_file_is_no_text(tmp_path):
    """The tool leaves early on a clean video; that must not be a failure."""
    assert read_screen_text(tmp_path / "never_written.json") == []
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert read_screen_text(broken) == []


def test_given_boxes_reach_the_tool(tmp_path):
    command = build_command(
        "in.mp4", "out.mp4", "lama", (0, 100, 0, 100), "boxes.json",
        inpaint_boxes=tmp_path / "screen_inpaint.json")
    assert command[command.index("--inpaint-mode") + 1] == "lama"
    assert command[command.index("--inpaint-boxes") + 1].endswith(
        "screen_inpaint.json")
    plain = build_command("in.mp4", "out.mp4", "sttn-det", (1, 2, 3, 4),
                          "boxes.json")
    assert "--inpaint-boxes" not in plain


def test_a_piece_covers_every_frame_it_will_be_drawn_on():
    """Burn widens by one OCR step; inpaint has to cover that same span."""
    pieces = [{"text": "SALE", "box": [10, 40, 20, 50],
               "start": 1.0, "end": 2.0, "step": 0.1}]
    boxes = boxes_from_pieces(pieces, fps=10, frame_count=100)
    # 0.9s → frame 10, 2.1s → frame 22, 1-indexed.
    assert min(boxes) == 10
    assert max(boxes) == 22
    assert boxes[10] == [(10, 40, 20, 50)]


def test_two_pieces_on_the_same_frame_keep_both_boxes():
    pieces = [
        {"text": "A", "box": [0, 10, 0, 10], "start": 0.0, "end": 1.0, "step": 0},
        {"text": "B", "box": [20, 30, 20, 30], "start": 0.0, "end": 1.0, "step": 0},
    ]
    boxes = boxes_from_pieces(pieces, fps=1, frame_count=2)
    assert (0, 10, 0, 10) in boxes[1]
    assert (20, 30, 20, 30) in boxes[1]

