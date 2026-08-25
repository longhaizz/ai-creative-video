"""Burn the translated subtitles into the finished video.

Ported from spy-ads subtitle_api.py, with two changes:

* It works on the cue list the pipeline already holds. The desktop tool went
  through an .srt file on disk because its steps were separate programs;
  here the cues come straight from the translation step.
* Only font, size and height on screen come from the client. Colour and the
  box are fixed: black text on a white rectangle with a thin black border.

Subtitles are placed with an ASS \\pos tag instead of ffmpeg margins,
because a share of the frame height ("75% down") behaves the same on 720p
and 1080p, while a margin in pixels does not.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from server import config
from server.jobs import PipelineError

# Fallback when wrap is called without a frame size. burn() always replaces
# this with chars_per_line(width, size).
MAX_CHARS_PER_LINE = 32
MAX_LINES_PER_CUE = 1

# 56px on a 1920-tall 9:16 frame. Other heights scale when size is omitted.
AUTO_SIZE = 56
AUTO_SIZE_HEIGHT = 1920
WRAP_WIDTH_RATIO = 0.80
CHAR_WIDTH_EM = 0.55

# Black on white. ASS BorderStyle 3 treats OutlineColour as the box fill in
# some renderers and BackColour in others, so both are set the same.
# A second, slightly larger black box behind the white one is the 2px border:
# ASS has no real rounded-rect stroke.
TEXT_COLOUR = "&H00000000"
BOX_FILL = "&H00FFFFFF"
BOX_BORDER = "&H00000000"
BOX_PADDING = 8
BOX_BORDER_WIDTH = 2
SHADOW = 0
ALIGNMENT = 5  # 5 means the \pos point is the middle of the text


def resolve_font_size(size: int | None, height: int) -> int:
    """None → scale from 56px at 1920 tall. A number is used as-is."""
    if size is None:
        size = round(AUTO_SIZE * max(height, 1) / AUTO_SIZE_HEIGHT)
    return max(8, min(200, int(size)))


def chars_per_line(width: int, size: int) -> int:
    """How many characters fit in ~80% of the frame at this font size."""
    return max(1, int(width * WRAP_WIDTH_RATIO / (CHAR_WIDTH_EM * max(size, 1))))


def wrap_text_lines(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> list[str]:
    """Break text into lines of at most max_chars, on spaces where possible."""
    text = " ".join((text or "").replace("\n", " ").split())
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        trial = f"{current} {word}".strip() if current else word
        if len(trial) <= max_chars:
            current = trial
            continue
        if current:
            lines.append(current)
        # A single word longer than the line has to be cut somewhere.
        while len(word) > max_chars:
            lines.append(word[:max_chars])
            word = word[max_chars:]
        current = word
    if current:
        lines.append(current)
    return lines


def split_cue(
    cue: dict,
    max_chars: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES_PER_CUE,
) -> list[dict]:
    """Turn one long cue into several short ones.

    The time of the original cue is shared out by text length, so a long
    part stays on screen longer than a short one.
    """
    lines = wrap_text_lines(cue.get("text") or "", max_chars)
    if not lines:
        return []
    chunks = [
        "\n".join(lines[i : i + max_lines]) for i in range(0, len(lines), max_lines)
    ]
    start_time = float(cue["start"])
    end_time = float(cue["end"])
    if len(chunks) == 1:
        return [{"start": start_time, "end": end_time, "text": chunks[0]}]

    total_chars = sum(len(chunk) for chunk in chunks) or 1
    span = max(0.05, end_time - start_time)
    out: list[dict] = []
    used = 0.0
    for index, chunk in enumerate(chunks):
        share = span * (len(chunk) / total_chars)
        start = start_time + used
        # The last part keeps the original end, so rounding never leaves a gap.
        end = end_time if index == len(chunks) - 1 else start + share
        if end <= start:
            end = start + 0.05
        used += share
        out.append({"start": start, "end": end, "text": chunk})
    return out


def normalize_cues(
    cues: list[dict],
    max_chars: int = MAX_CHARS_PER_LINE,
    max_lines: int = MAX_LINES_PER_CUE,
) -> list[dict]:
    out: list[dict] = []
    for cue in cues:
        out.extend(split_cue(cue, max_chars, max_lines))
    return out


def _ass_time(seconds: float) -> str:
    """ASS wants h:mm:ss.cc, with centiseconds and no leading zero on hours."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        whole += 1
        centis = 0
    return f"{hours}:{minutes:02d}:{whole:02d}.{centis:02d}"


def _ass_style(name: str, font: str, size: int, fill: str, outline: int) -> str:
    return (
        f"Style: {name},{font},{size},{TEXT_COLOUR},&H000000FF,"
        f"{fill},{fill},0,0,0,0,100,100,0,0,"
        f"3,{outline},{SHADOW},{ALIGNMENT},0,0,0,1"
    )


def write_ass(
    cues: list[dict],
    out_ass: Path,
    width: int,
    height: int,
    font: str,
    size: int,
    position: float,
) -> Path:
    """Write the subtitle file. `position` is a share of the frame height."""
    # A font name with a quote, colon or comma would break the ASS line it
    # sits on, so those characters go.
    font = (font or "Arial").replace("'", "").replace(":", "").replace(",", " ").strip()
    font = font or "Arial"
    x = width // 2
    y = int(round(height * position))
    border = BOX_PADDING + BOX_BORDER_WIDTH

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"{_ass_style('Box', font, size, BOX_BORDER, border)}\n"
        f"{_ass_style('Default', font, size, BOX_FILL, BOX_PADDING)}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    body = []
    for cue in cues:
        text = (cue.get("text") or "").strip().replace("\n", "\\N")
        if not text:
            continue
        start = _ass_time(cue["start"])
        end = _ass_time(cue["end"])
        pos = f"{{\\pos({x},{y})}}{text}"
        body.append(f"Dialogue: 0,{start},{end},Box,,0,0,0,,{pos}")
        body.append(f"Dialogue: 1,{start},{end},Default,,0,0,0,,{pos}")

    out_ass = Path(out_ass)
    out_ass.parent.mkdir(parents=True, exist_ok=True)
    out_ass.write_text(header + "\n".join(body) + "\n", encoding="utf-8")
    return out_ass


def _filter_path(ass: Path) -> str:
    """Escape the .ass path for the ffmpeg filter string.

    Inside -vf the colon separates options and the quote ends the value, so
    a Windows path like C:\\tmp\\a.ass would cut the filter in half.
    """
    return ass.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")


def burn(
    video: Path,
    cues: list[dict],
    out_path: Path,
    width: int,
    height: int,
    font: str = "Noto Sans",
    size: int | None = None,
    position: float = 0.75,
    ctx=None,
) -> Path:
    """Draw the cues onto the video for good. Returns out_path."""
    video = Path(video)
    out_path = Path(out_path)
    size = resolve_font_size(size, height)
    cues = normalize_cues(cues, max_chars=chars_per_line(width, size))
    if not cues:
        raise PipelineError("There is no text to burn", code="invalid_input")

    if ctx is not None:
        ctx.log(
            f"Burning {len(cues)} subtitles [{font} {size}px at "
            f"{position:.0%} of {width}x{height}]"
        )

    with tempfile.TemporaryDirectory() as work:
        ass = write_ass(
            cues, Path(work) / "burn.ass", width, height, font, size, position
        )
        result = subprocess.run(
            [
                config.FFMPEG_BIN, "-y", "-loglevel", "error",
                "-i", str(video),
                "-vf", f"ass='{_filter_path(ass)}'",
                # Burning the subtitles means the picture is encoded again,
                # and this is the last encode in the pipeline. Left to
                # itself ffmpeg picks CRF 23 here and undoes the detail
                # LatentSync just made.
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "14",
                "-pix_fmt", "yuv420p",
                # The audio is already final by this point, so copy it
                # instead of encoding it a second time.
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        raise PipelineError(
            "Burning the subtitles failed: " + (result.stderr or "")[-400:]
        )
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise PipelineError("Burning the subtitles produced no video")
    return out_path
