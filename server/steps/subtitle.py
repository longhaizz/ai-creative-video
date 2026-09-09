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

# The hook is drawn straight onto the picture: the old hook was painted out
# first, so a box behind the new one would hide the clean frame we paid a
# whole inpainting pass for. A black outline keeps it readable anyway.
HOOK_OUTLINE = "&H00000000"
HOOK_OUTLINE_WIDTH = 3
HOOK_COLOUR = "&H00FFFFFF"

# Where the \pos point sits on the hook: 4 is the middle of its left edge,
# 5 the centre, 6 the middle of the right edge. The x that goes with each
# is the matching edge of the box, so the text grows away from that edge.
HOOK_ALIGNMENTS = {"left": 4, "center": 5, "right": 6}


def ass_colour(value: str | None, fallback: str = HOOK_COLOUR) -> str:
    """Turn #RRGGBB into the &HAABBGGRR that ASS wants."""
    text = (value or "").strip().lstrip("#")
    if len(text) != 6:
        return fallback
    try:
        red, green, blue = (int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return fallback
    return f"&H00{blue:02X}{green:02X}{red:02X}"


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


def _clean_font(font: str) -> str:
    """A font name with a quote, colon or comma would break its ASS line."""
    font = (font or "Arial").replace("'", "").replace(":", "").replace(",", " ").strip()
    return font or "Arial"


def _ass_style(
    name: str,
    font: str,
    size: int,
    fill: str,
    outline: int,
    text_colour: str = TEXT_COLOUR,
    border_style: int = 3,
    alignment: int = ALIGNMENT,
) -> str:
    """One style line. border_style 3 is a filled box, 1 is an outline."""
    return (
        f"Style: {name},{_clean_font(font)},{size},{text_colour},&H000000FF,"
        f"{fill},{fill},0,0,0,0,100,100,0,0,"
        f"{border_style},{outline},{SHADOW},{alignment},0,0,0,1"
    )


def _hook_style(hook: dict | None, height: int) -> str:
    """The Hook style line, or nothing when this job has no hook."""
    if hook is None:
        return ""
    return _ass_style(
        "Hook",
        hook.get("font") or "Noto Sans",
        resolve_font_size(hook.get("size"), height),
        HOOK_OUTLINE,
        HOOK_OUTLINE_WIDTH,
        text_colour=ass_colour(hook.get("colour")),
        border_style=1,
        alignment=HOOK_ALIGNMENTS.get(hook.get("align"), ALIGNMENT),
    )


def hook_lines(hook: dict, width: int, size: int) -> list[str]:
    """The hook, one entry per line on screen.

    A client that says `prewrapped` has already broken the text where it
    breaks, measured with the real font on the picture the user was looking
    at. Nothing here can do better than that -- chars_per_line only knows an
    average letter width -- so those lines are kept as they came, and a
    single line that the client found room for stays a single line.

    Without that flag the text is one blob from an older client, and it is
    wrapped here the way subtitles are.
    """
    text = hook.get("text") or ""
    if hook.get("prewrapped"):
        return [line.strip() for line in text.split("\n") if line.strip()]
    box_width = max(1, int(width * (hook["right"] - hook["left"])))
    return wrap_text_lines(text, chars_per_line(box_width, size))


def hook_dialogue(hook: dict, width: int, height: int) -> str | None:
    """The hook line, placed in the box the client drew.

    It wraps to the width of that box, not to the width of the frame, so
    the new hook stays inside the area the old one was painted out of, and
    it sits against the edge the client asked to line it up with.
    """
    size = resolve_font_size(hook.get("size"), height)
    lines = hook_lines(hook, width, size)
    if not lines:
        return None
    # The x follows the alignment: text laid out from the left edge, from
    # the centre, or back from the right edge of the box that was drawn.
    share = {
        "left": hook["left"],
        "right": hook["right"],
    }.get(hook.get("align"), (hook["left"] + hook["right"]) / 2)
    x = int(round(width * share))
    y = int(round(height * (hook["top"] + hook["bottom"]) / 2))
    end = _ass_time(float(hook.get("end") or 0))
    return (
        f"Dialogue: 2,{_ass_time(0)},{end},Hook,,0,0,0,,"
        f"{{\\pos({x},{y})}}" + hook_body(lines, hook.get("colours"))
    )


def hook_body(lines: list[str], colours=None) -> str:
    """The lines as one ASS field, each in its own colour.

    ASS changes colour mid-text with a \\c tag, so two colours need neither a
    second style nor a second Dialogue -- the whole hook stays one line in
    the file and one pass of the encoder. A line with no colour of its own
    keeps the style's, which is the colour every line had before.
    """
    colours = list(colours or [])
    parts = []
    for index, line in enumerate(lines):
        colour = colours[index] if index < len(colours) else None
        # A \c tag wants &HBBGGRR& -- six digits between the markers, with
        # no alpha byte. ass_colour() writes the style form, &HAABBGGRR.
        tag = f"{{\\c&H{ass_colour(colour)[4:]}&}}" if colour else ""
        parts.append(tag + line)
    return "\\N".join(parts)


def write_ass(
    cues: list[dict],
    out_ass: Path,
    width: int,
    height: int,
    font: str,
    size: int,
    position: float,
    hook: dict | None = None,
) -> Path:
    """Write the subtitle file. `position` is a share of the frame height.

    `hook` is the one headline that sits over the box the client drew, in
    its own font, size and colour. It shares the file so both are drawn in
    the single encode the burn already does.
    """
    font = _clean_font(font)
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
        f"{_ass_style('Default', font, size, BOX_FILL, BOX_PADDING)}\n"
        f"{_hook_style(hook, height)}\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    body = []
    if hook is not None:
        line = hook_dialogue(hook, width, height)
        if line:
            body.append(line)
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


# The slowest ffmpeg in the pipeline: it encodes the whole picture again at
# preset slow, so it can run slower than the video plays. Twice what a two
# minute clip needs, and still short enough to catch a stall the same day.
BURN_TIMEOUT = 600.0


def _burn_once(command: list[str]):
    """Run the burn, and give up rather than hold the queue for ever."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=BURN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise PipelineError(
            f"Burning the subtitles ran longer than {BURN_TIMEOUT:.0f}s "
            "and was stopped"
        ) from None


def burn(
    video: Path,
    cues: list[dict],
    out_path: Path,
    width: int,
    height: int,
    font: str = "Noto Sans",
    size: int | None = None,
    position: float = 0.75,
    hook: dict | None = None,
    ctx=None,
) -> Path:
    """Draw the cues and the hook onto the video for good. Returns out_path."""
    video = Path(video)
    out_path = Path(out_path)
    size = resolve_font_size(size, height)
    cues = normalize_cues(cues, max_chars=chars_per_line(width, size))
    if not cues and hook is None:
        raise PipelineError("There is no text to burn", code="invalid_input")

    if ctx is not None:
        ctx.log(
            f"Burning {len(cues)} subtitles [{font} {size}px at "
            f"{position:.0%} of {width}x{height}]"
        )
        if hook is not None:
            ctx.log(f"Hook: {hook.get('text', '')[:60]}")

    with tempfile.TemporaryDirectory() as work:
        ass = write_ass(
            cues, Path(work) / "burn.ass", width, height, font, size, position,
            hook=hook,
        )
        result = _burn_once([
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
        ])

    if result.returncode != 0:
        raise PipelineError(
            "Burning the subtitles failed: " + (result.stderr or "")[-400:]
        )
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise PipelineError("Burning the subtitles produced no video")
    return out_path
