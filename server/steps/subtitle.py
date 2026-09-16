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
    screen: list[dict] | None = None,
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
        f"{_screen_styles(screen, font, size)}"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    body = screen_dialogues(screen or [], width, height)
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
    screen: list[dict] | None = None,
    ctx=None,
) -> Path:
    """Draw the cues and the hook onto the video for good. Returns out_path."""
    video = Path(video)
    out_path = Path(out_path)
    size = resolve_font_size(size, height)
    cues = normalize_cues(cues, max_chars=chars_per_line(width, size))
    if not cues and hook is None and not screen:
        raise PipelineError("There is no text to burn", code="invalid_input")

    if ctx is not None:
        ctx.log(
            f"Burning {len(cues)} subtitles [{font} {size}px at "
            f"{position:.0%} of {width}x{height}]"
        )
        if hook is not None:
            ctx.log(f"Hook: {hook.get('text', '')[:60]}")
        for piece in screen or []:
            ctx.log(describe_screen_piece(piece, width, height))

    with tempfile.TemporaryDirectory() as work:
        ass = write_ass(
            cues, Path(work) / "burn.ass", width, height, font, size, position,
            hook=hook, screen=screen,
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


# -- the text that was printed on the picture -------------------------------

# The OCR box is cut tight around the letters, but a font size counts the
# room above and below them too, so the size that drew those letters is
# bigger than the box is tall.
# ponytail: one ratio for every font; check it against a real video
SCREEN_SIZE_RATIO = 1.3
# How far apart stacked lines of screen text sit, as a share of the size.
SCREEN_LINE_SPACING = 1.2
# How much of the font size one line of letters really fills. A line box
# counts the room above and below the letters as well, and sizing the white
# box by that made it half again as tall as the text it covered.
# ponytail: one number for every font; check it against a real video
SCREEN_INK_SHARE = 0.75
# The white box is drawn tight around the text: no padding. It is there to
# cover the old words and carry the new ones, nothing more, and every pixel
# of it is a pixel of the advert that the viewer cannot see.
SCREEN_PADDING = 0
# How far the text may be shrunk to fit the box the old text sat in, as a
# share of the size that text was drawn at. English is about twice as wide
# as Devanagari at the same height, so without shrinking, a headline spilled
# onto a second line and its white box swallowed the line above it. With no
# floor the other way is just as bad: a two word badge whose translation
# runs to six words would shrink until nobody could read it.
# ponytail: 0.6 keeps a 1280 tall frame readable; measure it on a phone
SCREEN_MIN_SHRINK = 0.6
# Keep the white box off the very edge of the picture.
SCREEN_MARGIN = 8
# OCR only looks at some frames, so the text was already there a moment
# before it was first read, and stayed a moment after the last one.
SCREEN_TIME_PAD = 0.25
# A piece read in a single frame would otherwise be drawn for no time at all.
SCREEN_MIN_SECONDS = 0.5
# Which layer each half of a piece is drawn on. The white rectangle has to
# sit under the text, and ASS draws a higher layer over a lower one.
SCREEN_RECT_LAYER = 0
SCREEN_TEXT_LAYER = 1


def _text_width(text: str, size: int) -> float:
    """Roughly how wide this text is, at this font size."""
    return len(text) * CHAR_WIDTH_EM * max(size, 1)


def screen_layout(piece: dict, width: int, height: int) -> tuple:
    """Where one piece of translated screen text goes and how big it is.

    Returns (lines, size, (x0, y0, box_w, box_h)) in pixels.

    The size comes from the box OCR read the old text in, so a headline
    stays a headline and a badge stays a badge. When the translation is
    wider than that box, the white box grows sideways around the middle of
    the old one rather than the text shrinking: the old text was one line
    at that size on purpose. Only when there is no room left in the frame
    does it wrap.
    """
    xmin, xmax, ymin, ymax = piece["box"]
    old_w, old_h = max(xmax - xmin, 1), max(ymax - ymin, 1)
    size = resolve_font_size(int(round(old_h * SCREEN_SIZE_RATIO)), height)
    text = (piece.get("text") or "").strip()

    # Shrink to fit the footprint the old text had, rather than spread over
    # the picture around it. Only down to SCREEN_MIN_SHRINK: past that the
    # text stops being the design element it replaces.
    if _text_width(text, size) > old_w:
        fits = old_w / (CHAR_WIDTH_EM * max(len(text), 1))
        size = resolve_font_size(
            int(round(max(fits, size * SCREEN_MIN_SHRINK))), height)

    room = max(width - 2 * SCREEN_MARGIN, 1)
    lines = [text]
    if _text_width(text, size) + 2 * SCREEN_PADDING > room:
        # Wrapped to the whole room, not to the 80% a subtitle wraps to:
        # every line that is not needed makes the box taller.
        fits = max(1, int(room / (CHAR_WIDTH_EM * max(size, 1))))
        lines = wrap_text_lines(text, fits) or [text]

    text_w = max(_text_width(line, size) for line in lines)
    box_w = min(max(old_w, int(round(text_w)) + 2 * SCREEN_PADDING), room)
    # The letters themselves, not the line boxes around them.
    text_h = int(round(size * (SCREEN_INK_SHARE
                               + (len(lines) - 1) * SCREEN_LINE_SPACING)))
    box_h = min(max(old_h, text_h + 2 * SCREEN_PADDING), height)

    x0 = _keep_inside((xmin + xmax) // 2 - box_w // 2, box_w, width)
    y0 = _keep_inside((ymin + ymax) // 2 - box_h // 2, box_h, height)
    return lines, size, (x0, y0, box_w, box_h)


def _keep_inside(start: int, length: int, whole: int) -> int:
    """Slide a box of this length back inside the picture."""
    return max(min(start, whole - length - SCREEN_MARGIN), SCREEN_MARGIN)


def screen_times(piece: dict) -> tuple[float, float]:
    """When to show one piece, padded for the frames OCR did not look at."""
    start = max(float(piece.get("start") or 0.0) - SCREEN_TIME_PAD, 0.0)
    end = float(piece.get("end") or 0.0) + SCREEN_TIME_PAD
    return start, max(end, start + SCREEN_MIN_SECONDS)


def screen_dialogues(pieces: list[dict], width: int, height: int) -> list[str]:
    """The ASS lines that cover the old text and write the new one.

    Two lines per piece: a filled rectangle, then the translation on top of
    it. The rectangle is drawn rather than left to the style's own box,
    because the style's box hugs the text, and a translation shorter than
    the original would leave the ends of the old text showing around it.
    """
    body = []
    for piece in pieces:
        text = (piece.get("text") or "").strip()
        if not text:
            continue
        lines, size, (x0, y0, box_w, box_h) = screen_layout(piece, width, height)
        start, end = screen_times(piece)
        start, end = _ass_time(start), _ass_time(end)
        rect = (f"{{\\pos({x0},{y0})\\p1}}"
                f"m 0 0 l {box_w} 0 l {box_w} {box_h} l 0 {box_h}"
                "{\\p0}")
        body.append(f"Dialogue: {SCREEN_RECT_LAYER},{start},{end},ScreenBox,,0,0,0,,{rect}")
        middle = f"{{\\pos({x0 + box_w // 2},{y0 + box_h // 2})\\fs{size}}}"
        body.append(f"Dialogue: {SCREEN_TEXT_LAYER},{start},{end},Screen,,0,0,0,,"
                    + middle + "\\N".join(lines))
    return body


def _screen_styles(pieces: list[dict] | None, font: str, size: int) -> str:
    """The two styles screen text needs, or nothing when there is none.

    ScreenBox is only ever used for the drawn rectangle, so its text colour
    is the fill: a drawing takes its colour from PrimaryColour. Screen is
    the same black text as the subtitles, with no box of its own, because
    the rectangle underneath is already the box.
    """
    if not pieces:
        return ""
    return (
        _ass_style("ScreenBox", font, size, BOX_FILL, 0,
                   text_colour=BOX_FILL, border_style=1, alignment=7) + "\n"
        + _ass_style("Screen", font, size, BOX_FILL, 0,
                     text_colour=TEXT_COLOUR, border_style=1) + "\n"
    )


def describe_screen_piece(piece: dict, width: int, height: int) -> str:
    """One log line saying where a piece of translated text will be drawn.

    It carries the box OCR read the old text in and the box that will be
    painted over it, so a piece that lands in the wrong place, at the wrong
    size or at the wrong moment can be told apart from a piece that was
    read wrongly to begin with.
    """
    text = (piece.get("text") or "").strip()
    if not text:
        return f"Screen text (nothing to draw) {piece.get('box')}"
    lines, size, (x0, y0, box_w, box_h) = screen_layout(piece, width, height)
    start, end = screen_times(piece)
    oxmin, oxmax, oymin, oymax = piece["box"]
    return (
        f"Screen text t={start:.2f}-{end:.2f}s "
        f"read at x={oxmin}-{oxmax} y={oymin}-{oymax} "
        f"draw box x={x0}-{x0 + box_w} y={y0}-{y0 + box_h} ({box_w}x{box_h}) "
        f"centre=({x0 + box_w // 2},{y0 + box_h // 2}) "
        f"size={size}px layer={SCREEN_TEXT_LAYER} lines={len(lines)} "
        f"{text!r}"
    )
