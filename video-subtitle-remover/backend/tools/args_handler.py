import argparse
from enum import Enum

from .constant import InpaintMode

def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover Command Line Tool"
    )
    parser.add_argument(
        "--input", "-i", required=True, type=str,
        help="Input video file path"
    )
    parser.add_argument(
        "--output", "-o", required=False, type=str, default=None,
        help="Output video file path (optional)"
    )
    parser.add_argument(
        "--subtitle-area-coords", "-c", action="append", nargs=4, type=int, metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        help="Subtitle area coordinates (ymin ymax xmin xmax). Can be specified multiple times for multiple areas."
    )
    parser.add_argument(
        "--inpaint-mode", type=str, default="sttn-auto",
        choices=[mode.name.lower().replace('_','-') for mode in InpaintMode],
        help="Inpaint mode, default is sttn-auto"
    )
    # PATCH (dub server). The boxes are found anyway to build the mask; the
    # dub server reads them back to put the new subtitles where the old ones
    # were, instead of guessing a height. Off unless asked for.
    parser.add_argument(
        "--dump-boxes", type=str, default=None,
        help="Write the detected subtitle boxes to this JSON file"
    )
    # PATCH (dub server). What Whisper heard, written by the dub server. The
    # text in each box is read and compared with it, to keep the subtitles
    # and drop the logos and prices in the same band. Off unless given.
    parser.add_argument(
        "--speech-cues", type=str, default=None,
        help="JSON file with the language and the spoken cues"
    )
    # PATCH (dub server). Where the text that stays on screen is written, for
    # the translate step. Off unless asked for.
    parser.add_argument(
        "--dump-screen-text", type=str, default=None,
        help="Write the text that is left on screen to this JSON file"
    )
    # PATCH (dub server). Look for text in the whole frame, not only in the
    # subtitle area. What gets painted over is unchanged.
    parser.add_argument(
        "--scan-all-text", action="store_true",
        help="Read text anywhere in the frame, not only in the subtitle area"
    )
    # PATCH (dub server). Read the text and stop, painting over nothing. For
    # a job that wants the text translated but the subtitles left alone.
    parser.add_argument(
        "--detect-only", action="store_true",
        help="Find and dump the text, then stop without touching the video"
    )
    # PATCH (dub server). How long on-screen text must stay before it is
    # translated. 0 keeps flashes. Small type uses the second floor.
    parser.add_argument(
        "--screen-text-min-seconds", type=float, default=1.0,
        help="Seconds on-screen text must stay before it is translated"
    )
    parser.add_argument(
        "--screen-text-small-min-seconds", type=float, default=3.0,
        help="Seconds small on-screen text must stay before it is translated"
    )
    # PATCH (dub server). Boxes the caller already has, as {frame: boxes}.
    # Paint those out and skip finding any of its own.
    parser.add_argument(
        "--inpaint-boxes", type=str, default=None,
        help="JSON dump of frame -> boxes to paint out, skipping detection"
    )
    args = parser.parse_args()
    args.inpaint_mode = InpaintMode[args.inpaint_mode.replace('-','_').upper()]
    if args.subtitle_area_coords is None:
        args.subtitle_area_coords = []
    return args