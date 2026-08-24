"""Run the real pipeline once, by hand, on a real machine.

This is not a pytest file, and the name is deliberate: pytest only collects
test_*.py, so it never runs here by accident. It needs a GPU, every model
downloaded, and an OpenAI key, and it takes minutes.

Everything the pipeline decides is already covered by the fast tests. What
this proves is the part they cannot: that the models load, that they talk to
each other, and that a real video comes out the far end.

    export API_KEY=x OPENAI_API_KEY=sk-...
    python -m server.tests.smoke_run_dub clip.mp4

    python -m server.tests.smoke_run_dub clip.mp4 --lipsync --remove-subtitle \
        --burn-subtitle --lang vi
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from server import config
from server.pipeline import Models, _dub
from server.schemas import DubParams


class PrintingContext:
    """What JobRunner would hand the pipeline, but printing as it goes."""

    def __init__(self, params, workdir: Path):
        self.job_id = "smoke"
        self.params = params
        self.workdir = workdir
        self.started = time.monotonic()

    def _stamp(self) -> str:
        return f"[{time.monotonic() - self.started:6.1f}s]"

    def log(self, message: str) -> None:
        print(f"{self._stamp()} {message}", flush=True)

    def step(self, name: str) -> None:
        print(f"{self._stamp()} == {name} ==", flush=True)

    def check_cancel(self) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("smoke_out"))
    parser.add_argument("--lang", default="same")
    parser.add_argument("--voice", default="original")
    parser.add_argument("--whisper", default="medium")
    parser.add_argument("--lipsync", action="store_true")
    parser.add_argument("--remove-subtitle", action="store_true")
    parser.add_argument("--burn-subtitle", action="store_true")
    args = parser.parse_args()

    if not args.video.is_file():
        print(f"No such video: {args.video}")
        return 1
    if not config.OPENAI_API_KEY:
        print("Set OPENAI_API_KEY: the rewrite step needs it, even for 'same'")
        return 1

    work = args.out / "work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # The pipeline looks for the video by name, the way save_upload leaves it.
    shutil.copy2(args.video, work / f"video{args.video.suffix.lower()}")

    params = DubParams(
        target_lang=args.lang,
        voice_mode=args.voice,
        whisper_model=args.whisper,
        lipsync=args.lipsync,
        remove_subtitle=args.remove_subtitle,
        burn_subtitle=args.burn_subtitle,
    )

    from server.steps.synth import VoxCPMModel

    lipsync = None
    if config.LOAD_LIPSYNC:
        from server.steps.lipsync import LipsyncModel

        lipsync = LipsyncModel(
            config.LATENTSYNC_DIR,
            config.LATENTSYNC_CONFIG,
            config.LATENTSYNC_CHECKPOINT,
        )
    models = Models(
        voice=VoxCPMModel(),
        lipsync=lipsync,
    )

    ctx = PrintingContext(params, work)
    ctx.step("Loading the models")
    for model in models.as_list():
        started = time.monotonic()
        model.load()
        ctx.log(f"{model.name} loaded in {time.monotonic() - started:.1f}s")

    ctx.step("Running the pipeline")
    result = _dub(ctx, models)

    final = args.out / f"{args.video.stem}_dubbed.mp4"
    shutil.copy2(result, final)
    ctx.log(f"Done: {final} ({final.stat().st_size / 1024 / 1024:.1f} MB)")
    ctx.log(f"Files kept in {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
