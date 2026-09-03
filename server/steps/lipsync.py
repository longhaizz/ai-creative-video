"""LatentSync: move the mouth in the video to match a new voice track.

The models are loaded once when the server starts and stay on the GPU. That
is why scripts/inference.py was patched: upstream rebuilt the whole pipeline
inside main(), which cost 30-60 seconds on every single call.

Nothing here imports torch or latentsync at module level. The queue tests
must keep running on a laptop with no GPU and no model files.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

from server import config
from server.jobs import PipelineError
from server.steps import audio


SCENE_THRESHOLD = 0.3
MIN_SHOT_SECONDS = 0.4


class NoFaceError(PipelineError):
    """No face in the video.

    This is not a broken job. Ad creatives often have no talking head at
    all, so the pipeline skips lip sync and keeps the rest of the work.
    """

    def __init__(self, message: str):
        super().__init__(message, code="no_face")


@contextlib.contextmanager
def _inside(repo_dir: Path):
    """Run with repo_dir as the working directory.

    LatentSync reads "configs", "checkpoints/whisper/small.pt" and the mask
    image from paths relative to its own root, so it only works when that is
    the current directory.

    ponytail: chdir is process-wide, which is safe here only because one
    worker thread runs one job at a time. If a second worker ever appears,
    pass absolute paths into the upstream code instead.
    """
    old = Path.cwd()
    os.chdir(repo_dir)
    try:
        yield
    finally:
        os.chdir(old)


class LipsyncModel:
    """Holds the loaded pipeline between jobs."""

    name = "latentsync"

    def __init__(
        self,
        repo_dir: Path,
        config_path: Path,
        checkpoint_path: Path,
        enable_deepcache: bool = False,
    ):
        self.repo_dir = Path(repo_dir)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.enable_deepcache = enable_deepcache
        self._pipeline = None
        self._dtype = None
        self._config = None

    def load(self) -> None:
        """Build the pipeline and put it on the GPU. Called once, at start-up."""
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        from omegaconf import OmegaConf

        from scripts.inference import build_pipeline

        with _inside(self.repo_dir):
            self._config = OmegaConf.load(self.config_path)
            self._pipeline, self._dtype = build_pipeline(
                self._config,
                str(self.checkpoint_path.resolve()),
                self.enable_deepcache,
            )

    def run(
        self,
        video: Path,
        audio: Path,
        out_path: Path,
        steps: int,
        guidance: float,
        seed: int = 1247,
    ) -> Path:
        """Lip-sync one video. The paths must be absolute."""
        if self._pipeline is None:
            raise PipelineError("The lip sync model is not loaded")

        from accelerate.utils import set_seed

        set_seed(seed)
        config = self._config
        try:
            with _inside(self.repo_dir):
                self._pipeline(
                    video_path=str(video),
                    audio_path=str(audio),
                    video_out_path=str(out_path),
                    num_frames=config.data.num_frames,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    weight_dtype=self._dtype,
                    width=config.data.resolution,
                    height=config.data.resolution,
                    mask_image_path=config.data.mask_image_path,
                    temp_dir=str(self.repo_dir / "temp"),
                )
        except RuntimeError as error:
            # Upstream reports this as plain English text (image_processor.py).
            # Turn it into a code, so the client never has to match on a
            # sentence that upstream can reword at any time.
            if "Face not detected" in str(error):
                raise NoFaceError(
                    "No face was found in the video. Lip sync needs a "
                    "talking head, and a face in every frame."
                ) from error
            raise

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise PipelineError("Lip sync produced no video")
        return out_path

    def run_shots(
        self,
        video: Path,
        audio_path: Path,
        out_path: Path,
        work: Path,
        steps: int,
        guidance: float,
        ctx=None,
    ) -> Path:
        """Lip-sync each scene on its own, then join them.

        A cut that loses the face used to fail the whole video. Now only that
        shot is left as it was.
        """
        video = Path(video)
        audio_path = Path(audio_path)
        out_path = Path(out_path)
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)

        total = audio.duration(video)
        ranges = shot_ranges(total, detect_scenes(video))
        if len(ranges) <= 1:
            return self.run(video, audio_path, out_path, steps, guidance)

        width, height = audio.video_size(video)
        if ctx is not None:
            ctx.log(f"Lip sync in {len(ranges)} shots")

        pieces: list[Path] = []
        for index, (start, end) in enumerate(ranges):
            if ctx is not None:
                ctx.check_cancel()
            clip = work / f"shot_{index:03d}.mp4"
            wav = work / f"shot_{index:03d}.wav"
            cut_segment(video, start, end, clip)
            cut_segment(audio_path, start, end, wav)
            synced = work / f"shot_{index:03d}_lip.mp4"
            try:
                self.run(
                    clip.resolve(), wav.resolve(), synced.resolve(),
                    steps, guidance,
                )
                piece = synced
            except PipelineError as error:
                if ctx is not None:
                    ctx.log(f"Shot {index + 1}: skip lip sync ({error})")
                piece = clip
            norm = work / f"shot_{index:03d}_n.mp4"
            _scale_clip(piece, norm, width, height)
            pieces.append(norm)

        return concat_videos(pieces, out_path)


def shot_ranges(
    total: float,
    cuts: list[float],
    min_len: float = MIN_SHOT_SECONDS,
) -> list[tuple[float, float]]:
    """Turn cut timestamps into (start, end) shots spanning `total`."""
    edges = [0.0]
    for t in cuts:
        if t - edges[-1] >= min_len and total - t >= min_len:
            edges.append(t)
    edges.append(total)
    ranges = [
        (a, b) for a, b in zip(edges, edges[1:]) if b - a >= min_len
    ]
    return ranges or [(0.0, total)]


def detect_scenes(video: Path, threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Scene-change times from ffmpeg. Empty list means one shot.

    This decodes the picture once and writes nothing, so it is quick. The
    limit is here because a stalled ffmpeg holds the single worker, and this
    one cannot go through run_ffmpeg: the times are printed on stderr.
    """
    try:
        result = subprocess.run(
            [
                config.FFMPEG_BIN, "-hide_banner",
                "-i", str(video),
                "-vf", f"select='gt(scene,{threshold})',showinfo",
                "-an", "-f", "null", "-",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, timeout=audio.FFMPEG_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise PipelineError(
            f"Reading the scene cuts ran longer than "
            f"{audio.FFMPEG_TIMEOUT:.0f}s and was stopped"
        ) from None
    return parse_scene_times(result.stderr or "")


def parse_scene_times(ffmpeg_stderr: str) -> list[float]:
    times: list[float] = []
    for line in ffmpeg_stderr.splitlines():
        if "pts_time:" not in line:
            continue
        token = line.split("pts_time:", 1)[1].split()[0]
        try:
            t = float(token)
        except ValueError:
            continue
        if times and t - times[-1] < MIN_SHOT_SECONDS:
            continue
        times.append(t)
    return times


def cut_segment(src: Path, start: float, end: float, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    length = max(end - start, 0.05)
    command = [
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-i", str(src),
    ]
    if dest.suffix.lower() == ".wav":
        command += ["-vn", "-c:a", "pcm_s16le"]
    else:
        command += ["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    command.append(str(dest))
    audio.run_ffmpeg(command)
    return dest


def _scale_clip(src: Path, dest: Path, width: int, height: int) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio.run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(src),
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest),
    ])
    return dest


def concat_videos(paths: list[Path], dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    listing = dest.with_suffix(".concat.txt")
    listing.write_text(
        "".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in paths),
        encoding="utf-8",
    )
    audio.run_ffmpeg([
        config.FFMPEG_BIN, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy", str(dest),
    ])
    return dest
