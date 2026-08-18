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
import sys
from pathlib import Path

from server.jobs import PipelineError


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
        enable_deepcache: bool = True,
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
