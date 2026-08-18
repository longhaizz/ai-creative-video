"""What the client may send with POST /dub.

Every choice is a closed list. The desktop client shows a dropdown for each
one, and the server must not trust the client to send only good values.
"""

from __future__ import annotations

from typing import Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field, model_validator

# Bigger models are slower but read speech better. The server retries with
# large-v3 by itself when the first pass looks bad, so this is only a start.
WhisperModel = Literal["tiny", "base", "small", "medium", "large-v3"]

# "original" clones the voice from the video. The rest are voice presets.
VoiceMode = Literal[
    "original",
    "male_young",
    "male_middle",
    "male_old",
    "female_young",
    "female_middle",
    "female_old",
]

# How the subtitle remover paints over the old text.
VsrMode = Literal["sttn-det", "sttn-auto", "lama", "propainter"]


class DubParams(BaseModel):
    """One dub job. Files are sent next to this, not inside it."""

    # -- voice -------------------------------------------------------------
    voice_mode: VoiceMode = "original"
    cfg_value: float = Field(2.0, ge=1.0, le=3.0)
    inference_timesteps: int = Field(10, ge=5, le=30)
    target_lang: str = Field("same", max_length=16)
    whisper_model: WhisperModel = "medium"

    # -- remove the burned-in subtitles ------------------------------------
    remove_subtitle: bool = False
    vsr_mode: VsrMode = "sttn-det"
    # The area to scan, as a share of the frame. The defaults cover the
    # lower band where ad subtitles almost always sit.
    vsr_top: float = Field(0.60, ge=0.0, le=1.0)
    vsr_bottom: float = Field(0.96, ge=0.0, le=1.0)
    vsr_left: float = Field(0.03, ge=0.0, le=1.0)
    vsr_right: float = Field(0.97, ge=0.0, le=1.0)

    # -- lip sync ----------------------------------------------------------
    lipsync: bool = False
    latentsync_steps: int = Field(20, ge=10, le=50)
    latentsync_guidance: float = Field(1.5, ge=1.0, le=3.0)

    # -- burn new subtitles in the target language -------------------------
    burn_subtitle: bool = False
    subtitle_font: str = Field("Arial", max_length=64)
    subtitle_size: int = Field(28, ge=8, le=200)
    # Where the text sits, as a share of the frame height. 0.85 is near the
    # bottom, which is where people expect subtitles.
    subtitle_position: float = Field(0.85, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _area_must_be_a_real_box(self):
        if self.vsr_top >= self.vsr_bottom:
            raise ValueError("vsr_top must be smaller than vsr_bottom")
        if self.vsr_left >= self.vsr_right:
            raise ValueError("vsr_left must be smaller than vsr_right")
        return self


class DubRequest(DubParams):
    """The form POST /dub reads: the settings above, plus the files.

    The files live in the same model on purpose. FastAPI only spreads a form
    model into single fields when it is the one body argument, so an
    UploadFile sitting next to it would turn every setting into a nested
    field named "params".
    """

    video: UploadFile
    reference_audio: UploadFile | None = None

    def settings(self) -> DubParams:
        """The settings alone. The pipeline must not see open file handles."""
        return DubParams(
            **{name: getattr(self, name) for name in DubParams.model_fields}
        )
