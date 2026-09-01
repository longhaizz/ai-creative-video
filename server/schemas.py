"""What the client may send with POST /dub.

Every choice is a closed list. The desktop client shows a dropdown for each
one, and the server must not trust the client to send only good values.
"""

from __future__ import annotations

from typing import Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator, model_validator

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
    # These are the --inference_steps and --guidance_scale of the LatentSync
    # CLI. lipsync.run() passes them straight to the pipeline, so the values
    # in LatentSync/configs/unet/stage2_512.yaml are never read.
    #   50 steps    the default; the pipeline has no ceiling of its own, and
    #               time grows almost linearly, so 100 is our own stop sign
    #   1.5         the same guidance upstream uses for 512
    #   DeepCache   off, in LipsyncModel, so all the steps are really computed
    latentsync_steps: int = Field(50, ge=1, le=100)
    latentsync_guidance: float = Field(1.5, ge=1.0, le=3.0)

    # -- burn new subtitles in the target language -------------------------
    burn_subtitle: bool = False
    subtitle_font: str = Field("Noto Sans", max_length=64)
    # None / omitted / "" → server picks 56px at 1920 tall, scaled by height.
    # A number is exact pixels, not scaled.
    subtitle_size: int | None = Field(None, ge=8, le=200)
    # Where the text sits, as a share of the frame height. 0.75 is still the
    # lower third, higher than the old 0.85 default.
    subtitle_position: float = Field(0.75, ge=0.0, le=1.0)

    @field_validator("subtitle_size", mode="before")
    @classmethod
    def _blank_size_is_auto(cls, value):
        if value is None or value == "":
            return None
        return value

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


class SpeakParams(BaseModel):
    """One clone job: this text, in that voice."""

    text: str = Field(..., min_length=1, max_length=8000)
    cfg_value: float = Field(2.0, ge=1.0, le=3.0)
    inference_timesteps: int = Field(10, ge=5, le=30)

    @field_validator("text")
    @classmethod
    def _text_must_have_words(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("text must not be empty")
        return text


class SpeakRequest(SpeakParams):
    """The form POST /speak reads: the line, plus a voice sample."""

    audio: UploadFile

    def settings(self) -> SpeakParams:
        return SpeakParams(
            **{name: getattr(self, name) for name in SpeakParams.model_fields}
        )
