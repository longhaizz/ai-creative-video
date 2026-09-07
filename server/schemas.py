"""What the client may send with POST /dub.

Every choice is a closed list. The desktop client shows a dropdown for each
one, and the server must not trust the client to send only good values.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator, model_validator

# Bigger models are slower but read speech better. The job uses this size
# as-is; there is no automatic large-v3 retry.
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

JobKind = Literal["dub", "clone"]


class DubParams(BaseModel):
    """One dub job. Files are sent next to this, not inside it."""

    job_kind: JobKind = "dub"

    # Off means no new voice at all. The job keeps the sound the video came
    # with and only redoes the picture: take the old subtitles off, put new
    # ones on in the language already spoken. Every voice setting below is
    # then ignored, not rejected, so old clients still work.
    dub: bool = True

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
    # Where the text sits, as a share of the frame height. None / omitted
    # means the server puts the new text where the old subtitles sat, which
    # only the subtitle removal step can know; without it, 0.75 -- the lower
    # third, higher than the old 0.85 default. A number is used as given.
    subtitle_position: float | None = Field(None, ge=0.0, le=1.0)

    # The name of the file the client sent, without its folder. It is a hint
    # for the translator, not a setting: an ASR that mishears a word usually
    # still has it spelled right in the title. Filled in from the upload, so
    # a client never sends it.
    source_title: str = Field("", max_length=200)

    @field_validator("subtitle_size", "subtitle_position", mode="before")
    @classmethod
    def _blank_is_auto(cls, value):
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

    @model_validator(mode="after")
    def _no_dub_still_has_to_do_something(self):
        """Catch the two combinations that cannot mean anything.

        Both are caught here, before the job starts, because the work runs
        on a GPU for minutes. Finding out at the end that a flag was
        ignored is the expensive way to learn it.
        """
        if self.dub:
            return self
        if self.lipsync:
            raise ValueError(
                "lipsync has no new voice to follow when dub is false")
        if not (self.remove_subtitle or self.burn_subtitle):
            raise ValueError(
                "with dub false, ask for remove_subtitle or burn_subtitle, "
                "otherwise there is nothing to do")
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
        out = DubParams(
            **{name: getattr(self, name) for name in DubParams.model_fields}
        )
        out.source_title = _clean_title(self.video.filename)
        return out


# What a camera, a phone or a download names a file. A title made only of
# these says nothing about what is in the video, and a wrong hint is worse
# than none: it would have the translator rename the subject after it.
_EMPTY_TITLE_WORDS = {
    "img", "image", "vid", "video", "movie", "clip", "final", "copy", "new",
    "edit", "edited", "export", "output", "out", "draft", "test", "temp",
    "untitled", "download", "downloaded", "raw", "mp4", "full", "version",
    "v", "ver", "screen", "recording", "screenrecording", "whatsapp", "tiktok",
}


def _clean_title(filename: str | None) -> str:
    """The file name, safe to put in a prompt, or nothing.

    The client picks this name, so it is untrusted text that ends up inside
    an instruction to a model. Only one line of it is kept, and only a
    title's worth: a name cannot carry a paragraph of its own orders.

    A name is only worth passing on when a person wrote it about the video.
    "VID_20240115_final2" is what a phone wrote, and handing that over as a
    hint about the subject is worse than handing over nothing. Whether the
    words that survive really describe this video is left to the model,
    which reads them; here we only drop the names that say nothing to
    anybody.
    """
    stem = PurePosixPath((filename or "").replace("\\", "/")).stem
    stem = " ".join(re.split(r"[_\-.]+", stem))
    stem = " ".join(stem.split())[:120]

    told = [w for w in stem.lower().split() if _says_something(w)]
    return stem if told else ""


def _says_something(word: str) -> bool:
    """Is this one word of a file name about the video at all?

    The counters a phone hangs on a name are dropped with the word they
    sit on, so "final2" is as empty as "final". Two letters are asked of an
    alphabet that spells a word in several; one character is enough where
    it is already a word, which is why any letter outside ASCII counts.
    """
    bare = re.sub(r"\d+", "", word)
    if bare in _EMPTY_TITLE_WORDS:
        return False
    if any(c.isalpha() and ord(c) > 127 for c in bare):
        return True
    return sum(c.isalpha() for c in bare) >= 2


class CloneParams(BaseModel):
    """An audio-only voice cloning job.

    The server clones the voice from `reference_audio`, then speaks `text`
    with VoxCPM.
    """

    job_kind: Literal["clone"] = "clone"

    text: str = Field("", max_length=4000)

    # Keep aligned with DubParams, so the desktop UI can reuse sliders.
    cfg_value: float = Field(2.0, ge=1.0, le=3.0)
    inference_timesteps: int = Field(10, ge=5, le=30)

    @field_validator("text", mode="before")
    @classmethod
    def _strip_text(cls, value):
        return (value or "").strip()

    @model_validator(mode="after")
    def _text_must_not_be_blank(self):
        if not (self.text or "").strip():
            raise ValueError("text must not be blank")
        return self


class CloneRequest(CloneParams):
    """The form POST /speak reads: the settings above, plus the voice file."""

    audio: UploadFile

    def settings(self) -> CloneParams:
        """The settings alone. The pipeline must not see open file handles."""
        return CloneParams(
            **{name: getattr(self, name) for name in CloneParams.model_fields}
        )
