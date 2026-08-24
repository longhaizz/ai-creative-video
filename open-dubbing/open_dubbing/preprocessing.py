import dataclasses


@dataclasses.dataclass
class PreprocessingArtifacts:
    """Instance with preprocessing outputs.

    Attributes:
        video_file: A path to a video ad with no audio.
        audio_file: A path to an audio track from the ad.
        audio_vocals_file: A path to an audio track with vocals only.
        audio_background_file: A path to and audio track from the ad with removed
          vocals.
    """

    video_file: str | None
    audio_file: str
    audio_vocals_file: str | None = None
    audio_background_file: str | None = None
