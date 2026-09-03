"""The measure that tells us a spoken line went missing."""

from server.steps.transcribe import speech_seconds


def cue(start, end):
    return {"start": start, "end": end, "speech_start": start, "speech_end": end}


def test_overlapping_cues_are_counted_once():
    """Whisper cues overlap. Adding them up raw would report more speech
    than the clip is long, and a hole would hide behind the double count."""
    assert speech_seconds([cue(0.0, 2.0), cue(1.5, 3.0)]) == 3.0


def test_a_gap_between_cues_is_not_counted():
    """The gap is the missing line. It must not be part of the total."""
    assert speech_seconds([cue(0.0, 2.0), cue(5.0, 6.0)]) == 3.0
