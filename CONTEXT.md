# Dub pipeline

One video in, one dubbed video out. Speech is read from the mix, the voice
is replaced, and the original music stays underneath.

## Language

**Cue**:
One timed line of heard speech that TTS will speak. Usually one Whisper
segment; split only when too long for TTS.
_Avoid_: utterance, window, VAD blob

**SPEAKER_00**:
The single clone-voice slot for the whole clip. Not a counted person.
_Avoid_: speaker 0, diarization, SPEAKER_01 in the segment step

**Mix**:
The original soundtrack extracted from the video. Whisper transcribes this.
_Avoid_: vocals stem as the ASR source

**Vocals**:
The Demucs voice stem. Clone-reference source for original-voice mode.
_Avoid_: transcript source

**Music**:
The Demucs non-voice stem. The bed mixed under the new voice.
_Avoid_: accompaniment, no_vocals as a domain term in new code

**speech_start / speech_end**:
The word-timestamp span inside a Cue used to fit TTS. Tighter than start
and end, which Whisper often pads.
_Avoid_: using start/end as the TTS slot

**Block**:
The speech between two Anchors. One TTS take says a whole Block, so the
voice keeps one intonation across its sentences.
_Avoid_: speaking one Cue at a time

**Anchor**:
The end of a sentence followed by a pause of 0.15s or more, any pause of
0.8s or more, or a scene cut. Where the dub is put back on the original
clock. Ads are spoken without real breaks, so the sentence rule is the one
that fires.
_Avoid_: treating every Cue boundary as an anchor; expecting long silences

**Drift**:
How late a Block starts compared to the original. Capped, and never carried
past the next Anchor. A scene cut allows none.
_Avoid_: letting drift add up over the video

**Take**:
One TTS attempt at a Block. Judged by listening to it back with Whisper:
the words must match, and the length must be possible for those words.
_Avoid_: keeping the first take unheard
