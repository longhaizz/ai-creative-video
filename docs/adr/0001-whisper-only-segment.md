# Always SPEAKER_00; Whisper like the CLI

The segment step used to run Demucs, Silero VAD, Pyannote, and Whisper in a
child venv so two-speaker ads could clone two voices. Quiet Hindi voiceover
was then flagged as no-speech, language was locked, and Hindi always retried
large-v3 — worse than a plain faster-whisper pass on the mix.

We dropped Pyannote and Silero. Every Cue is SPEAKER_00 (one clone for the
clip). Whisper runs in-process on the mix with vad_filter=True, no language
lock, and no large-v3 retry — the same call as whisper-cli. Demucs stays,
because the dub still needs a music bed and a vocal stem for clone refs.

Two-voice ads share one clone. Mixed-language ads may flip mid-clip. That
is the cost of matching CLI transcripts on quiet Hindi VO.
