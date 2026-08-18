"""Save an upload to disk without trusting what the client says.

Two things must be checked here, at the door:

* the file type, because the whole pipeline shells out to ffmpeg later;
* the size, and it must be counted while reading. A Content-Length header is
  written by the client, so a client can lie about it.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

CHUNK = 1024 * 1024


def save_upload(
    upload: UploadFile,
    dest_dir: Path,
    allowed: set[str],
    max_bytes: int,
    name: str,
) -> Path:
    """Stream the upload into dest_dir/<name><ext>. Returns the new path."""
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"{name} must be one of: {', '.join(sorted(allowed))}",
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{name}{suffix}"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := upload.file.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{name} is larger than the "
                            f"{max_bytes // (1024 * 1024)} MB limit"
                        ),
                    )
                out.write(chunk)
    except Exception:
        # Do not leave half a file behind for the pipeline to trip over.
        path.unlink(missing_ok=True)
        raise
    return path
