"""A file name taken from the words the viewer will hear.

The desktop tool asks for this with name_from_content. The name is one
title in the target language, long enough to say what is offered and short
enough to read in a folder. Anything that cannot be turned into that title
leaves the original file name in place: a missing name must not fail a dub
that has already been spoken.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from server.steps import llm
from server.steps.translate import _resolve_output_lang

TITLE_MIN = 40
TITLE_MAX = 80

# Windows refuses these in a file name. A line break is one of them: a title
# is a single line. Letters, spaces and diacritics stay, because that is
# what a person searches for.
_ILLEGAL = re.compile(r"[\x00-\x1f\\/:*?\"<>|]+")
_EXTENSION = re.compile(r"\.(mp4|mov|mkv|webm|m4v)$", re.IGNORECASE)


def content_title(
    speech: str,
    target_lang: str,
    api_key: str,
    asr_meta: dict | None = None,
    log: Callable[[str], None] | None = None,
) -> str | None:
    """One title of 40 to 80 characters, or None to keep the original name.

    The model is asked a second time only when the first title is the wrong
    length. An empty answer or a failed request is not a length problem, so
    it is not asked again.
    """
    speech = (speech or "").strip()
    if not speech:
        return None
    _code, lang_name, _same = _resolve_output_lang(target_lang, asr_meta)
    system = (
        f"You name one video file from the spoken script the user sends. "
        f"Write the title in {lang_name}. "
        f"Use between {TITLE_MIN} and {TITLE_MAX} characters: enough to say "
        f"what is offered so a person can understand the video and search "
        f"for it later, and short enough to read as a file name. "
        f"Return only the title, with no file extension and no line break."
    )
    user = f"Spoken script:\n{speech}"
    first = _ask(system, user, api_key, log)
    if first is None:
        return None
    title = clean_title(first)
    if _fits(title):
        return title
    if log is not None:
        log(
            f"Title was {len(title)} characters, outside "
            f"{TITLE_MIN}-{TITLE_MAX}; asking once more"
        )
    second = _ask(
        system,
        f"{user}\n\nThe last title was {len(title)} characters: {title}\n"
        f"Write another title in {lang_name}, between {TITLE_MIN} and "
        f"{TITLE_MAX} characters.",
        api_key,
        log,
    )
    if second is None:
        return None
    title = clean_title(second)
    if _fits(title):
        return title
    if log is not None:
        log(
            f"Title was still {len(title)} characters; "
            f"keeping the original file name"
        )
    return None


def clean_title(text: str) -> str:
    """The title as a file name: words kept, characters Windows forbids gone."""
    cleaned = _ILLEGAL.sub(" ", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return _EXTENSION.sub("", cleaned).strip(" .")


def _fits(title: str) -> bool:
    return TITLE_MIN <= len(title) <= TITLE_MAX


def _ask(system: str, user: str, api_key: str, log) -> str | None:
    try:
        return llm.ask(system, user, api_key)
    except llm.LLMError as error:
        if log is not None:
            log(f"Naming failed: {error}; keeping the original file name")
        return None
