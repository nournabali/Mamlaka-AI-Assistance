"""Detect Arabic or English text and choose layout direction."""

from __future__ import annotations

import re
from typing import Iterable, Literal

Language = Literal["ar", "en"]

ARABIC = "ar"
ENGLISH = "en"

_ARABIC_LETTER_RE = re.compile(r"[ء-يٮ-ۓۺ-ۿݐ-ݿ]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# Ignore citations and filenames when detecting the user's language.
_CITATION_RE = re.compile(r"\[[^\]]*\]")
_FILENAME_RE = re.compile(r"\b[\w-]+\.pdf\b", re.IGNORECASE)
_NUMERIC_SUFFIX_RE = re.compile(r"(?<=\d)\s*[kmb]\b", re.IGNORECASE)


def _strip_non_authored(text: str) -> str:
    text = _CITATION_RE.sub(" ", text)
    text = _FILENAME_RE.sub(" ", text)
    # Numeric suffixes such as "$2.6M" carry no language signal.
    text = _NUMERIC_SUFFIX_RE.sub(" ", text)
    return text


def arabic_ratio(text: str) -> float:
    """Return the share of letters written in Arabic script."""
    cleaned = _strip_non_authored(text or "")
    arabic = len(_ARABIC_LETTER_RE.findall(cleaned))
    latin = len(_LATIN_LETTER_RE.findall(cleaned))
    total = arabic + latin
    if total == 0:
        return 0.0
    return arabic / total


def detect_language(text: str, threshold: float = 0.25, default: Language = ENGLISH) -> Language:
    """Detect Arabic or English from the scripts used in the text."""
    cleaned = _strip_non_authored(text or "")
    if not _ARABIC_LETTER_RE.search(cleaned) and not _LATIN_LETTER_RE.search(cleaned):
        return default
    return ARABIC if arabic_ratio(text) >= threshold else ENGLISH


def detect_language_with_history(
    text: str, previous_user_messages: Iterable[str] = (), default: Language = ENGLISH
) -> Language:
    """Detect language and use conversation history for script-free follow-ups."""
    cleaned = _strip_non_authored(text or "")
    has_script = bool(_ARABIC_LETTER_RE.search(cleaned) or _LATIN_LETTER_RE.search(cleaned))
    if has_script:
        return detect_language(text, default=default)
    for message in reversed(list(previous_user_messages)):
        candidate_clean = _strip_non_authored(message or "")
        if _ARABIC_LETTER_RE.search(candidate_clean) or _LATIN_LETTER_RE.search(candidate_clean):
            return detect_language(message, default=default)
    return default


def is_rtl(language: str) -> bool:
    return language == ARABIC


def text_direction(language: str) -> str:
    return "rtl" if is_rtl(language) else "ltr"


def text_align(language: str) -> str:
    return "right" if is_rtl(language) else "left"


def language_name(language: str) -> str:
    return "Arabic" if language == ARABIC else "English"
