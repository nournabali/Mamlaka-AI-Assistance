"""Shared language, citation, injection, and claim helpers."""

from mamlaka_ai.utils.citations import (
    Citation,
    citations_from_chunks,
    format_source_list,
    parse_citations,
    validate_and_clean,
)
from mamlaka_ai.utils.claims import Value, extract_values, has_revision_language, revision_score
from mamlaka_ai.utils.injection import InjectionCheck, check_user_input, neutralise_document_text
from mamlaka_ai.utils.language import (
    ARABIC,
    ENGLISH,
    detect_language,
    detect_language_with_history,
    is_rtl,
)

__all__ = [
    "Citation",
    "citations_from_chunks",
    "format_source_list",
    "parse_citations",
    "validate_and_clean",
    "Value",
    "extract_values",
    "has_revision_language",
    "revision_score",
    "InjectionCheck",
    "check_user_input",
    "neutralise_document_text",
    "ARABIC",
    "ENGLISH",
    "detect_language",
    "detect_language_with_history",
    "is_rtl",
]
