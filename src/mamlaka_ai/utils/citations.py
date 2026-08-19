"""Parse, validate, and render document citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

# Accept English and Arabic page labels with common dash styles.
_CITATION_RE = re.compile(
    r"\[\s*(?P<doc>[\w \-.]+?\.pdf)\s*[—–\-]\s*"
    r"(?:Page|page|PAGE|الصفحة|صفحة|ص\.?)\s*(?P<page>\d+)"
    r"(?:\s*[—–\-]\s*(?P<section>[^\]]+?))?\s*\]"
)

# A document reference without a page is invalid.
_PAGELESS_CITATION_RE = re.compile(r"\[\s*(?P<doc>[\w \-.]+?\.pdf)\s*\]")


@dataclass(frozen=True)
class Citation:
    document_name: str
    page_number: int
    section: str | None = None

    @property
    def key(self) -> Tuple[str, int]:
        return (self.document_name, self.page_number)

    def render(self, language: str = "en") -> str:
        page_word = "الصفحة" if language == "ar" else "Page"
        base = f"[{self.document_name} — {page_word} {self.page_number}"
        if self.section:
            base += f" — {self.section}"
        return base + "]"


def parse_citations(text: str) -> List[Citation]:
    """Return well-formed citations in reading order."""
    citations: List[Citation] = []
    for match in _CITATION_RE.finditer(text or ""):
        section = (match.group("section") or "").strip() or None
        citations.append(
            Citation(match.group("doc").strip(), int(match.group("page")), section)
        )
    return citations


def strip_citation_markers(text: str) -> str:
    """Remove validated inline citations for clean UI display."""
    cleaned = _CITATION_RE.sub("", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:؛،])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.split("\n")).strip()


def unique_citations(citations: Iterable[Citation]) -> List[Citation]:
    """Deduplicate citations by document and page."""
    seen: Set[Tuple[str, int]] = set()
    out: List[Citation] = []
    for citation in citations:
        if citation.key in seen:
            continue
        seen.add(citation.key)
        out.append(citation)
    return out


@dataclass
class ValidationOutcome:
    text: str
    valid: List[Citation]
    invalid: List[Citation]
    removed_count: int

    @property
    def has_valid_citation(self) -> bool:
        return bool(self.valid)


def validate_and_clean(
    text: str,
    allowed: Set[Tuple[str, int]],
    known_documents: Sequence[str] = (),
    allowed_sections: Set[Tuple[str, int, str]] | None = None,
) -> ValidationOutcome:
    """Remove citations not supported by this turn's evidence."""
    valid: List[Citation] = []
    invalid: List[Citation] = []
    known = set(known_documents)

    def _replace(match: re.Match) -> str:
        section = (match.group("section") or "").strip() or None
        citation = Citation(match.group("doc").strip(), int(match.group("page")), section)
        section_is_valid = (
            not citation.section
            or allowed_sections is None
            or (citation.document_name, citation.page_number, citation.section)
            in allowed_sections
        )
        if citation.key in allowed and section_is_valid:
            valid.append(citation)
            return match.group(0)
        invalid.append(citation)
        return ""

    cleaned = _CITATION_RE.sub(_replace, text or "")

    # Remove document references that have no page.
    def _replace_pageless(match: re.Match) -> str:
        document = match.group("doc").strip()
        invalid.append(Citation(document, 0, None))
        return ""

    cleaned = _PAGELESS_CITATION_RE.sub(_replace_pageless, cleaned)
    del known  # retained for signature stability / future per-doc diagnostics

    # Clean spacing left by removed citations.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:؛،])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n")).strip()

    return ValidationOutcome(
        text=cleaned,
        valid=unique_citations(valid),
        invalid=unique_citations(invalid),
        removed_count=len(invalid),
    )


def format_source_list(citations: Sequence[Citation], language: str = "en") -> str:
    """Return a Markdown source list."""
    if not citations:
        return ""
    heading = "**المصادر:**" if language == "ar" else "**Sources:**"
    lines = [heading]
    for citation in unique_citations(citations):
        lines.append(f"- `{citation.render(language)}`")
    return "\n".join(lines)


def citations_from_chunks(chunks: Iterable) -> List[Citation]:
    """Build citations from retrieved chunks."""
    out: List[Citation] = []
    for scored in chunks:
        chunk = getattr(scored, "chunk", scored)
        out.append(Citation(chunk.document_name, chunk.page_number, chunk.section or None))
    return unique_citations(out)
