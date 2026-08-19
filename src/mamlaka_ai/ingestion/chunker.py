"""Split PDF pages into section-aware chunks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

from mamlaka_ai.config import settings
from mamlaka_ai.ingestion.pdf_loader import SECTION_HEADING_RE, PdfPage
from mamlaka_ai.utils.claims import extract_values, revision_score

_SENTENCE_END_RE = re.compile(r"(?<=[.!?؟])\s+(?=[A-Zء-ي])")

HEADER_SECTION = "Document Header"


@dataclass
class Chunk:
    """Text and provenance for one indexed chunk."""

    document_name: str
    page_number: int
    section: str
    chunk_id: str
    text: str

    # Search and conflict metadata.
    document_title: str = ""
    section_number: int | None = None
    part_index: int = 0
    part_count: int = 1
    char_count: int = 0
    revision_score: int = 0
    value_types: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Chunk":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    @property
    def citation_key(self) -> tuple[str, int]:
        return (self.document_name, self.page_number)

    def embedding_text(self) -> str:
        """Return text with title and section context for embedding."""
        header = f"{self.document_title} | {self.section}".strip(" |")
        return f"{header}\n{self.text}" if header else self.text


def _make_chunk_id(document_name: str, page_number: int, section: str, part: int) -> str:
    stem = document_name.replace(".pdf", "")
    slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")[:40] or "body"
    digest = hashlib.sha1(
        f"{document_name}|{page_number}|{section}|{part}".encode("utf-8")
    ).hexdigest()[:6]
    return f"{stem}::p{page_number}::{slug}::{part}::{digest}"


def _split_page_into_sections(page: PdfPage) -> List[tuple[str, int | None, str]]:
    """Split a page by numbered headings while preserving order."""
    sections: List[tuple[str, int | None, List[str]]] = []
    current_title = HEADER_SECTION
    current_number: int | None = None
    current_lines: List[str] = []

    for line in page.text.split("\n"):
        match = SECTION_HEADING_RE.match(line)
        if match:
            if current_lines and any(l.strip() for l in current_lines):
                sections.append((current_title, current_number, current_lines))
            current_number = int(match.group(1))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines and any(l.strip() for l in current_lines):
        sections.append((current_title, current_number, current_lines))

    return [
        (title, number, "\n".join(lines).strip())
        for title, number, lines in sections
        if "\n".join(lines).strip()
    ]


def _split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Split long text on paragraphs and sentences with overlap."""
    if len(text) <= max_chars:
        return [text]

    units: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        sentences = _SENTENCE_END_RE.split(paragraph)
        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip()
            if len(candidate) > max_chars and buffer:
                units.append(buffer)
                # Keep context across chunk boundaries.
                buffer = (buffer[-overlap:] + " " + sentence).strip() if overlap else sentence
            else:
                buffer = candidate
        if buffer:
            units.append(buffer)

    # Combine adjacent small units when they fit.
    packed: List[str] = []
    for unit in units:
        if packed and len(packed[-1]) + len(unit) + 2 <= max_chars:
            packed[-1] = f"{packed[-1]}\n\n{unit}"
        else:
            packed.append(unit)
    return packed


def chunk_page(
    page: PdfPage,
    max_chars: int | None = None,
    overlap: int | None = None,
) -> List[Chunk]:
    max_chars = max_chars or settings.chunk_max_chars
    overlap = settings.chunk_overlap_chars if overlap is None else overlap

    chunks: List[Chunk] = []
    for section_title, section_number, body in _split_page_into_sections(page):
        # Keep the heading because it carries retrieval context.
        heading = (
            f"{section_number}. {section_title}" if section_number else section_title
        )
        full_text = body if section_title == HEADER_SECTION else f"{heading}\n{body}"

        parts = _split_long_text(full_text, max_chars, overlap)
        for index, part in enumerate(parts):
            values = extract_values(part)
            chunks.append(
                Chunk(
                    document_name=page.document_name,
                    page_number=page.page_number,
                    section=section_title,
                    chunk_id=_make_chunk_id(
                        page.document_name, page.page_number, section_title, index
                    ),
                    text=part,
                    document_title=page.document_title,
                    section_number=section_number,
                    part_index=index,
                    part_count=len(parts),
                    char_count=len(part),
                    revision_score=revision_score(part, section_title),
                    value_types=sorted({v.value_type for v in values}),
                    values=[f"{v.value_type}:{v.normalised}" for v in values],
                )
            )
    return chunks


def chunk_pages(pages: Sequence[PdfPage], **kwargs: Any) -> List[Chunk]:
    """Chunk all non-empty pages in order."""
    chunks: List[Chunk] = []
    for page in pages:
        if page.is_empty:
            continue
        chunks.extend(chunk_page(page, **kwargs))
    if not chunks:
        raise ValueError("Chunking produced no chunks — check PDF text extraction.")
    return chunks
