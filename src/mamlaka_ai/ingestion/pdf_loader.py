"""Extract approved PDFs page by page with PyMuPDF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pymupdf

from mamlaka_ai.config import KNOWLEDGE_BASE_FILES, settings

# Numbered section headings such as "1. Introduction".
SECTION_HEADING_RE = re.compile(r"^\s*(\d{1,2})\.\s+(\S.{1,70})\s*$")


@dataclass
class PdfPage:
    """Text and metadata for one PDF page."""

    document_name: str
    page_number: int  # 1-indexed, as a human would cite it
    text: str
    document_title: str
    page_count: int

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _normalise_text(raw: str) -> str:
    """Clean extracted text while preserving line structure."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("﻿", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def _infer_document_title(first_page_text: str, fallback: str) -> str:
    """Infer a title from the first non-numbered lines."""
    lines = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]
    title_lines: List[str] = []
    for line in lines[:3]:
        if SECTION_HEADING_RE.match(line):
            break
        title_lines.append(line)
        if len(title_lines) == 2:
            break
    if not title_lines:
        return fallback
    return " — ".join(title_lines)


def load_pdf(path: Path) -> List[PdfPage]:
    """Extract all pages with 1-based page numbers."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    with pymupdf.open(path) as doc:
        page_count = doc.page_count
        raw_pages = [_normalise_text(page.get_text("text")) for page in doc]

    if not raw_pages:
        raise ValueError(f"PDF contains no pages: {path.name}")

    title = _infer_document_title(raw_pages[0], fallback=path.stem.replace("_", " "))

    pages = [
        PdfPage(
            document_name=path.name,
            page_number=i + 1,
            text=text,
            document_title=title,
            page_count=page_count,
        )
        for i, text in enumerate(raw_pages)
    ]

    if all(p.is_empty for p in pages):
        raise ValueError(
            f"No extractable text in {path.name}. A scanned PDF would need OCR, "
            "which this pipeline deliberately does not perform."
        )
    return pages


def load_knowledge_base(
    data_dir: Path | None = None, filenames: Sequence[str] = KNOWLEDGE_BASE_FILES
) -> List[PdfPage]:
    """Load every approved PDF in a stable order."""
    directory = Path(data_dir) if data_dir else settings.data_dir
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    pages: List[PdfPage] = []
    missing: List[str] = []
    for name in filenames:
        candidate = directory / name
        if not candidate.exists():
            missing.append(name)
            continue
        pages.extend(load_pdf(candidate))

    if missing:
        raise FileNotFoundError(
            "Missing required knowledge-base PDF(s): "
            + ", ".join(missing)
            + f" (looked in {directory})"
        )
    return pages
