from __future__ import annotations

from mamlaka_ai.config import KNOWLEDGE_BASE_FILES
from mamlaka_ai.ingestion.chunker import chunk_pages
from mamlaka_ai.ingestion.pdf_loader import load_knowledge_base


def test_three_authorised_pdfs_are_loaded_page_by_page() -> None:
    pages = load_knowledge_base()
    assert len(pages) == 6
    assert {page.document_name for page in pages} == set(KNOWLEDGE_BASE_FILES)
    assert all(page.page_number in {1, 2} for page in pages)
    assert all(page.text.strip() for page in pages)


def test_every_chunk_keeps_required_provenance() -> None:
    chunks = chunk_pages(load_knowledge_base())
    assert len(chunks) == 19
    for chunk in chunks:
        assert chunk.document_name in KNOWLEDGE_BASE_FILES
        assert chunk.page_number >= 1
        assert chunk.section
        assert chunk.chunk_id
        assert chunk.text


def test_an_extra_pdf_cannot_join_the_allow_list(tmp_path) -> None:
    # Missing approved documents must fail rather than silently indexing whatever
    # happens to be in DATA_DIR.
    (tmp_path / "unapproved.pdf").write_bytes(b"%PDF-1.4\n")
    try:
        load_knowledge_base(tmp_path)
    except FileNotFoundError as exc:
        assert "Missing required knowledge-base PDF" in str(exc)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("An incomplete/unauthorised corpus was accepted")
