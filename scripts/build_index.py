#!/usr/bin/env python3
"""Build or inspect the FAISS index for the approved PDFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mamlaka_ai.config import KNOWLEDGE_BASE_FILES, settings
from mamlaka_ai.ingestion.chunker import chunk_pages
from mamlaka_ai.ingestion.pdf_loader import load_knowledge_base
from mamlaka_ai.retrieval.embedder import get_embedder
from mamlaka_ai.retrieval.vector_store import VectorStore


def _print_chunk_table(chunks) -> None:
    header = f"{'#':>3}  {'document':<38} {'pg':>2}  {'section':<28} {'chars':>5}  {'rev':>3}"
    print(header)
    print("-" * len(header))
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"{index:>3}  {chunk.document_name:<38} {chunk.page_number:>2}  "
            f"{chunk.section[:28]:<28} {chunk.char_count:>5}  {chunk.revision_score:>3}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Almamlaka RAG vector index.")
    parser.add_argument("--force", action="store_true", help="rebuild even if an index exists")
    parser.add_argument("--inspect", action="store_true", help="print chunks without embedding")
    parser.add_argument("--data-dir", type=Path, default=None, help="override DATA_DIR")
    parser.add_argument("--index-dir", type=Path, default=None, help="override INDEX_DIR")
    args = parser.parse_args(argv)

    data_dir = args.data_dir or settings.data_dir
    index_dir = args.index_dir or settings.index_dir

    print(f"Knowledge base : {', '.join(KNOWLEDGE_BASE_FILES)}")
    print(f"Data directory : {data_dir}")

    try:
        pages = load_knowledge_base(data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2

    documents = sorted({p.document_name for p in pages})
    print(f"Extracted      : {len(pages)} pages from {len(documents)} documents")

    chunks = chunk_pages(pages)
    print(f"Chunked        : {len(chunks)} chunks "
          f"(max {settings.chunk_max_chars} chars, overlap {settings.chunk_overlap_chars})")

    if args.inspect:
        print()
        _print_chunk_table(chunks)
        return 0

    if not args.force and (index_dir / "manifest.json").exists():
        try:
            manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        if (
            manifest.get("embedding_model") == settings.embedding_model
            and manifest.get("chunk_count") == len(chunks)
        ):
            print(
                f"\nIndex already current at {index_dir} "
                f"({manifest.get('chunk_count')} chunks, {manifest.get('embedding_model')}). "
                "Use --force to rebuild."
            )
            return 0

    print(f"Embedding model: {settings.embedding_model} (device={settings.embedding_device})")
    print("Loading the embedding model (first run downloads it)…")
    embedder = get_embedder()
    print(f"Embedding      : {len(chunks)} chunks, dim={embedder.dimension}, "
          f"e5_prefixes={embedder.uses_e5_prefixes}")

    store = VectorStore.build(chunks, embedder)
    target = store.save(index_dir)

    print(f"\nIndex written  : {target}")
    print(f"  faiss.index   ({store.index.ntotal} vectors, dim {store.dimension})")
    print(f"  chunks.json   ({len(store.chunks)} chunks with document/page/section metadata)")
    print("  manifest.json")
    print("\nDone. Start the app with:  streamlit run src/mamlaka_ai/ui/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
