from __future__ import annotations

import json
from pathlib import Path

import pytest

from mamlaka_ai.ingestion.chunker import Chunk


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def indexed_chunks() -> list[Chunk]:
    payload = json.loads(
        (ROOT / "artifacts" / "index" / "chunks.json").read_text(encoding="utf-8")
    )
    return [Chunk.from_dict(item) for item in payload]
