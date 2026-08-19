"""FAISS vector storage with chunk metadata and model checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from mamlaka_ai.config import settings
from mamlaka_ai.ingestion.chunker import Chunk


@dataclass
class ScoredChunk:
    """A retrieved chunk with ranking metadata."""

    chunk: Chunk
    score: float  # cosine similarity to the query — comparable across sources
    fusion_score: float = 0.0  # reciprocal-rank-fusion score, used for ordering
    dense_rank: int | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    source: str = "dense"  # dense | lexical | hybrid | revision-sweep | value-sweep

    @property
    def citation(self) -> str:
        return f"[{self.chunk.document_name} — Page {self.chunk.page_number}]"

    def debug_row(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_name": self.chunk.document_name,
            "page_number": self.chunk.page_number,
            "section": self.chunk.section,
            "score": round(self.score, 4),
            "fusion_score": round(self.fusion_score, 4),
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
            "lexical_score": None if self.lexical_score is None else round(self.lexical_score, 3),
            "found_by": self.source,
            "chars": self.chunk.char_count,
            "revision_score": self.chunk.revision_score,
            "text": self.chunk.text,
        }


class VectorStore:
    def __init__(self, chunks: List[Chunk], index, embedding_model: str, dimension: int) -> None:
        self.chunks = chunks
        self.index = index
        self.embedding_model = embedding_model
        self.dimension = dimension

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder) -> "VectorStore":
        import faiss

        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        vectors = embedder.encode_passages([c.embedding_text() for c in chunks])
        if vectors.shape[0] != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count.")
        dimension = int(vectors.shape[1])
        # Exact search is sufficient for this small corpus.
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        return cls(list(chunks), index, embedder.model_name, dimension)

    def save(self, index_dir: Path | None = None) -> Path:
        import faiss

        directory = Path(index_dir) if index_dir else settings.index_dir
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "faiss.index"))
        (directory / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "embedding_model": self.embedding_model,
                    "dimension": self.dimension,
                    "chunk_count": len(self.chunks),
                    "documents": sorted({c.document_name for c in self.chunks}),
                    "pages_per_document": {
                        doc: sorted({c.page_number for c in self.chunks if c.document_name == doc})
                        for doc in sorted({c.document_name for c in self.chunks})
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, index_dir: Path | None = None, expected_model: str | None = None) -> "VectorStore":
        import faiss

        directory = Path(index_dir) if index_dir else settings.index_dir
        index_file = directory / "faiss.index"
        chunks_file = directory / "chunks.json"
        manifest_file = directory / "manifest.json"

        missing = [p.name for p in (index_file, chunks_file, manifest_file) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Vector index incomplete in {directory} (missing: {', '.join(missing)}). "
                "Run:  python scripts/build_index.py"
            )

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        model_name = manifest.get("embedding_model", "")
        if expected_model and model_name and expected_model != model_name:
            raise ValueError(
                f"Index was built with embedding model '{model_name}' but EMBEDDING_MODEL is "
                f"'{expected_model}'. Rebuild the index:  python scripts/build_index.py --force"
            )

        chunks = [
            Chunk.from_dict(payload)
            for payload in json.loads(chunks_file.read_text(encoding="utf-8"))
        ]
        index = faiss.read_index(str(index_file))
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index/metadata mismatch: {index.ntotal} vectors vs {len(chunks)} chunks. "
                "Rebuild the index:  python scripts/build_index.py --force"
            )
        return cls(chunks, index, model_name, int(manifest.get("dimension", index.d)))

    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(vector, min(k, self.index.ntotal))
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def search_subset(
        self, query_vector: np.ndarray, subset: Iterable[int]
    ) -> List[Tuple[int, float]]:
        """Score selected chunks by cosine similarity."""
        positions = list(subset)
        if not positions:
            return []
        vector = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        matrix = np.vstack([self.index.reconstruct(int(p)) for p in positions])
        similarities = (matrix @ vector.T).ravel()
        ranked = sorted(zip(positions, similarities), key=lambda pair: -pair[1])
        return [(int(p), float(s)) for p, s in ranked]

    def __len__(self) -> int:
        return len(self.chunks)
