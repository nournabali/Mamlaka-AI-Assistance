"""Hybrid dense and BM25 retrieval with conflict-aware sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from mamlaka_ai.config import settings
from mamlaka_ai.ingestion.chunker import Chunk
from mamlaka_ai.retrieval.embedder import Embedder, get_embedder
from mamlaka_ai.retrieval.lexical import BM25
from mamlaka_ai.retrieval.vector_store import ScoredChunk, VectorStore
from mamlaka_ai.utils.claims import extract_values

RRF_K = 60  # Standard reciprocal-rank damping.


@dataclass
class RetrievalResult:
    query: str
    search_query: str  # Query after reference resolution or translation.
    chunks: List[ScoredChunk] = field(default_factory=list)
    best_score: float = 0.0
    passed_gate: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def documents(self) -> List[str]:
        seen: List[str] = []
        for scored in self.chunks:
            if scored.chunk.document_name not in seen:
                seen.append(scored.chunk.document_name)
        return seen

    def allowed_citations(self) -> set[tuple[str, int]]:
        """Return document and page pairs that may be cited."""
        return {sc.chunk.citation_key for sc in self.chunks}

    def allowed_citation_sections(self) -> set[tuple[str, int, str]]:
        """Return document, page, and section triples that may be cited."""
        return {
            (sc.chunk.document_name, sc.chunk.page_number, sc.chunk.section.strip())
            for sc in self.chunks
            if sc.chunk.section.strip()
        }

    def debug_rows(self) -> List[dict]:
        return [sc.debug_row() for sc in self.chunks]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or get_embedder(store.embedding_model or None)
        self.chunks: List[Chunk] = store.chunks
        self.bm25 = BM25([c.embedding_text() for c in self.chunks])
        # Cache chunk groups used by conflict sweeps.
        self._revision_positions = [
            i for i, c in enumerate(self.chunks) if c.revision_score >= 3
        ]
        self._value_positions: Dict[str, List[int]] = {}
        for i, chunk in enumerate(self.chunks):
            for value_type in chunk.value_types:
                self._value_positions.setdefault(value_type, []).append(i)

    def retrieve(
        self,
        query: str,
        expansions: Sequence[str] = (),
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Retrieve evidence for a query.

        Args:
            query: Standalone search query.
            expansions: Additional query forms, such as a translation.
            top_k: Primary result limit.

        Returns:
            Ranked evidence and relevance-gate status.
        """
        top_k = top_k or settings.top_k
        notes: List[str] = []
        if not query.strip():
            return RetrievalResult(query=query, search_query=query, notes=["empty query"])

        query_variants = [query, *[e for e in expansions if e and e.strip() != query.strip()]]

        # Dense search.
        dense_k = max(settings.dense_candidates, top_k * 3)
        cosine: Dict[int, float] = {}
        dense_ranking: List[int] = []
        for variant in query_variants:
            vector = self.embedder.encode_query(variant)
            for position, score in self.store.search(vector, dense_k):
                cosine[position] = max(cosine.get(position, -1.0), score)
                if position not in dense_ranking:
                    dense_ranking.append(position)
        # Rank each chunk by its best score across query forms.
        dense_ranking.sort(key=lambda p: -cosine[p])
        dense_rank_of = {p: r + 1 for r, p in enumerate(dense_ranking)}

        # BM25 search.
        lexical_scores: Dict[int, float] = {}
        lexical_ranking: List[int] = []
        for variant in query_variants:
            for position, score in self.bm25.top_n(variant, settings.lexical_candidates):
                if score > lexical_scores.get(position, 0.0):
                    lexical_scores[position] = score
        for position, _ in sorted(lexical_scores.items(), key=lambda kv: -kv[1]):
            lexical_ranking.append(position)
        lexical_rank_of = {p: r + 1 for r, p in enumerate(lexical_ranking)}

        # Fuse dense and BM25 rankings.
        fused: Dict[int, float] = {}
        for position, rank in dense_rank_of.items():
            fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)
        for position, rank in lexical_rank_of.items():
            fused[position] = fused.get(position, 0.0) + 1.0 / (RRF_K + rank)

        primary_vector = self.embedder.encode_query(query)
        self._ensure_cosine(cosine, fused.keys(), primary_vector)

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], -cosine.get(kv[0], 0.0)))
        selected: Dict[int, ScoredChunk] = {}
        for position, fusion_score in ordered[:top_k]:
            selected[position] = self._make_scored(
                position,
                cosine.get(position, 0.0),
                fusion_score,
                dense_rank_of.get(position),
                lexical_rank_of.get(position),
                lexical_scores.get(position),
            )

        best_score = max((sc.score for sc in selected.values()), default=0.0)

        # Add related revision chunks.
        added = self._revision_sweep(selected, primary_vector, best_score, cosine, fused)
        if added:
            notes.append(
                f"revision sweep added {added} chunk(s) that describe a revised value"
            )

        # Add chunks with comparable value types.
        added = self._value_sweep(selected, primary_vector, best_score, cosine, fused)
        if added:
            notes.append(f"value sweep added {added} chunk(s) carrying comparable figures")

        final = sorted(
            selected.values(),
            key=lambda sc: (-sc.fusion_score, -sc.score),
        )
        best_score = max((sc.score for sc in final), default=0.0)
        passed_gate = best_score >= settings.min_similarity
        if not passed_gate:
            notes.append(
                f"best cosine {best_score:.3f} < gate {settings.min_similarity:.3f} "
                "— treated as no supporting evidence"
            )

        return RetrievalResult(
            query=query,
            search_query=query_variants[0],
            chunks=final,
            best_score=best_score,
            passed_gate=passed_gate,
            notes=notes,
        )

    def _ensure_cosine(self, cosine: Dict[int, float], positions, vector) -> None:
        """Score chunks missing from the dense search."""
        missing = [p for p in positions if p not in cosine]
        if missing:
            for position, score in self.store.search_subset(vector, missing):
                cosine[position] = score

    def _make_scored(
        self,
        position: int,
        score: float,
        fusion_score: float,
        dense_rank: int | None,
        lexical_rank: int | None,
        lexical_score: float | None,
        source: str | None = None,
    ) -> ScoredChunk:
        if source is None:
            if dense_rank and lexical_rank:
                source = "hybrid"
            elif lexical_rank:
                source = "lexical"
            else:
                source = "dense"
        return ScoredChunk(
            chunk=self.chunks[position],
            score=score,
            fusion_score=fusion_score,
            dense_rank=dense_rank,
            lexical_rank=lexical_rank,
            lexical_score=lexical_score,
            source=source,
        )

    def _revision_sweep(
        self,
        selected: Dict[int, ScoredChunk],
        vector,
        best_score: float,
        cosine: Dict[int, float],
        fused: Dict[int, float],
    ) -> int:
        candidates = [p for p in self._revision_positions if p not in selected]
        if not candidates or best_score <= 0:
            return 0
        threshold = best_score - settings.revision_sweep_delta
        added = 0
        for position, score in self.store.search_subset(vector, candidates):
            if added >= settings.revision_sweep_max or score < threshold:
                break
            cosine[position] = score
            # Keep revisions visible without displacing primary evidence.
            fusion_score = min(fused.values(), default=0.01) * 0.9
            fused[position] = fusion_score
            selected[position] = self._make_scored(
                position, score, fusion_score, None, None, None, source="revision-sweep"
            )
            added += 1
        return added

    def _value_sweep(
        self,
        selected: Dict[int, ScoredChunk],
        vector,
        best_score: float,
        cosine: Dict[int, float],
        fused: Dict[int, float],
    ) -> int:
        """Add nearby chunks with comparable value types."""
        present_types = {
            value.value_type
            for scored in selected.values()
            for value in extract_values(scored.chunk.text)
        }
        if not present_types or best_score <= 0:
            return 0

        candidates: List[int] = []
        for value_type in present_types:
            for position in self._value_positions.get(value_type, []):
                if position not in selected and position not in candidates:
                    candidates.append(position)
        if not candidates:
            return 0

        threshold = best_score - settings.value_sweep_delta
        added = 0
        for position, score in self.store.search_subset(vector, candidates):
            if added >= settings.value_sweep_max or score < threshold:
                break
            cosine[position] = score
            fusion_score = min(fused.values(), default=0.01) * 0.8
            fused[position] = fusion_score
            selected[position] = self._make_scored(
                position, score, fusion_score, None, None, None, source="value-sweep"
            )
            added += 1
        return added


def load_retriever(index_dir=None) -> Retriever:
    store = VectorStore.load(index_dir, expected_model=settings.embedding_model)
    return Retriever(store)
