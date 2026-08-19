"""Multilingual sentence embeddings for queries and passages."""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import List, Sequence

import numpy as np

from mamlaka_ai.config import settings

# Models that use query and passage prefixes.
_E5_PREFIX_MODELS = ("e5-", "/e5", "multilingual-e5")
_BGE_INSTRUCTION_MODELS = ("bge-m3",)  # bge-m3 needs no prefix

_load_lock = threading.Lock()


class Embedder:
    """Encode normalized query and passage vectors."""

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._model = None

    # Load the model only when embeddings are first requested.
    @property
    def model(self):
        if self._model is None:
            with _load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def uses_e5_prefixes(self) -> bool:
        lowered = self.model_name.lower()
        if any(tag in lowered for tag in _BGE_INSTRUCTION_MODELS):
            return False
        return any(tag in lowered for tag in _E5_PREFIX_MODELS)

    @property
    def dimension(self) -> int:
        # Support old and new sentence-transformers accessors.
        getter = getattr(self.model, "get_embedding_dimension", None)
        if getter is None:
            getter = self.model.get_sentence_embedding_dimension
        return int(getter())

    def _prefix(self, texts: Sequence[str], kind: str) -> List[str]:
        if not self.uses_e5_prefixes:
            return list(texts)
        tag = "query: " if kind == "query" else "passage: "
        return [f"{tag}{t}" for t in texts]

    def _encode(self, texts: Sequence[str], kind: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        prepared = self._prefix(texts, kind)
        vectors = self.model.encode(
            prepared,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine == dot product
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "passage")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]


@lru_cache(maxsize=4)
def get_embedder(model_name: str | None = None, device: str | None = None) -> Embedder:
    """Return a cached embedder instance."""
    return Embedder(model_name, device)
