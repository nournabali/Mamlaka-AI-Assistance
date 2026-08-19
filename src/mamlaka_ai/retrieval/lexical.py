"""Unicode-aware BM25 retrieval for exact terms and figures."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence, Tuple

# Arabic diacritics and tatweel.
_ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_TOKEN_RE = re.compile(r"[\w؀-ۿ]+(?:[.,]\d+)*", re.UNICODE)


def normalise_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"[ىي]", "ي", text)
    text = text.replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return text


def tokenize(text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    # Keep comma-formatted currency/percent figures as searchable tokens.
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = normalise_arabic(text)
    tokens = _TOKEN_RE.findall(text)
    out: List[str] = []
    for token in tokens:
        if token.startswith("ال") and len(token) > 4:
            out.append(token[2:])
        out.append(token)
    return out


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = [tokenize(d) for d in documents]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.doc_count = len(documents)
        self.avg_length = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0

        self.term_frequencies: List[Counter] = [Counter(t) for t in self.doc_tokens]
        document_frequency: Counter = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def score(self, query: str) -> List[float]:
        query_terms = tokenize(query)
        scores = [0.0] * self.doc_count
        if not query_terms or not self.doc_count:
            return scores
        for index in range(self.doc_count):
            frequencies = self.term_frequencies[index]
            length = self.doc_lengths[index] or 1
            total = 0.0
            for term in query_terms:
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1)
                )
                total += idf * frequency * (self.k1 + 1) / denominator
            scores[index] = total
        return scores

    def top_n(self, query: str, n: int) -> List[Tuple[int, float]]:
        scored = [(i, s) for i, s in enumerate(self.score(query)) if s > 0]
        scored.sort(key=lambda pair: -pair[1])
        return scored[:n]
