"""Resolve follow-ups and add English search queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from mamlaka_ai.config import settings
from mamlaka_ai.generation.llm import LLMClient, LLMError, get_client
from mamlaka_ai.prompts.rewrite import (
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_prompt,
    parse_rewrite_result,
)

# Signals that a question depends on earlier turns.
_DEPENDENCY_PATTERNS = (
    r"\b(?:she|he|her|his|him|they|them|their|its|it)\b",
    r"\b(?:this|that|these|those)\b",
    r"^\s*(?:and|also|what about|how about|why|then)\b",
    r"^\s*(?:what|which|who|when|where|how)\b[^?]{0,25}\?$",  # very short bare question
    r"\bthe (?:same|latter|former|above)\b",
    r"\bthe (?:project|initiative|document|committee|team)\b",
    # Arabic pronouns and connectives.
    r"(?:^|\s)(?:هو|هي|هم|هن|هذا|هذه|ذلك|تلك|نفسه|نفسها)(?:\s|$|\?|؟)",
    r"^\s*(?:و|ف)(?:ما|من|في|كم|هل|أين|متى|لماذا|كيف)",
    r"(?:ها|ه|هم)$",
    r"\bالمشروع\b|\bالمبادرة\b|\bاللجنة\b|\bالفريق\b",
)
_COMPILED_DEPENDENCY = [re.compile(p, re.IGNORECASE) for p in _DEPENDENCY_PATTERNS]


@dataclass
class ResolvedQuery:
    original: str
    standalone: str
    english: str
    response_language: str = "en"
    depends_on_history: bool = False
    used_llm: bool = False
    rewritten: bool = False
    note: str = ""

    def expansions(self) -> List[str]:
        """Return additional search queries without duplicates."""
        out: List[str] = []
        for candidate in (self.english,):
            normalised = (candidate or "").strip()
            if normalised and normalised != self.standalone.strip():
                out.append(normalised)
        return out


def looks_context_dependent(question: str, word_limit: int = 9) -> bool:
    """Check whether a question likely needs conversation context."""
    text = (question or "").strip()
    if not text:
        return False
    if len(text.split()) <= 4:
        return True
    matched = any(p.search(text) for p in _COMPILED_DEPENDENCY)
    return matched and len(text.split()) <= 25


def _heuristic_fallback(question: str, history: Sequence[Tuple[str, str]]) -> str:
    """Combine the latest question with the previous user question."""
    previous_user = [content for role, content in history if role == "user"]
    if not previous_user:
        return question
    return f"{previous_user[-1].strip()} {question.strip()}".strip()


def resolve_query(
    question: str,
    history: Sequence[Tuple[str, str]] = (),
    language: str = "en",
    client: LLMClient | None = None,
) -> ResolvedQuery:
    """Return standalone search forms and the requested answer language."""
    question = (question or "").strip()
    # Every later conversational turn is rewritten. This lets the model resolve
    # presentation requests such as translating or restating the previous answer
    # without maintaining a brittle list of possible user phrasings.
    needs_resolution = bool(history)
    needs_translation = settings.cross_lingual_expansion and language == "ar"

    if not settings.enable_query_rewrite or (not needs_resolution and not needs_translation):
        return ResolvedQuery(
            original=question,
            standalone=question,
            english=question,
            response_language=language,
            depends_on_history=False,
            note="no rewrite needed",
        )

    active_client = client or get_client()
    prompt = build_rewrite_prompt(
        question, history, max_turns=settings.max_history_turns
    )
    try:
        response = active_client.chat(
            system=REWRITE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
        )
        standalone, english, response_language, depends_on_history = parse_rewrite_result(
            response.text,
            fallback=question,
            fallback_language=language,
            fallback_depends_on_history=needs_resolution,
        )
        # Preserve exact terms for a new standalone topic. The model-provided
        # English form remains available as a cross-language search expansion.
        search_query = standalone if depends_on_history else question
        return ResolvedQuery(
            original=question,
            standalone=search_query,
            english=english,
            response_language=response_language,
            depends_on_history=depends_on_history,
            used_llm=True,
            rewritten=search_query.strip() != question,
            note="llm rewrite",
        )
    except LLMError as exc:
        fallback = _heuristic_fallback(question, history) if needs_resolution else question
        return ResolvedQuery(
            original=question,
            standalone=fallback,
            english=fallback,
            response_language=language,
            depends_on_history=needs_resolution,
            used_llm=False,
            rewritten=fallback != question,
            note=f"heuristic fallback ({exc.__class__.__name__})",
        )
