"""Answer generation and LLM support."""

from mamlaka_ai.generation.answerer import (
    KIND_ANSWER,
    KIND_CAPABILITY_REFUSED,
    KIND_ERROR,
    KIND_GREETING,
    KIND_GUESS_REFUSED,
    KIND_INJECTION_REFUSED,
    KIND_NOT_FOUND,
    Answerer,
    AnswerResult,
)
from mamlaka_ai.generation.conflict import ConflictReport, build_conflict_notice, detect_conflicts
from mamlaka_ai.generation.llm import (
    GroqClient,
    LLMClient,
    LLMError,
    LLMUnavailable,
    OllamaClient,
    get_client,
)
from mamlaka_ai.generation.query_rewriter import ResolvedQuery, resolve_query

__all__ = [
    "Answerer",
    "AnswerResult",
    "KIND_ANSWER",
    "KIND_CAPABILITY_REFUSED",
    "KIND_ERROR",
    "KIND_GREETING",
    "KIND_GUESS_REFUSED",
    "KIND_INJECTION_REFUSED",
    "KIND_NOT_FOUND",
    "ConflictReport",
    "build_conflict_notice",
    "detect_conflicts",
    "LLMError",
    "LLMUnavailable",
    "LLMClient",
    "GroqClient",
    "OllamaClient",
    "get_client",
    "ResolvedQuery",
    "resolve_query",
]
