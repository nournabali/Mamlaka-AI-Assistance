"""Prompt templates and localized messages."""

from mamlaka_ai.prompts.refusals import (
    CAPABILITY_REFUSAL,
    FULL_DOCUMENT_REFUSAL,
    GREETING,
    GUESS_REFUSAL,
    INDEX_MISSING,
    INJECTION_REFUSAL,
    LLM_ERROR,
    LLM_UNAVAILABLE,
    NOT_FOUND,
    UNCITED_ANSWER,
    message,
)
from mamlaka_ai.prompts.rewrite import (
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_prompt,
    parse_rewrite_response,
)
from mamlaka_ai.prompts.system import (
    INSUFFICIENT_EVIDENCE,
    build_user_message,
    format_context_block,
    system_prompt,
)

__all__ = [
    "CAPABILITY_REFUSAL",
    "FULL_DOCUMENT_REFUSAL",
    "GREETING",
    "GUESS_REFUSAL",
    "INDEX_MISSING",
    "INJECTION_REFUSAL",
    "LLM_ERROR",
    "LLM_UNAVAILABLE",
    "NOT_FOUND",
    "UNCITED_ANSWER",
    "message",
    "REWRITE_SYSTEM_PROMPT",
    "build_rewrite_prompt",
    "parse_rewrite_response",
    "INSUFFICIENT_EVIDENCE",
    "build_user_message",
    "format_context_block",
    "system_prompt",
]
