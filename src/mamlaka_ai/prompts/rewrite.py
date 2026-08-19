"""Prompt and parser for query rewriting."""

from __future__ import annotations

import json
from typing import List, Sequence, Tuple

REWRITE_SYSTEM_PROMPT = """You rewrite search queries for a document retrieval system. \
You never answer questions and you never add facts.

You are given a short conversation and the user's LATEST message. Produce a single self-contained \
content question, so that it can be understood with no conversation history at all. Also select the \
language in which the final answer must be written.

Rules:
- Replace pronouns and references ("she", "he", "they", "it", "its", "that", "the project", "هو", \
"هي", "ذلك") with the specific name or noun they refer to, taken from the conversation.
- For an elliptical follow-up such as "What about X?" or its Arabic equivalent, preserve the exact \
property, relationship, or action requested by the immediately preceding user question and apply it \
to X. Do not substitute a merely related property.
- If the latest message asks to translate, repeat, restate, shorten, expand, reformat, or answer in \
another language, recover the content question from the most recent USER message that contains it. \
Do not use a previous assistant reply as factual input. Express the standalone request in the \
requested answer language while preserving its exact meaning and any requested presentation format, \
such as bullets, a numbered list, a table, or a shorter answer.
- Set depends_on_history to true only when the latest message needs an earlier user message to \
identify its subject or requested content. Set it to false when the latest message independently \
identifies a new subject or question, even if it is short, informal, or misspelled. When it is false, \
repeat the latest message unchanged in standalone so retrieval keeps the user's exact search terms.
- Set response_language to "en" for English or "ar" for Arabic. Honor an explicitly requested \
answer language regardless of the language used to make that request. If no answer language is \
requested, use the language of the latest substantive user question.
- Keep the original meaning. Do not answer, explain, expand the scope, or add any information that \
is not already in the conversation.
- If the latest message is already self-contained, repeat it unchanged.
- Also give an English version of the same question, for searching English documents. If the \
question is already English, repeat it.

Respond with ONLY a JSON object, no other text:
{"standalone": "<self-contained content question>", "english": "<English version>", \
"response_language": "<en or ar>", "depends_on_history": <true or false>}"""


def build_rewrite_prompt(
    question: str,
    history: Sequence[Tuple[str, str]],
    max_turns: int = 6,
    max_chars_per_turn: int = 400,
) -> str:
    """Build a bounded conversation prompt for query rewriting."""
    recent = list(history)[-max_turns:]
    lines: List[str] = ["CONVERSATION SO FAR (for reference resolution only):"]
    if not recent:
        lines.append("(no previous turns)")
    for role, content in recent:
        label = "USER" if role == "user" else "ASSISTANT (previous reply — not a source of truth)"
        text = " ".join((content or "").split())
        if len(text) > max_chars_per_turn:
            text = text[:max_chars_per_turn] + "…"
        lines.append(f"{label}: {text}")

    lines.append("")
    lines.append(f"LATEST USER MESSAGE: {question}")
    lines.append("")
    lines.append(
        'Return only the JSON object with keys "standalone", "english", and '
        '"response_language", and "depends_on_history".'
    )
    return "\n".join(lines)


def parse_rewrite_response(raw: str, fallback: str) -> Tuple[str, str]:
    """Parse query fields while preserving the original two-value API."""
    standalone, english, _, _ = parse_rewrite_result(raw, fallback, "en", False)
    return standalone, english


def parse_rewrite_result(
    raw: str,
    fallback: str,
    fallback_language: str,
    fallback_depends_on_history: bool,
) -> Tuple[str, str, str, bool]:
    """Parse rewritten queries and a supported response language.

    The language falls back to the caller's script-based detection when a
    provider omits the new field or returns an unsupported value.
    """
    text = (raw or "").strip()
    if not text:
        return fallback, fallback, fallback_language, fallback_depends_on_history

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            standalone = str(payload.get("standalone") or "").strip()
            english = str(payload.get("english") or "").strip()
            response_language = str(payload.get("response_language") or "").strip().lower()
            if response_language not in {"ar", "en"}:
                response_language = fallback_language
            depends_on_history = payload.get("depends_on_history")
            if not isinstance(depends_on_history, bool):
                depends_on_history = fallback_depends_on_history
            if _usable_query(standalone):
                return (
                    standalone,
                    english if _usable_query(english) else standalone,
                    response_language,
                    depends_on_history,
                )
            return (
                fallback,
                fallback,
                fallback_language,
                fallback_depends_on_history,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # Accept a short plain rewrite when JSON parsing fails.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) == 1 and len(lines[0]) <= 300 and _usable_query(lines[0]):
        return lines[0], lines[0], fallback_language, fallback_depends_on_history
    return fallback, fallback, fallback_language, fallback_depends_on_history


def _usable_query(candidate: str) -> bool:
    """Check that a rewritten query is usable."""
    text = (candidate or "").strip()
    if not text or not any(character.isalnum() for character in text):
        return False
    lowered = text.casefold()
    return not any(
        marker in lowered
        for marker in (
            "<self-contained question",
            "latest user message:",
            "user question (data",
        )
    )
