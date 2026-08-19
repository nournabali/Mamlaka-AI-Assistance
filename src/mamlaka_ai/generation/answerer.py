"""Run one grounded RAG conversation turn."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from mamlaka_ai.config import settings
from mamlaka_ai.generation.conflict import ConflictReport, build_conflict_notice, detect_conflicts
from mamlaka_ai.generation.llm import LLMClient, LLMError, LLMResponse, LLMUnavailable, get_client
from mamlaka_ai.generation.query_rewriter import ResolvedQuery, resolve_query
from mamlaka_ai.prompts import refusals
from mamlaka_ai.prompts.system import (
    INSUFFICIENT_EVIDENCE,
    build_user_message,
    format_context_block,
    system_prompt,
)
from mamlaka_ai.retrieval.retriever import RetrievalResult, Retriever
from mamlaka_ai.utils.citations import (
    Citation,
    parse_citations,
    validate_and_clean,
)
from mamlaka_ai.utils.injection import check_user_input
from mamlaka_ai.utils.language import detect_language_with_history

# Result types used by the UI and tests.
KIND_ANSWER = "answer"
KIND_NOT_FOUND = "not_found"
KIND_CAPABILITY_REFUSED = "capability_refused"
KIND_INJECTION_REFUSED = "injection_refused"
KIND_GUESS_REFUSED = "guess_refused"
KIND_GREETING = "greeting"
KIND_ERROR = "error"

_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|greetings|thanks|thank\s+you|"
    r"thx|salam|salaam|مرحبا|مرحباً|أهلا|أهلاً|السلام\s+عليكم|صباح\s+الخير|مساء\s+الخير|"
    r"شكرا|شكراً|شكرًا|تحياتي)\s*[!.،؟?]*\s*$",
    re.IGNORECASE,
)

# Creative and entertainment tasks are unsupported capabilities, not missing
# project facts. Keep this route narrow so document-related requests still
# reach retrieval and receive the evidence-specific response when appropriate.
_UNSUPPORTED_TASK_RE = re.compile(
    r"(?:"
    r"\b(?:tell|share)\s+(?:me\s+)?(?:a\s+)?(?:joke|riddle|story)\b"
    r"|\bmake\s+me\s+laugh\b"
    r"|\b(?:write|compose|create)\s+(?:me\s+)?(?:a\s+)?(?:poem|song|joke|riddle)\b"
    r"|(?:قل|قول|احك|احكي)\s+(?:لي\s+)?(?:نكتة|لغز|قصة)"
    r"|(?:اكتب|ألّف|الف)\s+(?:لي\s+)?(?:قصيدة|أغنية|اغنية|نكتة|لغز)"
    r"|أضحكني|اضحكني"
    r")",
    re.IGNORECASE,
)

_FULL_DOCUMENT_REQUEST_RE = re.compile(
    r"(?:"
    r"\b(?:give|show|provide|display|print|paste|copy|send|return|extract)\b.{0,80}"
    r"\b(?:"
    r"(?:full|complete|entire|whole)\s+(?:content|contents|text)(?:\s+of)?"
    r"|(?:full|complete|entire|whole)\s+(?:documents?|docs?|pdfs?|files?)"
    r"(?:\s+(?:content|contents|text|information|info))?"
    r"|(?:all\s+(?:the\s+)?(?:content|contents|text|information|info)|everything)"
    r"\s+(?:in|from|of)\s+(?:the\s+)?(?:documents?|docs?|pdfs?|files?)"
    r"|all\s+(?:the\s+)?(?:documents?|docs?|pdfs?|files?)"
    r")\b"
    r"|"
    r"(?:أعطني|اعطني|أعطيني|اعطيني|اعرض|أرسل|ارسل|انسخ|استخرج).{0,80}"
    r"(?:"
    r"(?:كل|كافة|جميع)\s+(?:المحتوى|المحتويات|النص|النصوص|المعلومات)"
    r"|(?:المحتوى|المحتويات|النص|النصوص)\s+(?:الكامل|الكاملة|كاملاً|كاملًا|بالكامل)"
    r"|(?:كامل|كافة|جميع)\s+(?:المستندات|الوثائق|الملفات)"
    r"|(?:المستند|الوثيقة|الملف)\s+(?:كاملاً|كاملًا|بالكامل)"
    r"|(?:المستندات|الوثائق|الملفات)\s+(?:كاملة|كلها|بالكامل)"
    r")"
    r")",
    re.IGNORECASE,
)


@dataclass
class AnswerResult:
    answer: str
    language: str
    kind: str = KIND_ANSWER
    citations: List[Citation] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    conflict: ConflictReport | None = None
    debug: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_grounded_answer(self) -> bool:
        return self.kind == KIND_ANSWER

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflict and self.conflict.has_conflict)


class Answerer:
    """Answer one question using retrieved document evidence."""

    def __init__(self, retriever: Retriever, client: LLMClient | None = None) -> None:
        self.retriever = retriever
        self.client = client or get_client()

    def answer(
        self,
        question: str,
        history: Sequence[Tuple[str, str]] = (),
        top_k: int | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        question = (question or "").strip()
        previous_user_messages = [c for r, c in history if r == "user"]
        language = detect_language_with_history(question, previous_user_messages)
        debug: Dict[str, Any] = {"language": language, "question": question}

        if not question:
            return AnswerResult(
                answer=refusals.message(refusals.GREETING, language),
                language=language,
                kind=KIND_GREETING,
                debug=debug,
            )

        # Block unsafe requests before calling the model.
        screening = check_user_input(question)
        debug["injection_check"] = {
            "verdict": screening.verdict,
            "matched": screening.matched,
        }
        if screening.verdict == "injection":
            return AnswerResult(
                answer=refusals.message(refusals.INJECTION_REFUSAL, language),
                language=language,
                kind=KIND_INJECTION_REFUSED,
                debug=debug,
            )
        if screening.verdict == "forced_guess":
            return AnswerResult(
                answer=refusals.message(refusals.GUESS_REFUSAL, language),
                language=language,
                kind=KIND_GUESS_REFUSED,
                debug=debug,
            )

        if _UNSUPPORTED_TASK_RE.search(question):
            debug["route"] = "unsupported_capability"
            return AnswerResult(
                answer=refusals.message(refusals.CAPABILITY_REFUSAL, language),
                language=language,
                kind=KIND_CAPABILITY_REFUSED,
                debug=debug,
            )

        if _FULL_DOCUMENT_REQUEST_RE.search(question):
            debug["route"] = "full_document_refused"
            return AnswerResult(
                answer=refusals.message(refusals.FULL_DOCUMENT_REFUSAL, language),
                language=language,
                kind=KIND_CAPABILITY_REFUSED,
                debug=debug,
            )

        if _GREETING_RE.match(question):
            debug["route"] = "greeting"
            return AnswerResult(
                answer=refusals.message(refusals.GREETING, language),
                language=language,
                kind=KIND_GREETING,
                debug=debug,
            )

        # Resolve follow-ups and add an English search query when needed.
        resolved = resolve_query(question, history, language, self.client)
        language = resolved.response_language
        debug["language"] = language
        debug["resolved_query"] = {
            "standalone": resolved.standalone,
            "english": resolved.english,
            "response_language": resolved.response_language,
            "depends_on_history": resolved.depends_on_history,
            "rewritten": resolved.rewritten,
            "used_llm": resolved.used_llm,
            "note": resolved.note,
        }

        # A vague follow-up can become an explicit full-document request only
        # after reference resolution, so enforce the same boundary again.
        if _FULL_DOCUMENT_REQUEST_RE.search(resolved.standalone):
            debug["route"] = "resolved_full_document_refused"
            return AnswerResult(
                answer=refusals.message(refusals.FULL_DOCUMENT_REFUSAL, language),
                language=language,
                kind=KIND_CAPABILITY_REFUSED,
                debug=debug,
            )

        # Retrieve fresh evidence for every turn.
        retrieval = self.retriever.retrieve(
            resolved.standalone, expansions=resolved.expansions(), top_k=top_k
        )
        debug["retrieval"] = {
            "best_score": round(retrieval.best_score, 4),
            "gate": settings.min_similarity,
            "passed_gate": retrieval.passed_gate,
            "chunk_count": len(retrieval.chunks),
            "notes": retrieval.notes,
        }

        # Refuse when retrieval found no strong evidence.
        if retrieval.is_empty or not retrieval.passed_gate:
            debug["route"] = "gated_not_found"
            debug["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            return AnswerResult(
                answer=refusals.message(refusals.NOT_FOUND, language),
                language=language,
                kind=KIND_NOT_FOUND,
                retrieval=retrieval,
                debug=debug,
            )

        # Detect conflicting values before generation.
        report = detect_conflicts(retrieval.chunks)
        notice = build_conflict_notice(report, language)
        debug["conflicts"] = report.summary()

        # Generate from the retrieved context.
        context_block, sanitised = format_context_block(retrieval.chunks, language)
        debug["context_sanitised"] = sanitised
        user_message = build_user_message(
            question=resolved.standalone,
            context_block=context_block,
            language=language,
            conflict_notice=notice,
        )

        try:
            response = self._generate(language, user_message)
        except LLMUnavailable as exc:
            debug["route"] = "llm_unavailable"
            debug["error"] = str(exc)
            return AnswerResult(
                answer=refusals.message(
                    refusals.LLM_UNAVAILABLE,
                    language,
                    provider=getattr(self.client, "provider", "configured provider"),
                    model=self.client.model,
                    detail=str(exc),
                ),
                language=language,
                kind=KIND_ERROR,
                retrieval=retrieval,
                conflict=report,
                debug=debug,
            )
        except LLMError as exc:
            debug["route"] = "llm_error"
            debug["error"] = str(exc)
            return AnswerResult(
                answer=refusals.message(refusals.LLM_ERROR, language, detail=str(exc)),
                language=language,
                kind=KIND_ERROR,
                retrieval=retrieval,
                conflict=report,
                debug=debug,
            )

        raw = response.text.strip()
        debug["llm"] = {
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "duration_ms": response.duration_ms,
        }
        debug["raw_answer"] = raw

        # Convert the model sentinel into a localized refusal.
        if self._is_refusal_sentinel(raw):
            debug["route"] = "model_refused"
            debug["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            return AnswerResult(
                answer=refusals.message(refusals.NOT_FOUND, language),
                language=language,
                kind=KIND_NOT_FOUND,
                retrieval=retrieval,
                conflict=report,
                debug=debug,
            )

        # Validate citations before returning the answer.
        final_text, citations, route = self._verify_citations(
            raw, retrieval, language, user_message
        )
        debug["route"] = route
        debug["elapsed_ms"] = int((time.perf_counter() - started) * 1000)

        if route == "rejected_fabricated_citations":
            return AnswerResult(
                answer=refusals.message(refusals.UNCITED_ANSWER, language),
                language=language,
                kind=KIND_NOT_FOUND,
                retrieval=retrieval,
                conflict=report,
                debug=debug,
            )

        return AnswerResult(
            answer=final_text,
            language=language,
            kind=KIND_ANSWER,
            citations=citations,
            retrieval=retrieval,
            conflict=report,
            debug=debug,
        )

    def _generate(self, language: str, user_message: str) -> LLMResponse:
        """Generate from the standalone question and retrieved context."""
        return self.client.chat(
            system=system_prompt(language),
            messages=[{"role": "user", "content": user_message}],
        )

    @staticmethod
    def _is_refusal_sentinel(text: str) -> bool:
        if not text:
            return True
        if INSUFFICIENT_EVIDENCE in text:
            # Ignore incidental mentions of the sentinel in longer text.
            stripped = text.replace(INSUFFICIENT_EVIDENCE, "").strip(" \n\t.。،,:!\"'*_-")
            return len(stripped) < 40
        return False

    def _verify_citations(
        self,
        raw: str,
        retrieval: RetrievalResult,
        language: str,
        user_message: str,
    ) -> Tuple[str, List[Citation], str]:
        """Validate citations and retry once when citations are missing."""
        if not settings.enforce_citation_validation:
            return raw, parse_citations(raw), "unvalidated"

        allowed = retrieval.allowed_citations()
        allowed_sections = retrieval.allowed_citation_sections()
        known_documents = retrieval.documents()
        attempted = parse_citations(raw)

        if not attempted:
            retried = self._retry_for_citations(language, user_message, raw)
            if retried is not None:
                outcome = validate_and_clean(
                    retried, allowed, known_documents, allowed_sections
                )
                if outcome.has_valid_citation:
                    return outcome.text, outcome.valid, "cited_after_retry"
            return raw, [], "rejected_fabricated_citations"

        outcome = validate_and_clean(raw, allowed, known_documents, allowed_sections)
        if outcome.has_valid_citation:
            route = "verified" if not outcome.invalid else "verified_after_stripping"
            return outcome.text, outcome.valid, route

        return outcome.text, [], "rejected_fabricated_citations"

    def _retry_for_citations(
        self, language: str, user_message: str, previous: str
    ) -> str | None:
        reminder = (
            "أعد صياغة الإجابة نفسها بالعربية دون تغيير أي معلومة، مع إضافة إشارة المصدر مباشرةً "
            "بعد كل معلومة بالصيغة [اسم_الملف.pdf — الصفحة N]. استخدم فقط المستندات والصفحات "
            "الواردة في المقتطفات."
            if language == "ar"
            else "Rewrite the same answer without changing any fact, adding an inline citation "
            "immediately after each factual claim in the form [filename.pdf — Page N]. "
            "Use only documents and pages that appear in the excerpts."
        )
        try:
            response = self.client.chat(
                system=system_prompt(language),
                messages=[
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": previous},
                    {"role": "user", "content": reminder},
                ],
            )
        except LLMError:
            return None
        return response.text.strip() or None
