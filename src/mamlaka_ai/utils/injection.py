"""Screen user input and neutralize instructions in retrieved text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Pattern, Tuple

Verdict = Literal["clean", "injection", "forced_guess"]


@dataclass
class InjectionCheck:
    verdict: Verdict
    matched: List[str]

    @property
    def is_blocked(self) -> bool:
        return self.verdict != "clean"


# User input patterns.

_INJECTION_PATTERNS: Tuple[str, ...] = (
    # Instruction overrides.
    r"\bignore\s+(?:all\s+|any\s+|your\s+|the\s+)?(?:previous|prior|above|earlier|preceding|foregoing)\b",
    r"\bignore\s+(?:all\s+)?(?:your\s+)?(?:instructions|rules|guidelines|prompt|directives)\b",
    r"\bdisregard\s+(?:all\s+|any\s+|your\s+|the\s+)?(?:previous|prior|above|instructions|rules|guidelines)\b",
    r"\bforget\s+(?:everything|all|your|the)\b.{0,40}\b(?:instructions|rules|pdfs?|documents|context|training)\b",
    r"\bforget\s+(?:the\s+)?(?:pdfs?|documents|files|sources)\b",
    r"\boverride\s+(?:your\s+)?(?:instructions|rules|system|prompt|restrictions)\b",
    r"\b(?:new|updated)\s+(?:instructions|system\s+prompt)\s*:",
    r"\bstop\s+(?:following|obeying)\b",
    # Persona changes and jailbreaks.
    r"\bact\s+as\s+(?:if\s+you\s+are\s+)?(?:chatgpt|gpt|dan|an?\s+unrestricted|an?\s+unfiltered|a\s+jailbro?ken)",
    r"\byou\s+are\s+(?:now\s+)?(?:chatgpt|dan|an?\s+unrestricted|an?\s+unfiltered|jailbro?ken|free\s+of)",
    r"\bpretend\s+(?:that\s+)?you\s+(?:are|have)\b",
    r"\broleplay\s+as\b",
    r"\bdeveloper\s+mode\b",
    r"\bjail\s?break\b",
    r"\bwith\s+no\s+(?:restrictions|rules|filters|limits|guardrails)\b",
    r"\bwithout\s+(?:any\s+)?(?:restrictions|rules|filters|limits|guardrails)\b",
    # Prompt and configuration extraction.
    r"\b(?:tell|show|reveal|print|repeat|output|give|display|dump)\b.{0,30}\b(?:your\s+)?"
    r"(?:system\s+prompt|initial\s+prompt|instructions|prompt\s+template|guidelines|configuration)\b",
    r"\bwhat\s+(?:are|were)\s+your\s+(?:instructions|rules|system\s+prompt)\b",
    r"\brepeat\s+(?:everything|the\s+text)\s+above\b",
    # Knowledge-source overrides.
    r"\b(?:answer|respond|reply)\b.{0,40}\bfrom\s+your\s+own\s+(?:knowledge|training|memory)\b",
    r"\buse\s+your\s+(?:own\s+)?(?:general\s+|pretrained\s+|internal\s+)?(?:knowledge|training\s+data)\b",
    r"\b(?:without|don'?t|do\s+not)\s+(?:using|use|consult)\b.{0,30}\b(?:documents?|pdfs?|sources?|context)\b",
    r"\bnot\s+(?:limited|restricted|bound)\s+(?:to|by)\b.{0,30}\b(?:documents?|pdfs?)\b",
    # Arabic equivalents
    r"تجاهل\s+(?:كل\s+)?(?:التعليمات|الأوامر|ما\s+سبق|التوجيهات)",
    r"انس(?:َ|ى)?\s+(?:التعليمات|المستندات|الملفات|كل\s+شيء)",
    r"تصرف\s+كأنك",
    r"أنت\s+الآن\s+(?:شات|نموذج|بدون)",
    r"(?:اظهر|أظهر|اكتب|اعرض)\s+(?:لي\s+)?(?:موجه|تعليمات)\s*(?:النظام|النظام)",
    r"ما\s+(?:هي\s+)?تعليمات(?:ك|\s+النظام)",
    r"دون\s+(?:أي\s+)?(?:قيود|شروط)",
    r"من\s+(?:معرفتك|معلوماتك)\s+(?:العامة|الخاصة)",
    r"تخل(?:ص|ي)\s+من\s+القيود",
)

# Requests for unsupported guesses use a separate response.
_FORCED_GUESS_PATTERNS: Tuple[str, ...] = (
    r"\b(?:best|educated|rough)\s+guess\b",
    r"\bguess\s+(?:anyway|it|the\s+answer)\b",
    r"\bjust\s+guess\b",
    r"\btake\s+a\s+guess\b",
    r"\bmake\s+(?:something|it)\s+up\b",
    r"\bmake\s+up\s+(?:an?\s+)?(?:answer|value|number|date)\b",
    r"\b(?:even|but)\s+(?:if|though)\s+(?:it'?s|it\s+is|the\s+answer\s+is)?\s*"
    r"(?:not|isn'?t)\s+in\s+the\s+(?:pdfs?|documents?)\b",
    r"\b(?:isn'?t|is\s+not|not)\s+in\s+the\s+(?:pdfs?|documents?)\b.{0,60}\b"
    r"(?:guess|answer|try|estimate|assume)\b",
    r"\b(?:speculate|extrapolate|invent|fabricate)\b",
    r"\bapproximate\s+it\s+anyway\b",
    r"\bwhat\s+do\s+you\s+think\s+it\s+(?:might|would|could)\s+be\b",
    r"\bif\s+you\s+had\s+to\s+guess\b",
    r"\banyway\b.{0,20}\bguess\b",
    # Arabic
    r"خمّ?ن",
    r"احزر",
    r"(?:اعط|أعط)ني\s+تقدير",
    r"حتى\s+لو\s+لم\s+(?:يكن|تكن)\s+(?:موجود|في\s+المستندات)",
    r"من\s+(?:عندك|تلقاء\s+نفسك)",
    r"افترض\s+(?:قيمة|رقم|جواب)",
)

_COMPILED_INJECTION: List[Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS
]
_COMPILED_FORCED_GUESS: List[Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in _FORCED_GUESS_PATTERNS
]


def _normalise(text: str) -> str:
    """Normalize simple obfuscation before matching."""
    text = re.sub(r"[​-‏‪-‮﻿]", "", text or "")
    text = re.sub(r"[_*`~]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def check_user_input(text: str) -> InjectionCheck:
    """Classify a user message before retrieval."""
    normalised = _normalise(text)
    if not normalised.strip():
        return InjectionCheck("clean", [])

    injection_hits = [m.group(0) for p in _COMPILED_INJECTION if (m := p.search(normalised))]
    if injection_hits:
        return InjectionCheck("injection", injection_hits)

    guess_hits = [m.group(0) for p in _COMPILED_FORCED_GUESS if (m := p.search(normalised))]
    if guess_hits:
        return InjectionCheck("forced_guess", guess_hits)

    return InjectionCheck("clean", [])


# Retrieved document patterns.

# Neutralize instructions aimed at an AI reader.
_DOC_INSTRUCTION_PATTERNS: Tuple[str, ...] = (
    r"\bignore\s+(?:all\s+|any\s+|your\s+|the\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instructions?|prompts?|rules?)\b",
    r"\bdisregard\s+(?:all\s+|any\s+|your\s+|the\s+)?(?:previous|prior|above)\b",
    r"\byou\s+(?:are|must|should)\s+now\s+\w+",
    r"\bsystem\s*(?:prompt|message)\s*:",
    r"\b(?:assistant|ai|model|chatbot)\s*(?:instruction|directive)s?\s*:",
    r"\bnew\s+instructions?\s*:",
    r"\bact\s+as\b",
    r"\breveal\s+your\b",
    r"تجاهل\s+التعليمات",
    r"تعليمات\s+النظام\s*:",
)
_COMPILED_DOC_INSTRUCTION = [re.compile(p, re.IGNORECASE) for p in _DOC_INSTRUCTION_PATTERNS]

_NEUTRALISED_MARK = "[instruction-like text in source neutralised]"


def neutralise_document_text(text: str) -> Tuple[str, bool]:
    """Return sanitized document text and whether it changed."""
    changed = False
    cleaned = text or ""
    for pattern in _COMPILED_DOC_INSTRUCTION:
        cleaned, count = pattern.subn(_NEUTRALISED_MARK, cleaned)
        if count:
            changed = True
    # Prevent a passage from closing the context delimiter early.
    for token in ("</DOCUMENT_CONTEXT>", "<DOCUMENT_CONTEXT>", "</EVIDENCE>", "<EVIDENCE>"):
        if token in cleaned:
            cleaned = cleaned.replace(token, token.replace("<", "(").replace(">", ")"))
            changed = True
    return cleaned, changed
