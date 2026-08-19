"""Extract typed values and detect revision language."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

# Value extraction.

_MONEY_RE = re.compile(
    r"(?:US\s*\$|\$|USD\s*)\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+(?:\.[0-9]+)?)\s*"
    r"(million|m\b|billion|bn\b|k\b|thousand|مليون|ألف)?",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    # Common Levantine and MSA Arabic month names.
    "يناير": 1, "كانون الثاني": 1, "فبراير": 2, "شباط": 2, "مارس": 3, "آذار": 3,
    "أبريل": 4, "نيسان": 4, "مايو": 5, "أيار": 5, "يونيو": 6, "حزيران": 6,
    "يوليو": 7, "تموز": 7, "أغسطس": 8, "آب": 8, "سبتمبر": 9, "أيلول": 9,
    "أكتوبر": 10, "تشرين الأول": 10, "نوفمبر": 11, "تشرين الثاني": 11,
    "ديسمبر": 12, "كانون الأول": 12,
}

_MONTH_ALTERNATION = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))

# English and Arabic dates in month-first or day-first order.
_DATE_MDY_RE = re.compile(
    rf"\b({_MONTH_ALTERNATION})\s+([0-9]{{1,2}})(?:\s*[-–]\s*[0-9]{{1,2}})?\s*,?\s*([0-9]{{4}})\b",
    re.IGNORECASE,
)
_DATE_DMY_RE = re.compile(
    rf"\b([0-9]{{1,2}})\s+({_MONTH_ALTERNATION})\s*,?\s*([0-9]{{4}})\b",
    re.IGNORECASE,
)

_PERCENT_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:%|percent|بالمئة|في المئة|بالمائة)")

_MULTIPLIERS = {
    "million": 1_000_000, "m": 1_000_000, "مليون": 1_000_000,
    "billion": 1_000_000_000, "bn": 1_000_000_000,
    "k": 1_000, "thousand": 1_000, "ألف": 1_000,
}

VALUE_TYPES = ("money", "date", "percent")


@dataclass(frozen=True)
class Value:
    """A single normalised factual value found in a passage."""

    value_type: str  # money | date | percent
    normalised: str  # canonical form, used for equality comparisons
    surface: str  # exactly as it appeared in the document

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.value_type}:{self.normalised}"


def _normalise_money(amount: str, suffix: str | None) -> str:
    try:
        number = float(amount.replace(",", ""))
    except ValueError:  # pragma: no cover - regex guarantees numeric
        return amount
    if suffix:
        number *= _MULTIPLIERS.get(suffix.strip().lower().rstrip("."), 1)
    return f"USD {number:.2f}".rstrip("0").rstrip(".") if number % 1 else f"USD {int(number)}"


def _normalise_date(month_token: str, day: str, year: str) -> str:
    month = _MONTHS.get(month_token.strip().lower())
    if month is None:
        return f"{year}-??-{int(day):02d}"
    return f"{year}-{month:02d}-{int(day):02d}"


def extract_values(text: str) -> List[Value]:
    """All money / date / percent values in ``text``, de-duplicated, in order."""
    found: List[Value] = []
    seen: Set[str] = set()

    def _add(value: Value) -> None:
        key = f"{value.value_type}|{value.normalised}"
        if key not in seen:
            seen.add(key)
            found.append(value)

    for match in _MONEY_RE.finditer(text):
        _add(
            Value(
                "money",
                _normalise_money(match.group(1), match.group(2)),
                match.group(0).strip(),
            )
        )
    for match in _DATE_MDY_RE.finditer(text):
        _add(
            Value(
                "date",
                _normalise_date(match.group(1), match.group(2), match.group(3)),
                match.group(0).strip(),
            )
        )
    for match in _DATE_DMY_RE.finditer(text):
        _add(
            Value(
                "date",
                _normalise_date(match.group(2), match.group(1), match.group(3)),
                match.group(0).strip(),
            )
        )
    for match in _PERCENT_RE.finditer(text):
        _add(Value("percent", f"{float(match.group(1)):g}%", match.group(0).strip()))
    return found


def value_types_in(text: str) -> Set[str]:
    return {value.value_type for value in extract_values(text)}


# Revision detection.

# Stronger revision phrases receive higher weights.
_REVISION_PATTERNS: Dict[str, int] = {
    r"\brevis(?:ed|ion|ing)\b": 3,
    r"\bwas (?:moved|changed|shifted|pushed) to\b": 3,
    r"\bwas adjusted to\b": 3,
    r"\b(?:adjusted|amended|updated|superseded) to\b": 3,
    r"\boriginally (?:approved|planned|scheduled|budgeted)\b": 3,
    r"\bfollowing\b.{0,60}\b(?:revision|review|meeting|decision|approval)\b": 2,
    r"\bnew\b.{0,30}\bdate\b": 2,
    r"\bsupersedes?\b": 3,
    r"\bno longer\b": 1,
    r"\binstead of\b": 1,
    r"\bpreviously\b": 1,
    r"\bmoved to\b": 2,
    # Arabic
    r"تم\s+(?:تعديل|تحديث|نقل|تأجيل)": 3,
    r"مراجعة": 2,
    r"المعدّل|المعدل|المنقّح|المنقح": 3,
    r"بدلاً من|بدلا من": 1,
    r"في الأصل|أصلاً|أصلا": 2,
}

_COMPILED_REVISION = [(re.compile(p, re.IGNORECASE), w) for p, w in _REVISION_PATTERNS.items()]

# Revision-oriented headings may restate an earlier value.
_REVISION_SECTION_RE = re.compile(
    r"revis|updat|amend|chang|note|addend|correct|معدّل|معدل|منقح|ملاحظات", re.IGNORECASE
)


def revision_score(text: str, section: str = "") -> int:
    """Return the strength of revision language in a passage."""
    score = sum(weight for pattern, weight in _COMPILED_REVISION if pattern.search(text))
    if section and _REVISION_SECTION_RE.search(section):
        score += 2
    return score


def has_revision_language(text: str, section: str = "", threshold: int = 3) -> bool:
    return revision_score(text, section) >= threshold


def matched_revision_cues(text: str) -> List[str]:
    """Return phrases that contributed to the revision score."""
    cues: List[str] = []
    for pattern, _ in _COMPILED_REVISION:
        match = pattern.search(text)
        if match:
            cues.append(match.group(0))
    return cues


@dataclass(frozen=True)
class LabelledValue:
    """A typed value with a label derived from its source text."""

    value: Value
    label_text: str
    claim_key: str


_LABEL_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "been", "being", "by", "for", "from",
    "following", "in", "is", "it", "its", "new", "of", "on", "or", "original",
    "originally", "overall", "planned", "previous", "previously", "project", "public",
    "approved", "revised", "the", "this", "to", "updated", "was", "were", "will", "with",
    "adjusted", "amended", "changed", "moved", "shifted", "pushed", "date", "value",
    "إلى", "الى", "أن", "إن", "او", "أو", "بعد", "تم", "تعديل", "تحديث", "تاريخ",
    "الجديد", "الجديدة", "الحالي", "الحالية", "المشروع", "العام", "العامة", "في", "من",
}
_LABEL_TOKEN_RE = re.compile(r"[A-Za-zء-ي][A-Za-zء-ي0-9_-]*")
_LABEL_CLAUSE_RE = re.compile(r"[,;؛]\s*")


def _normalise_label_token(token: str) -> str:
    token = token.lower().strip("_-")
    if token.isascii():
        if token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"
        elif token.endswith("ing") and len(token) > 6:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 5:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
    return token


_NORMALISED_LABEL_STOPWORDS = {
    _normalise_label_token(token) for token in _LABEL_STOPWORDS
}


def _claim_key(label: str) -> str:
    """Normalize words found in a claim label."""
    tokens: List[str] = []
    for surface in _LABEL_TOKEN_RE.findall(label or ""):
        token = _normalise_label_token(surface)
        if token and token not in _NORMALISED_LABEL_STOPWORDS and token not in tokens:
            tokens.append(token)
    return " ".join(sorted(tokens))


def _clean_label(label: str) -> str:
    label = re.sub(r"^\s*\d{1,2}\.\s*", "", label or "")
    return re.sub(r"\s+", " ", label).strip(" :-–—\n\t")


def _value_occurrences(text: str) -> List[tuple[int, Value]]:
    """Return each value and its starting offset in reading order."""
    occurrences: List[tuple[int, Value]] = []
    cursor = 0
    for value in extract_values(text):
        index = text.find(value.surface, cursor)
        if index == -1:
            index = text.find(value.surface)
        if index == -1:
            continue
        occurrences.append((index, value))
        cursor = max(cursor, index + len(value.surface))
    occurrences.sort(key=lambda pair: pair[0])
    return occurrences


def labelled_values(text: str, section: str = "") -> List[LabelledValue]:
    """Derive claim labels only from the source text and section."""
    if not text:
        return []

    occurrences = _value_occurrences(text)
    if not occurrences:
        return []

    out: List[LabelledValue] = []
    previous_end = 0
    for start, value in occurrences:
        line_start = text.rfind("\n", 0, start) + 1
        label = _clean_label(text[line_start:start])
        if label:
            label = _clean_label(_LABEL_CLAUSE_RE.split(label)[-1])
        key = _claim_key(label)

        if not key:
            label = _clean_label(text[previous_end:start])
            if label:
                label = _clean_label(_LABEL_CLAUSE_RE.split(label)[-1])
            key = _claim_key(label)
        if not key and section:
            label = _clean_label(section)
            key = _claim_key(label)

        out.append(LabelledValue(value, label, key))
        previous_end = start + len(value.surface)
    return out


def iter_value_surfaces(values: Iterable[Value]) -> List[str]:
    return [value.surface for value in values]
