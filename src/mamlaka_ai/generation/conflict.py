"""Detect conflicting values and revisions in retrieved chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from mamlaka_ai.utils.claims import labelled_values, matched_revision_cues, revision_score

# Minimum score for revision language.
REVISION_THRESHOLD = 3

@dataclass
class ClaimInstance:
    """A value asserted by one chunk."""

    normalised: str
    surface: str
    document_name: str
    page_number: int
    section: str
    chunk_id: str
    revision_score: int
    label_text: str = ""
    revision_cues: List[str] = field(default_factory=list)

    @property
    def is_revision(self) -> bool:
        return self.revision_score >= REVISION_THRESHOLD

    def citation(self, language: str = "en") -> str:
        page_word = "الصفحة" if language == "ar" else "Page"
        return f"[{self.document_name} — {page_word} {self.page_number} — {self.section}]"


@dataclass
class ConflictGroup:
    topic: str  # corpus-derived normalised claim label; retained for API stability
    value_type: str
    # Normalized value mapped to supporting chunks.
    variants: Dict[str, List[ClaimInstance]] = field(default_factory=dict)

    @property
    def distinct_count(self) -> int:
        return len(self.variants)

    @property
    def is_conflict(self) -> bool:
        sources = {
            claim.chunk_id
            for claims in self.variants.values()
            for claim in claims
        }
        # Values must come from separate chunks to form a conflict.
        return self.distinct_count >= 2 and len(sources) >= 2

    def revised_variants(self) -> List[str]:
        return [
            value
            for value, claims in self.variants.items()
            if any(claim.is_revision for claim in claims)
        ]

    def original_variants(self) -> List[str]:
        revised = set(self.revised_variants())
        return [value for value in self.variants if value not in revised]

    def topic_label(self, language: str = "en") -> str:
        labels = [
            claim.label_text
            for claims in self.variants.values()
            for claim in claims
            if claim.label_text
        ]
        return min(labels, key=len) if labels else self.topic


@dataclass
class ConflictReport:
    groups: List[ConflictGroup] = field(default_factory=list)

    @property
    def has_conflict(self) -> bool:
        return any(group.is_conflict for group in self.groups)

    def conflicts(self) -> List[ConflictGroup]:
        return [group for group in self.groups if group.is_conflict]

    def summary(self) -> List[str]:
        """Return short conflict summaries for debugging."""
        lines: List[str] = []
        for group in self.conflicts():
            surfaces = [
                f"{claims[0].surface} ({claims[0].document_name} p{claims[0].page_number}"
                f"{', revised' if claims[0].is_revision else ''})"
                for claims in group.variants.values()
            ]
            lines.append(f"{group.topic}: " + " vs ".join(surfaces))
        return lines


def detect_conflicts(scored_chunks: Sequence) -> ConflictReport:
    """Find competing values for the same claim."""
    groups: Dict[Tuple[str, str], ConflictGroup] = {}

    for scored in scored_chunks:
        chunk = getattr(scored, "chunk", scored)
        chunk_revision = revision_score(chunk.text, chunk.section)
        cues = matched_revision_cues(chunk.text)

        for labelled in labelled_values(chunk.text, chunk.section):
            if not labelled.claim_key:
                continue
            key = (labelled.claim_key, labelled.value.value_type)
            group = groups.setdefault(
                key, ConflictGroup(labelled.claim_key, labelled.value.value_type)
            )
            instance = ClaimInstance(
                normalised=labelled.value.normalised,
                surface=labelled.value.surface,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                revision_score=chunk_revision,
                label_text=labelled.label_text,
                revision_cues=cues,
            )
            existing = group.variants.setdefault(labelled.value.normalised, [])
            # Keep one assertion per document page and value.
            if not any(
                claim.document_name == instance.document_name
                and claim.page_number == instance.page_number
                for claim in existing
            ):
                existing.append(instance)

    return ConflictReport(groups=[g for g in groups.values() if g.is_conflict])


def _english_notice(report: ConflictReport) -> str:
    lines = [
        "<CONFLICT_NOTICE>",
        "The excerpts above contain incompatible values for the same item. This was detected by the "
        "application, not by you. Apply this notice only when the item directly answers the user's "
        "exact question; it never authorizes answering a related question instead. When relevant, "
        "you MUST report every value below, each with its own citation, and explain the relationship "
        "between them. Do not pick one silently.",
    ]
    for group in report.conflicts():
        lines.append("")
        lines.append(f"Item in conflict: {group.topic_label('en')}")
        for value, claims in group.variants.items():
            citations = " ".join(claim.citation("en") for claim in claims)
            marker = ""
            if claims[0].is_revision:
                cue = claims[0].revision_cues[0] if claims[0].revision_cues else "revision"
                marker = (
                    f'  <-- the source describes this as the REVISED value (wording: "{cue}")'
                )
            lines.append(f"  - {claims[0].surface}  {citations}{marker}")
        revised = group.revised_variants()
        original = group.original_variants()
        if revised and original:
            lines.append(
                f"  Guidance: state that {original[0]} is the earlier/originally stated value and "
                f"that {revised[0]} is explicitly described as the revised value that replaced it. "
                "Cite both."
            )
        else:
            lines.append(
                "  Guidance: no excerpt says which of these supersedes the other. State that the "
                "documents disagree and that the excerpts do not indicate which value is current. "
                "Cite both."
            )
    lines.append("</CONFLICT_NOTICE>")
    return "\n".join(lines)


def _arabic_notice(report: ConflictReport) -> str:
    lines = [
        "<CONFLICT_NOTICE>",
        "تحتوي المقتطفات أعلاه على قيم متعارضة للأمر نفسه. هذا التعارض رصده التطبيق وليس أنت، وهو "
        "يُطبَّق فقط إذا كان البند يجيب مباشرةً عن سؤال المستخدم المحدد، ولا يجيز أبداً الإجابة عن "
        "سؤال قريب بدلاً منه. وعندما يكون ذا صلة، يجب أن تذكر كل قيمة من القيم التالية مع مصدرها "
        "الخاص، وأن توضّح العلاقة بينها، ولا تختر إحداها بصمت.",
    ]
    for group in report.conflicts():
        lines.append("")
        lines.append(f"البند المتعارض: {group.topic_label('ar')}")
        for value, claims in group.variants.items():
            citations = " ".join(claim.citation("ar") for claim in claims)
            marker = ""
            if claims[0].is_revision:
                cue = claims[0].revision_cues[0] if claims[0].revision_cues else "revision"
                marker = f'  <-- يصف المصدر هذه القيمة بأنها القيمة المعدَّلة (بصيغة: "{cue}")'
            lines.append(f"  - {claims[0].surface}  {citations}{marker}")
        revised = group.revised_variants()
        original = group.original_variants()
        if revised and original:
            lines.append(
                f"  التوجيه: بيّن أن {original[0]} هي القيمة الأصلية أو الأسبق، وأن {revised[0]} "
                "موصوفة صراحةً بأنها القيمة المعدَّلة التي حلّت محلها. واذكر مصدر كل منهما."
            )
        else:
            lines.append(
                "  التوجيه: لا يوضّح أي مقتطف أي القيمتين تنسخ الأخرى. اذكر أن المستندات متعارضة "
                "وأن المقتطفات لا تبيّن أيّها القيمة السارية. واذكر مصدر كل منهما."
            )
    lines.append("</CONFLICT_NOTICE>")
    return "\n".join(lines)


def build_conflict_notice(report: ConflictReport, language: str = "en") -> str:
    """Build a localized conflict notice for the prompt."""
    if not report.has_conflict:
        return ""
    return _arabic_notice(report) if language == "ar" else _english_notice(report)
