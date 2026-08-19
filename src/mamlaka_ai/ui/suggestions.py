"""Bilingual suggested questions for the empty conversation state."""

from __future__ import annotations

import random
from typing import Sequence


# Each pair represents one topic in English and Arabic. Selecting by topic
# prevents the same question from appearing twice in different languages.
SUGGESTED_QUESTION_POOL: Sequence[tuple[str, str]] = (
    ("What is the project's main goal?", "ما الهدف الرئيسي للمشروع؟"),
    ("Who is on the project team?", "من هم أعضاء فريق المشروع؟"),
    ("What is the project timeline?", "ما الجدول الزمني للمشروع؟"),
    ("What is the revised project budget?", "ما ميزانية المشروع المعدّلة؟"),
    ("What is the revised launch date?", "ما تاريخ الإطلاق المعدّل؟"),
    ("Which platforms does the project cover?", "ما المنصات التي يشملها المشروع؟"),
    ("How many languages will the mobile app support?", "كم لغة سيدعمها تطبيق الهاتف؟"),
    (
        "What are the approval rules for budget changes?",
        "ما قواعد الموافقة على تغييرات الميزانية؟",
    ),
    (
        "Can user data be shared with third parties?",
        "هل يمكن مشاركة بيانات المستخدمين مع أطراف ثالثة؟",
    ),
    ("What is the highest-risk timeline item?", "ما العنصر الأعلى خطورة في الجدول الزمني؟"),
    ("How will project success be measured?", "كيف سيُقاس نجاح المشروع؟"),
    ("How are technical issues escalated?", "كيف يتم تصعيد المشكلات التقنية؟"),
)


def choose_suggested_questions(
    count: int = 3,
    rng: random.Random | None = None,
) -> tuple[str, ...]:
    """Choose distinct topics while representing both interface languages.

    The optional random generator makes selection deterministic in tests. The
    normal UI call uses Python's process-level random generator.
    """
    if count < 2:
        raise ValueError("At least two suggestions are required for a bilingual selection.")
    if count > len(SUGGESTED_QUESTION_POOL):
        raise ValueError("Requested more suggestions than the available topic pool.")

    generator = rng or random
    topics = generator.sample(list(SUGGESTED_QUESTION_POOL), count)

    # Guarantee at least one question in each language; remaining cards vary.
    languages = ["en", "ar"] + [generator.choice(("en", "ar")) for _ in range(count - 2)]
    generator.shuffle(languages)
    return tuple(
        topic[0] if language == "en" else topic[1]
        for topic, language in zip(topics, languages)
    )
