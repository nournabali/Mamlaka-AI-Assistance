from __future__ import annotations

import random

from mamlaka_ai.ui.suggestions import (
    SUGGESTED_QUESTION_POOL,
    choose_suggested_questions,
)
from mamlaka_ai.utils.language import detect_language


def test_suggestions_are_distinct_and_bilingual() -> None:
    for seed in range(30):
        questions = choose_suggested_questions(rng=random.Random(seed))
        languages = {detect_language(question) for question in questions}

        assert len(questions) == 3
        assert len(set(questions)) == 3
        assert languages == {"ar", "en"}


def test_suggestions_vary_across_conversations() -> None:
    selections = {
        choose_suggested_questions(rng=random.Random(seed))
        for seed in range(10)
    }

    assert len(SUGGESTED_QUESTION_POOL) >= 10
    assert len(selections) > 1
