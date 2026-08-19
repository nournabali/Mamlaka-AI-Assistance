from __future__ import annotations

import runpy

from mamlaka_ai.config import PROJECT_ROOT


EVALUATION_MODULE = runpy.run_path(str(PROJECT_ROOT / "scripts" / "run_evaluation.py"))
CASES = EVALUATION_MODULE["CASES"]
EVALUATION_CATEGORIES = EVALUATION_MODULE["EVALUATION_CATEGORIES"]
RATE_LIMIT_DELAY = EVALUATION_MODULE["_rate_limit_delay"]
AnswerResult = EVALUATION_MODULE["AnswerResult"]


def test_live_evaluation_covers_required_behavior_categories() -> None:
    case_names = {name for name, _, _ in CASES}
    minimum_counts = {
        "arabic": 7,
        "paraphrase": 3,
        "ambiguous_follow_up": 3,
        "adversarial": 4,
        "expected_refusal": 10,
    }

    assert set(EVALUATION_CATEGORIES) == set(minimum_counts)
    for category, minimum in minimum_counts.items():
        names = EVALUATION_CATEGORIES[category]
        assert len(names) >= minimum
        assert set(names) <= case_names


def test_live_evaluation_honors_provider_retry_delay() -> None:
    limited = AnswerResult(
        "Groq error: Rate limit reached. Please try again in 12.75s.",
        "en",
        kind="error",
    )
    assert RATE_LIMIT_DELAY(limited) == 12.75
    assert RATE_LIMIT_DELAY(AnswerResult("another error", "en", kind="error")) is None
