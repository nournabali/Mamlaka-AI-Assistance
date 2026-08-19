from __future__ import annotations

from mamlaka_ai.generation.conflict import build_conflict_notice, detect_conflicts


def test_launch_date_conflict_marks_april_as_revision(indexed_chunks) -> None:
    report = detect_conflicts(
        [indexed_chunks[3], indexed_chunks[10], indexed_chunks[17]]
    )
    assert report.has_conflict
    group = next(
        group
        for group in report.conflicts()
        if set(group.variants) == {"2027-03-15", "2027-04-01"}
    )
    assert set(group.variants) == {"2027-03-15", "2027-04-01"}
    assert group.revised_variants() == ["2027-04-01"]
    notice = build_conflict_notice(report, "en")
    assert "April 1, 2027" in notice
    assert "REVISED value" in notice
    assert "exact question" in notice
    assert "never authorizes answering a related question" in notice


def test_budget_conflict_marks_2_6m_as_revision(indexed_chunks) -> None:
    report = detect_conflicts(
        [indexed_chunks[3], indexed_chunks[15], indexed_chunks[16]]
    )
    group = next(
        group
        for group in report.conflicts()
        if set(group.variants) == {"USD 2400000", "USD 2600000"}
    )
    assert set(group.variants) == {"USD 2400000", "USD 2600000"}
    assert group.revised_variants() == ["USD 2600000"]


def test_budget_threshold_is_not_confused_with_total_budget(indexed_chunks) -> None:
    report = detect_conflicts(
        [indexed_chunks[9], indexed_chunks[15], indexed_chunks[16]]
    )
    assert not report.has_conflict
