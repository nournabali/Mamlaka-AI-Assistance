from __future__ import annotations

import os

import pytest

from mamlaka_ai.retrieval.retriever import load_retriever


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_EMBEDDING_TESTS") != "1",
        reason="set RUN_EMBEDDING_TESTS=1 to load the multilingual embedding model",
    ),
]


@pytest.fixture(scope="module")
def retriever():
    return load_retriever()


@pytest.mark.parametrize(
    ("question", "required_sections"),
    [
        ("What is the main goal of the Digital Expansion Initiative?", {"Project Goals"}),
        ("Who is the project lead?", {"Key Details", "Project Team"}),
        ("Which department is Sara Al-Rashid from?", {"Project Team"}),
        ("What is Lina Haddad's department?", {"Project Team"}),
        ("Which department is Omar Fayez from?", {"Project Team"}),
        ("What is the project's budget?", {"Key Details", "Budget Notes"}),
        ("When will the project launch?", {"Key Details", "Revised Launch Date"}),
        ("Who approves a budget change greater than $50,000?", {"Governance Structure"}),
        ("How many languages will the mobile app support?", {"Platform Scope"}),
        ("Can user data be shared with third parties?", {"Data Privacy Policy"}),
        ("What is the highest-risk item on the project timeline?", {"Risk Notes"}),
    ],
)
def test_required_evidence_is_retrieved(retriever, question, required_sections) -> None:
    result = retriever.retrieve(question)
    sections = {item.chunk.section for item in result.chunks}
    assert result.passed_gate
    assert required_sections <= sections


def test_arabic_query_retrieves_platform_scope(retriever) -> None:
    result = retriever.retrieve(
        "كم عدد اللغات التي سيدعمها تطبيق الهاتف؟",
        expansions=["How many languages will the mobile app support?"],
    )
    assert result.passed_gate
    assert "Platform Scope" in {item.chunk.section for item in result.chunks}


@pytest.mark.parametrize(
    ("question", "english"),
    [
        ("من هو مدير المشروع؟", "Who is the project manager?"),
        ("ما هو قسم لينا حداد؟", "What is Lina Haddad's department?"),
        ("في أي قسم يعمل عمر فايز؟", "Which department is Omar Fayez from?"),
    ],
)
def test_arabic_team_queries_retrieve_english_team_evidence(
    retriever, question, english
) -> None:
    result = retriever.retrieve(question, expansions=[english])
    sections = {item.chunk.section for item in result.chunks}
    assert result.passed_gate
    assert {"Project Team", "Key Details"} & sections
