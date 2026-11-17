"""Tests for orchestration and routing behavior."""

from rag_app.agents.orchestrator import (
    is_compound_question,
    is_small_talk,
    route_after_router,
)
from rag_app.agents.retrieval import route_after_grade, source_label


def test_router_routes_clarify_and_direct_explicitly():
    assert route_after_router({"route": "clarify"}) == "clarify"
    assert route_after_router({"route": "direct"}) == "direct"


def test_router_defaults_retrieval_decision_to_decomposition():
    assert route_after_router({"route": "retrieve"}) == "decompose"


def test_only_greetings_and_small_talk_are_direct():
    assert is_small_talk("Hello!")
    assert is_small_talk("How are you?")
    assert not is_small_talk("What is the Tatkal booking period?")


def test_simple_question_stays_single_but_compound_question_can_be_split():
    assert not is_compound_question("When does Tatkal booking open?")
    assert is_compound_question(
        "When does Tatkal booking open for AC and Non-AC classes?"
    )


def test_pdf_source_label_uses_one_based_page_number():
    assert source_label({"source": r"D:\AI\docs\tatkal_faq.pdf", "page": 1}) == (
        "tatkal_faq.pdf, p. 2"
    )


def test_retrieval_retries_only_when_no_relevant_documents_remain():
    assert route_after_grade({"good_docs": [], "attempts": 0}, 2) == "rewrite"
    assert route_after_grade({"good_docs": [], "attempts": 2}, 2) == "answer"
    assert route_after_grade({"good_docs": [object()], "attempts": 0}, 2) == "answer"
