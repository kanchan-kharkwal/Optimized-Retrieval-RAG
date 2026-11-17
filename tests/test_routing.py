"""Tests for orchestration and routing behavior."""

from rag_app.agents.orchestrator import route_after_router
from rag_app.agents.retrieval import route_after_grade


def test_router_routes_clarify_and_direct_explicitly():
    assert route_after_router({"route": "clarify"}) == "clarify"
    assert route_after_router({"route": "direct"}) == "direct"


def test_router_defaults_retrieval_decision_to_decomposition():
    assert route_after_router({"route": "retrieve"}) == "decompose"


def test_retrieval_retries_only_when_no_relevant_documents_remain():
    assert route_after_grade({"good_docs": [], "attempts": 0}, 2) == "rewrite"
    assert route_after_grade({"good_docs": [], "attempts": 2}, 2) == "answer"
    assert route_after_grade({"good_docs": [object()], "attempts": 0}, 2) == "answer"
