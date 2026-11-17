"""Command-line entry point for the RAG application."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from rag_app.agents.orchestrator import build_graph
from rag_app.agents.retrieval import build_retrieval_graph
from rag_app.config import Settings
from rag_app.models import AgentState
from rag_app.retrieval import build_retrievers


AgentGraph = CompiledStateGraph[AgentState, Any, Any, Any]


def build_application(settings: Optional[Settings] = None) -> AgentGraph:
    settings = settings or Settings.from_env()
    llm = ChatOpenAI(model=settings.llm_model, temperature=0)
    retrievers = build_retrievers(settings)
    retrieval_graph = build_retrieval_graph(llm, retrievers, settings)
    return build_graph(llm, retrieval_graph, settings)


_default_graph = None


def get_default_graph() -> AgentGraph:
    global _default_graph
    if _default_graph is None:
        _default_graph = build_application()
    return _default_graph


def ask(
    question: str, thread_id: str = "default", graph: Optional[AgentGraph] = None
) -> str:
    active_graph = graph or get_default_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = active_graph.invoke(
        {"question": question, "sub_answers": []}, config
    )

    while "__interrupt__" in result:
        prompt = result["__interrupt__"][0].value["question"]
        reply = input(f"\n[Clarification needed] {prompt}\n> ")
        result = active_graph.invoke(Command(resume=reply), config)

    return result["answer"]


def main() -> None:
    graph = build_application()
    print("Agentic RAG ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(
            "\nAssistant:",
            ask(question, thread_id=str(uuid.uuid4()), graph=graph),
            "\n",
        )
