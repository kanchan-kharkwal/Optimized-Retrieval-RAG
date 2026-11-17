"""Retrieval subgraph and its node functions."""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import END, START, StateGraph

from rag_app.config import Settings
from rag_app.models import GradeResult, RetrievalState, RewrittenQuery
from rag_app.retrieval import Retrievers, reciprocal_rank_fusion


def source_label(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source", "?")).replace("\\", "/").rsplit("/", 1)[-1]
    page = metadata.get("page")
    if isinstance(page, int):
        return f"{source}, p. {page + 1}"
    return source


def route_after_grade(
    state: RetrievalState, max_retries: int
) -> Literal["rewrite", "answer"]:
    if state["good_docs"]:
        return "answer"
    if state.get("attempts", 0) >= max_retries:
        return "answer"
    return "rewrite"


def build_retrieval_graph(
    llm: BaseChatModel, retrievers: Retrievers, settings: Settings
) -> CompiledStateGraph[RetrievalState, Any, Any, Any]:
    def bm25_node(state: RetrievalState):
        return {"bm25_docs": retrievers.bm25.invoke(state["sub_query"])}

    def dense_node(state: RetrievalState):
        return {"dense_docs": retrievers.dense.invoke(state["sub_query"])}

    def fuse_node(state: RetrievalState):
        fused = reciprocal_rank_fusion(
            [state["bm25_docs"], state["dense_docs"]],
            k=settings.rrf_k,
            top_n=settings.fused_top_n,
        )
        return {"fused_docs": fused}

    def grade_node(state: RetrievalState):
        docs = state["fused_docs"]
        numbered = "\n\n".join(
            f"[{i}] {doc.page_content}" for i, doc in enumerate(docs)
        )
        grader = llm.with_structured_output(GradeResult)
        result = grader.invoke(
            [
                SystemMessage(
                    content="You grade retrieved passages. Return the indices of "
                    "passages that contain information useful for answering "
                    "the query. Be strict."
                ),
                HumanMessage(
                    content=f"Query: {state['sub_query']}\n\nPassages:\n{numbered}"
                ),
            ]
        )
        good = [docs[i] for i in result.relevant_ids if 0 <= i < len(docs)]
        return {"good_docs": good}

    def rewrite_node(state: RetrievalState):
        rewriter = llm.with_structured_output(RewrittenQuery)
        result = rewriter.invoke(
            [
                SystemMessage(
                    content="Rewrite the search query to retrieve better results. "
                    "Use different keywords, expand acronyms, add context. "
                    "Do not change the intent."
                ),
                HumanMessage(
                    content=f"Original: {state['original_query']}\n"
                    f"Last attempt: {state['sub_query']}"
                ),
            ]
        )
        return {
            "sub_query": result.query,
            "attempts": state.get("attempts", 0) + 1,
        }

    def answer_node(state: RetrievalState):
        docs = state.get("good_docs", [])
        if not docs:
            return {"answer": "No relevant information found in the knowledge base."}
        context = "\n\n".join(
            f"[{source_label(doc.metadata)}] {doc.page_content}"
            for doc in docs
        )
        message = llm.invoke(
            [
                SystemMessage(
                    content="You are an IRCTC information assistant. Answer the "
                    "user using only facts explicitly stated in the provided "
                    "IRCTC document excerpts. Do not use prior knowledge, make "
                    "assumptions, or add policy details that are absent from "
                    "the excerpts. Answer only what the user asked; do not "
                    "summarize unrelated questions or topics found in the "
                    "excerpts. Keep the answer to at most three sentences. Do "
                    "not write citations; the application will add source "
                    "citations. If the excerpts do not answer the question, say "
                    "that you could not find the information in the provided "
                    "IRCTC documents."
                ),
                HumanMessage(
                    content=f"Question: {state['original_query']}\n\n"
                    f"Context:\n{context}"
                ),
            ]
        )
        answer = re.sub(r"\[[^\]\r\n]{1,80}\]", "", message.content).strip()
        sources = sorted({source_label(doc.metadata) for doc in docs})
        references = "\n".join(f"- {source}" for source in sources)
        return {"answer": f"{answer}\n\nSources:\n{references}"}

    graph = StateGraph(RetrievalState)
    graph.add_node("bm25", bm25_node)
    graph.add_node("dense", dense_node)
    graph.add_node("fuse", fuse_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "bm25")
    graph.add_edge(START, "dense")
    graph.add_edge(["bm25", "dense"], "fuse")
    graph.add_edge("fuse", "grade")
    graph.add_conditional_edges(
        "grade",
        lambda state: route_after_grade(state, settings.max_retries),
        {"rewrite": "rewrite", "answer": "answer"},
    )
    graph.add_edge("rewrite", "bm25")
    graph.add_edge("rewrite", "dense")
    graph.add_edge("answer", END)
    return graph.compile()
