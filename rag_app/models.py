"""Structured model outputs and graph state types."""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["clarify", "direct", "retrieve"] = Field(
        description="clarify: query is ambiguous and cannot be answered without "
        "more info. direct: greeting or small talk only. retrieve: any factual "
        "question that should be checked against the IRCTC knowledge base."
    )
    clarifying_question: str = Field(
        default="", description="Question to ask the user if route == clarify."
    )


class Decomposition(BaseModel):
    sub_queries: List[str] = Field(
        description="Self-contained, independent sub-queries. If the question is "
        "single-part, return exactly one."
    )


class GradeResult(BaseModel):
    relevant_ids: List[int] = Field(
        description="Indices of passages that help answer the query. Empty if none."
    )


class RewrittenQuery(BaseModel):
    query: str


class RetrievalState(TypedDict, total=False):
    sub_query: str
    original_query: str
    attempts: int
    bm25_docs: List[Document]
    dense_docs: List[Document]
    fused_docs: List[Document]
    good_docs: List[Document]
    answer: str


class SubAnswer(TypedDict):
    sub_query: str
    answer: str
    retries: int


class AgentState(TypedDict, total=False):
    question: str
    route: str
    clarifying_question: str
    clarification_count: int
    sub_queries: List[str]
    sub_answers: Annotated[List[SubAnswer], operator.add]
    answer: str


class SubQueryInput(TypedDict):
    sub_query: str
