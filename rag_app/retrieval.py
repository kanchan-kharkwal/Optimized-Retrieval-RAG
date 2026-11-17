"""Retriever construction, indexing, and rank fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import OllamaEmbeddings

from rag_app.config import Settings
from rag_app.ingestion import load_documents


@dataclass(frozen=True)
class Retrievers:
    bm25: BM25Retriever
    dense: BaseRetriever


def build_retrievers(settings: Settings) -> Retrievers:
    chunks = load_documents(settings.docs_dir)
    bm25 = BM25Retriever.from_documents(chunks, k=settings.top_k)
    vectorstore = FAISS.from_documents(
        chunks,
        OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        ),
    )
    dense = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    return Retrievers(bm25=bm25, dense=dense)


def reciprocal_rank_fusion(
    rank_lists: List[List[Document]], k: int = 60, top_n: int = 6
) -> List[Document]:
    scores: dict[str, float] = {}
    by_key: dict[str, Document] = {}
    for ranked in rank_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            by_key[key] = doc
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [by_key[key] for key in ordered]
