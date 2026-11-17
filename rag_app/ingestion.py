"""Document loading and text splitting."""

from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SAMPLE_DOCS = [
    "LangGraph is a library for building stateful, multi-actor LLM applications "
    "as graphs. It supports cycles, persistence via checkpointers, and "
    "human-in-the-loop interrupts.",
    "BM25 is a sparse, keyword-based ranking function that scores documents by "
    "term frequency and inverse document frequency with length normalisation.",
    "Dense retrieval encodes queries and documents into vectors with an "
    "embedding model and ranks by cosine similarity, capturing semantic meaning.",
    "Reciprocal Rank Fusion combines multiple ranked lists by summing "
    "1/(k + rank) for each document. It needs no score calibration; k=60 is "
    "the common default.",
    "Self-corrective RAG grades retrieved documents for relevance and rewrites "
    "the query when the retrieved context is insufficient.",
]


def load_documents(docs_dir: Path) -> List[Document]:
    docs: List[Document] = []
    if docs_dir.exists():
        for path in docs_dir.rglob("*"):
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
            elif suffix in {".txt", ".md"}:
                docs.append(
                    Document(
                        page_content=path.read_text(errors="ignore"),
                        metadata={"source": str(path)},
                    )
                )
    if not docs:
        print(f"[index] No files in ./{docs_dir}, using built-in sample docs.")
        docs = [
            Document(page_content=text, metadata={"source": f"sample_{i}"})
            for i, text in enumerate(SAMPLE_DOCS)
        ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)
