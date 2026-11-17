"""Compatibility entry point for the modular Agentic RAG application.

Run with ``python RAG.PY`` or import ``ask`` from this module.
"""

from rag_app.cli import ask, build_application, main
from rag_app.config import Settings
from rag_app.ingestion import load_documents
from rag_app.retrieval import reciprocal_rank_fusion

__all__ = [
    "Settings",
    "ask",
    "build_application",
    "load_documents",
    "reciprocal_rank_fusion",
]


if __name__ == "__main__":
    main()
