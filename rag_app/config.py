"""Application settings and environment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    docs_dir: Path = Path("docs")
    top_k: int = 8
    fused_top_n: int = 6
    rrf_k: int = 60
    max_retries: int = 2
    max_subqueries: int = 4
    max_clarifications: int = 2

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_model=os.getenv("LLM_MODEL", "llama3.2:3b"),
            embedding_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            docs_dir=Path(os.getenv("DOCS_DIR", "docs")),
        )
