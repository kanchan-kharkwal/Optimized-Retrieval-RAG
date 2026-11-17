"""Tests for retrieval and rank-fusion behavior."""

from pathlib import Path

from langchain_core.documents import Document

from rag_app import ingestion
from rag_app.retrieval import reciprocal_rank_fusion


def test_load_documents_reads_pdf_pages(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "guide.pdf"
    pdf_path.touch()

    class FakePDFLoader:
        def __init__(self, path: str):
            assert path == str(pdf_path)

        def load(self):
            return [
                Document(
                    page_content="PDF content",
                    metadata={"source": str(pdf_path), "page": 0},
                )
            ]

    monkeypatch.setattr(ingestion, "PyPDFLoader", FakePDFLoader)

    chunks = ingestion.load_documents(tmp_path)

    assert len(chunks) == 1
    assert chunks[0].page_content == "PDF content"
    assert chunks[0].metadata["source"] == str(pdf_path)


def test_reciprocal_rank_fusion_prefers_document_ranked_by_both_retrievers():
    shared = Document(page_content="appears in both lists")
    bm25_only = Document(page_content="appears in BM25")
    dense_only = Document(page_content="appears in dense")

    result = reciprocal_rank_fusion(
        [[shared, bm25_only], [shared, dense_only]], top_n=3
    )

    assert result[0] == shared
    assert set(doc.page_content for doc in result) == {
        "appears in both lists",
        "appears in BM25",
        "appears in dense",
    }
