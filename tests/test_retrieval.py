"""Tests for retrieval and rank-fusion behavior."""

from langchain_core.documents import Document

from rag_app.retrieval import reciprocal_rank_fusion


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
