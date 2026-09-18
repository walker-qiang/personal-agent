"""RAG pure-logic tests.

External embedding models and ChromaDB persistence are intentionally excluded
from the default unit-test gate. They are optional capabilities and should be
validated separately when the environment is explicitly prepared.
"""

from __future__ import annotations

from matrix.rag.embedder import _hash_embedding
from matrix.rag.indexer import _split_paragraphs
from matrix.rag.retriever import _rrf_fuse


def test_hash_embedding_fallback_is_deterministic() -> None:
    vector = _hash_embedding("hello world")

    assert len(vector) == 256
    assert all(-1.0 <= value <= 1.0 for value in vector)
    assert _hash_embedding("hello") == _hash_embedding("hello")


def test_rrf_fuse_prefers_documents_present_in_both_rankings() -> None:
    fused = _rrf_fuse(
        [
            {"id": "a", "score": 0.9, "content": "A"},
            {"id": "b", "score": 0.7, "content": "B"},
        ],
        [
            {"id": "b", "score": 1.5, "content": "B"},
            {"id": "c", "score": 1.0, "content": "C"},
        ],
        k=60,
    )

    assert fused[0]["id"] == "b"
    assert len(fused) == 3


def test_split_paragraphs_removes_empty_paragraphs() -> None:
    chunks = _split_paragraphs(
        "# Title\n\nThis is a paragraph.\n\n## Section 2\nContent here.",
    )

    assert chunks == [
        "# Title",
        "This is a paragraph.",
        "## Section 2\nContent here.",
    ]
