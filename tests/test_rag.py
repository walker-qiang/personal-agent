"""RAG 模块冒烟测试：embedder / bm25 / retriever / indexer 单元测试 + 端到端检索验证。"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_deps():
    """Skip if dependencies are not installed."""
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        import rank_bm25  # noqa: F401
        import jieba  # noqa: F401
    except ImportError as e:
        pytest.skip(f"RAG dependency missing: {e}")


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------


class TestEmbedder:
    def test_hash_embedding_fallback(self):
        """Real embedder may be slow; test hash fallback directly."""
        from matrix.rag.embedder import _hash_embedding

        vec = _hash_embedding("hello world")
        assert len(vec) == 256
        assert -1.0 <= vec[0] <= 1.0
        # Deterministic
        assert _hash_embedding("hello") == _hash_embedding("hello")

    def test_real_embedder_encodes(self):
        """Test that a real LocalEmbedder produces 512-dim vectors."""
        _require_deps()
        from matrix.rag.embedder import LocalEmbedder

        embedder = LocalEmbedder()
        # Skip if sentence-transformers couldn't load (no model cached)
        if embedder._model is None:
            pytest.skip("No real model available")

        vecs = embedder.encode(["你好世界", "RAG 测试"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 512
        assert all(-1.0 <= v <= 1.0 for v in vecs[0])

    def test_encode_single(self):
        _require_deps()
        from matrix.rag.embedder import LocalEmbedder

        embedder = LocalEmbedder()
        if embedder._model is None:
            pytest.skip("No real model available")

        vec = embedder.encode_single("测试")
        assert len(vec) == 512


# ---------------------------------------------------------------------------
# BM25 tests
# ---------------------------------------------------------------------------


class TestBM25:
    def test_index_and_search(self):
        _require_deps()
        from matrix.rag.bm25 import BM25Retriever

        bm25 = BM25Retriever()
        docs = [
            {"id": "1", "content": "Python 是人工智能领域最流行的编程语言"},
            {"id": "2", "content": "Java 是企业级应用开发的主流语言"},
            {"id": "3", "content": "Python 和 Java 都可以用于后端开发"},
        ]
        bm25.add_documents(docs)

        results = bm25.search("人工智能 Python", top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "1"  # most relevant
        assert results[0]["score"] > 0

    def test_empty_search(self):
        _require_deps()
        from matrix.rag.bm25 import BM25Retriever

        bm25 = BM25Retriever()
        results = bm25.search("anything", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        _require_deps()
        self.tmpdir = tempfile.mkdtemp(prefix="rag_test_")
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_index_and_query(self):
        from matrix.rag.embedder import LocalEmbedder
        from matrix.rag.indexer import DocumentIndexer
        from matrix.rag.retriever import HybridRetriever

        embedder = LocalEmbedder()
        if embedder._model is None:
            pytest.skip("No real model available")

        # Create a temp docs directory with some markdown files
        docs_dir = Path(self.tmpdir) / "docs"
        docs_dir.mkdir()
        (docs_dir / "ai.md").write_text(
            "# 人工智能\n人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。",
            encoding="utf-8",
        )
        (docs_dir / "python.md").write_text(
            "# Python 编程\nPython 是一种解释型、面向对象的高级编程语言，广泛用于数据科学和机器学习。",
            encoding="utf-8",
        )
        (docs_dir / "golang.md").write_text(
            "# Go 语言\nGo 是 Google 开发的一种静态类型、编译型编程语言，适合构建高性能网络服务。",
            encoding="utf-8",
        )

        persist_dir = Path(self.tmpdir) / "chromadb"

        indexer = DocumentIndexer(embedder=embedder, persist_dir=str(persist_dir))
        chunk_count = indexer.index_directory(str(docs_dir))
        assert chunk_count >= 3  # at least one per file

        retriever = HybridRetriever(embedder=embedder, persist_dir=str(persist_dir))

        # Query: AI-related
        results = retriever.query("机器学习", top_k=2)
        assert len(results) >= 1
        # Should return AI or Python doc first
        found = {r["title"] for r in results}
        # title may be absolute path; check basename
        basenames = {Path(t).name for t in found}
        assert "ai.md" in basenames or "python.md" in basenames

        # Query: Go-related
        results = retriever.query("高性能网络服务", top_k=2)
        assert len(results) >= 1
        found = {r["title"] for r in results}
        basenames = {Path(t).name for t in found}
        assert "golang.md" in basenames

    def test_query_returns_structured(self):
        from matrix.rag.embedder import LocalEmbedder
        from matrix.rag.indexer import DocumentIndexer
        from matrix.rag.retriever import HybridRetriever

        embedder = LocalEmbedder()
        if embedder._model is None:
            pytest.skip("No real model available")

        docs_dir = Path(self.tmpdir) / "docs"
        docs_dir.mkdir()
        (docs_dir / "test.md").write_text("# Test\nTest content for structured output.", encoding="utf-8")

        persist_dir = Path(self.tmpdir) / "chromadb"

        indexer = DocumentIndexer(embedder=embedder, persist_dir=str(persist_dir))
        indexer.index_directory(str(docs_dir))

        retriever = HybridRetriever(embedder=embedder, persist_dir=str(persist_dir))
        results = retriever.query("structured output", top_k=1)

        assert len(results) == 1
        r = results[0]
        assert "id" in r
        assert "title" in r
        assert "content" in r
        assert "score" in r
        assert r["score"] > 0


# ---------------------------------------------------------------------------
# RRF fusion tests
# ---------------------------------------------------------------------------


class TestRRF:
    def test_rrf_fuse(self):
        from matrix.rag.retriever import _rrf_fuse

        vector = [
            {"id": "a", "score": 0.9, "content": "A"},
            {"id": "b", "score": 0.7, "content": "B"},
        ]
        bm25 = [
            {"id": "b", "score": 1.5, "content": "B"},
            {"id": "c", "score": 1.0, "content": "C"},
        ]

        fused = _rrf_fuse(vector, bm25, k=60)
        # b appears in both lists → higher combined score
        assert fused[0]["id"] == "b"
        assert len(fused) == 3  # a, b, c


# ---------------------------------------------------------------------------
# Indexer tests
# ---------------------------------------------------------------------------


class TestIndexer:

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.tmpdir = tempfile.mkdtemp(prefix="rag_test_")
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_markdown_chunking(self):
        from matrix.rag.indexer import _split_paragraphs

        text = "# Title\n\nThis is a paragraph.\n\n## Section 2\nContent here."
        chunks = _split_paragraphs(text)
        assert len(chunks) >= 1

    def test_empty_directory(self):
        _require_deps()
        from matrix.rag.embedder import LocalEmbedder
        from matrix.rag.indexer import DocumentIndexer

        embedder = LocalEmbedder()
        if embedder._model is None:
            pytest.skip("No real model available")

        empty_dir = Path(self.tmpdir) / "empty"
        empty_dir = Path(tempfile.mkdtemp(prefix="rag_empty_"))

        persist_dir = Path(tempfile.mkdtemp(prefix="rag_persist_"))
        indexer = DocumentIndexer(embedder=embedder, persist_dir=str(persist_dir))
        count = indexer.index_directory(str(empty_dir))
        assert count == 0

        shutil.rmtree(str(empty_dir), ignore_errors=True)
        shutil.rmtree(str(persist_dir), ignore_errors=True)