"""RAG pure-logic tests.

External embedding models and ChromaDB persistence are intentionally excluded
from the default unit-test gate. They are optional capabilities and should be
validated separately when the environment is explicitly prepared.

本文件标记为 integration：默认运行（`addopts` 含 `-m 'not integration'`）会跳过，
准备环境后用 `uv sync --extra rag && pytest -m integration` 单独验证。
"""

from __future__ import annotations

from pathlib import Path

import pytest

# 可选能力：缺少 rag extra 时整个文件标记为 skipped，而不是收集期报错。
# 必须在导入 matrix.rag.* 之前执行（indexer 在模块层 import chromadb）。
pytest.importorskip("chromadb", reason="tests/test_rag.py 需要可选 rag extra：uv sync --extra rag")

from matrix.rag.embedder import _hash_embedding  # noqa: E402
from matrix.rag.indexer import RagSourcePolicy, _split_paragraphs  # noqa: E402
from matrix.rag.retriever import _rrf_fuse  # noqa: E402

pytestmark = pytest.mark.integration


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


def test_rag_source_policy_is_allowlist_and_secret_aware(tmp_path: Path) -> None:
    root = tmp_path / "personal-assets"
    allowed = root / "21-知识"
    allowed.mkdir(parents=True)
    (allowed / "guide.md").write_text("allowed", encoding="utf-8")
    (allowed / "credentials.yaml").write_text("secret", encoding="utf-8")
    (allowed / ".env.md").write_text("secret", encoding="utf-8")
    (root / "00-长乐道").mkdir()
    (root / "00-长乐道" / "private.md").write_text("private", encoding="utf-8")
    (root / "92-系统").mkdir()
    (root / "92-系统" / "runtime.md").write_text("runtime", encoding="utf-8")

    policy = RagSourcePolicy()

    assert policy.is_allowed_file(root, allowed / "guide.md")
    assert not policy.is_allowed_file(root, allowed / "credentials.yaml")
    assert not policy.is_allowed_file(root, allowed / ".env.md")
    assert not policy.is_allowed_file(
        root, root / "00-长乐道" / "private.md",
    )
    assert not policy.is_allowed_file(
        root, root / "92-系统" / "runtime.md",
    )
    assert not policy.is_allowed_file(
        root, tmp_path / "outside.md",
    )


def test_rag_source_policy_supports_direct_allowed_root(tmp_path: Path) -> None:
    root = tmp_path / "21-知识"
    root.mkdir()
    note = root / "nested" / "note.txt"
    note.parent.mkdir()
    note.write_text("allowed", encoding="utf-8")

    assert RagSourcePolicy().is_allowed_file(root, note)
    assert RagSourcePolicy(["."]).is_allowed_file(root, note)
    assert not RagSourcePolicy(["."]).is_allowed_file(
        root, root / "secret.md",
    )


def test_rag_source_policy_rejects_sensitive_root(tmp_path: Path) -> None:
    root = tmp_path / "92-系统"
    root.mkdir()
    runtime_file = root / "runtime.md"
    runtime_file.write_text("runtime", encoding="utf-8")

    assert not RagSourcePolicy(["."]).is_allowed_file(root, runtime_file)
