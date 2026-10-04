"""DocumentIndexer: 扫描目录，分块，向量化并写入 ChromaDB。"""

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import chromadb

from ..config import DEFAULT_RAG_ALLOWED_DIRS
from .context_guard import ContextGuard
from .embedder import LocalEmbedder, get_embedding_namespace

if TYPE_CHECKING:
    from .knowledge_graph import KnowledgeGraph, EntityExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 支持的文件扩展名
_SUPPORTED_EXTS = {".md", ".txt", ".yaml", ".yml"}

# 分块参数
_CHUNK_SIZE = 500  # 字符数
_CHUNK_OVERLAP = 100  # 字符数

# ChromaDB 集合名前缀；实际名称包含 embedding 模型和维度指纹。
_COLLECTION_BASE_NAME = "documents"

# Directory and filename filters are always active, even when the allowlist is
# overridden. They prevent accidental indexing of runtime material or secrets.
_SENSITIVE_DIRECTORY_NAMES = frozenset({
    ".agents",
    ".cache",
    ".codex",
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "cache",
    "credential",
    "credentials",
    "dist",
    "keys",
    "log",
    "logs",
    "node_modules",
    "private",
    "secret",
    "secrets",
    "system",
    "token",
    "tokens",
    "var",
    "附件",
    "系统",
    "91-附件",
    "92-系统",
})

_SENSITIVE_FILENAME_RE = re.compile(
    r"(^\.env(?:\..*)?$|"
    r"(^|[-_.])(api[-_]?key|credential|credentials|"
    r"password|passwd|secret|secrets|token|tokens)([-_.]|$)|"
    r"^(id_rsa|id_dsa|authorized_keys)$)",
    re.IGNORECASE,
)

# 元数据键名
_META_INDEXED_AT = ".indexed_at"
# 持久化源文件纳秒级 mtime，避免旧 ISO 时间戳的微秒精度损失导致重复索引。
_META_SOURCE_MTIME_NS = ".source_mtime_ns"
_LEGACY_MTIME_TOLERANCE_SEC = 1e-6

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _file_mtime_ns(path: str) -> int:
    """获取文件最后修改时间（Unix 纳秒时间戳）。"""
    return os.stat(path).st_mtime_ns


def _file_hash(path: str) -> str:
    """计算文件内容的 MD5 哈希（用于快速比较）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_paragraphs(text: str) -> List[str]:
    """按空行 / 段落边界拆分文本。"""
    # 按连续空行拆分
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """将文本按段落分块，每个块约 chunk_size 字符，块间 overlap 字符重叠。

    策略：先按段落拆分，然后在段落边界上合并，避免截断句子。
    """
    paragraphs = _split_paragraphs(text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len > chunk_size and current_chunk:
            # 当前块已满，保存
            chunks.append("\n\n".join(current_chunk))
            # 保留 overlap 部分：取最后几个段落直到达到 overlap 长度
            overlap_paras: List[str] = []
            overlap_len = 0
            for p in reversed(current_chunk):
                if overlap_len >= overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_len += len(p)
            current_chunk = overlap_paras
            current_len = overlap_len

        current_chunk.append(para)
        current_len += para_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


class RagSourcePolicy:
    """Fail-closed source policy for files entering the RAG index."""

    def __init__(
        self,
        allowed_dirs: Sequence[str] = DEFAULT_RAG_ALLOWED_DIRS,
    ) -> None:
        normalized: list[str] = []
        for raw_dir in allowed_dirs:
            value = str(raw_dir).strip().replace("\\", "/")
            if not value:
                continue
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                logger.warning("rag: ignoring unsafe allowed directory: %s", raw_dir)
                continue
            normalized.append(Path(os.path.normpath(value)).as_posix())
        self.allowed_dirs = tuple(dict.fromkeys(normalized))

    @staticmethod
    def _resolved(path: str | Path) -> Path:
        return Path(path).expanduser().resolve(strict=False)

    def _relative_path(
        self,
        docs_path: str | Path,
        source_path: str | Path,
    ) -> tuple[Path, Path] | None:
        docs_root = self._resolved(docs_path)
        candidate = self._resolved(source_path)
        try:
            relative = candidate.relative_to(docs_root)
        except ValueError:
            return None
        return docs_root, relative

    def _allowed_relative(self, docs_root: Path, relative: Path) -> bool:
        if not relative.parts:
            return False
        if (
            docs_root.name.startswith(".")
            or docs_root.name.casefold() in _SENSITIVE_DIRECTORY_NAMES
        ):
            return False
        if any(
            part.startswith(".") or part.casefold() in _SENSITIVE_DIRECTORY_NAMES
            for part in relative.parts[:-1]
        ):
            return False

        allowed = tuple(Path(item) for item in self.allowed_dirs)
        if any(item == Path(".") for item in allowed):
            return True

        relative_parts = relative.parts
        if docs_root.name in {
            item.name for item in allowed if len(item.parts) == 1
        }:
            return True
        return any(
            len(item.parts) <= len(relative_parts)
            and relative_parts[:len(item.parts)] == item.parts
            for item in allowed
        )

    def is_allowed_directory(
        self,
        docs_path: str | Path,
        directory_path: str | Path,
    ) -> bool:
        """Return whether walking into a directory is permitted."""
        result = self._relative_path(docs_path, directory_path)
        if result is None:
            return False
        docs_root, relative = result
        if any(
            part.startswith(".") or part.casefold() in _SENSITIVE_DIRECTORY_NAMES
            for part in relative.parts
        ):
            return False
        return self._allowed_relative(docs_root, relative / "_placeholder")

    def is_allowed_file(
        self,
        docs_path: str | Path,
        file_path: str | Path,
    ) -> bool:
        """Return whether a file may be read and indexed."""
        result = self._relative_path(docs_path, file_path)
        if result is None:
            return False
        docs_root, relative = result
        if not self._allowed_relative(docs_root, relative):
            return False
        filename = relative.name
        return not (
            filename.startswith(".")
            or _SENSITIVE_FILENAME_RE.search(filename) is not None
        )


# ---------------------------------------------------------------------------
# DocumentIndexer
# ---------------------------------------------------------------------------


class DocumentIndexer:
    """文档索引器：扫描目录中的 .md / .txt / .yaml 文件，按段落分块，
    向量化后写入 ChromaDB 持久化存储，支持增量索引。
    """

    def __init__(
        self,
        embedder: Optional[LocalEmbedder] = None,
        persist_dir: Optional[str] = None,
        chunk_size: int = _CHUNK_SIZE,
        chunk_overlap: int = _CHUNK_OVERLAP,
        knowledge_graph: Optional["KnowledgeGraph"] = None,
        entity_extractor: Optional["EntityExtractor"] = None,
        allowed_dirs: Sequence[str] = DEFAULT_RAG_ALLOWED_DIRS,
    ) -> None:
        """
        Args:
            embedder: LocalEmbedder 实例，若为 None 则自动创建。
            persist_dir: ChromaDB 持久化目录，若为 None 则使用默认临时目录。
            chunk_size: 分块大小（字符数）。
            chunk_overlap: 块间重叠字符数。
            knowledge_graph: 可选的知识图谱实例，索引时自动抽取实体。
            entity_extractor: 可选的实体抽取器，若 None 且 knowledge_graph 存在则自动创建。
            allowed_dirs: 相对于 docs_path 的 RAG 允许目录，默认 fail-closed。
        """
        self.embedder = embedder or LocalEmbedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name, self.embedding_dimension = get_embedding_namespace(
            self.embedder, _COLLECTION_BASE_NAME,
        )

        # 初始化 ChromaDB 客户端
        if persist_dir is None:
            persist_dir = os.path.join(
                os.path.expanduser("~"), ".personal-agent", "chromadb"
            )
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Context Guard: 文档入库前的清洗层（移除 prompt injection、截断超长行）
        self._context_guard = ContextGuard()
        self._source_policy = RagSourcePolicy(allowed_dirs)

        # Knowledge Graph: 可选的实体图谱抽取
        self._knowledge_graph = knowledge_graph
        self._entity_extractor = entity_extractor
        self.last_index_changed = False

        logger.info(
            "DocumentIndexer 已初始化, persist_dir=%s, collection=%s, kg=%s",
            persist_dir,
            self.collection_name,
            "enabled" if knowledge_graph else "disabled",
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def index_directory(self, docs_path: str) -> int:
        """扫描目录并索引所有支持的文件。

        增量索引：对比文件 mtime_ns 与 ChromaDB 中记录的时间戳，只处理变更文件。

        Args:
            docs_path: 文档目录路径。

        Returns:
            本次索引新增/更新的 chunk 数量。
        """
        if not os.path.isdir(docs_path):
            raise ValueError(f"目录不存在: {docs_path}")

        # Remove chunks from sources that are no longer allowed before
        # collecting files, so a previous broad index cannot leak old data.
        purged_chunks = self._purge_disallowed_chunks(docs_path)

        # 收集所有需要索引的文件
        files_to_index = self._find_files(docs_path)
        changed_files = self._filter_changed(files_to_index)
        self.last_index_changed = bool(changed_files or purged_chunks)

        if not changed_files:
            logger.info("没有文件需要更新索引，清理了 %d 个受限 chunk。", purged_chunks)
            return 0

        total_chunks = 0
        for file_path in changed_files:
            mtime_ns = _file_mtime_ns(file_path)
            chunk_count = self._index_file(file_path, mtime_ns)
            total_chunks += chunk_count

        logger.info(
            "索引完成: 处理了 %d 个文件，共 %d 个 chunk。",
            len(changed_files),
            total_chunks,
        )
        return total_chunks

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _find_files(self, docs_path: str) -> List[str]:
        """递归扫描目录，返回所有支持的文件路径。"""
        files: List[str] = []
        docs_root = str(Path(docs_path).expanduser().resolve())
        for root, dirs, filenames in os.walk(
            docs_root, topdown=True, followlinks=False,
        ):
            dirs[:] = [
                directory for directory in dirs
                if self._source_policy.is_allowed_directory(
                    docs_root, os.path.join(root, directory),
                )
            ]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                file_path = os.path.join(root, fname)
                if ext in _SUPPORTED_EXTS and self._source_policy.is_allowed_file(
                    docs_root, file_path,
                ):
                    files.append(str(Path(file_path).resolve(strict=False)))
        logger.debug("扫描到 %d 个支持的文件。", len(files))
        return sorted(set(files))

    def _purge_disallowed_chunks(self, docs_path: str) -> int:
        """Delete existing chunks whose source is outside the current policy."""
        try:
            existing = self._collection.get(include=["metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("rag: unable to inspect existing chunks: %s", exc)
            return 0

        ids_to_delete: list[str] = []
        ids = existing.get("ids", []) or []
        metadatas = existing.get("metadatas", []) or []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else None
            source_file = metadata.get("source_file") if isinstance(metadata, dict) else ""
            if not source_file or not self._source_policy.is_allowed_file(
                docs_path, source_file,
            ):
                ids_to_delete.append(chunk_id)

        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.warning(
                "rag: purged %d chunks outside source policy", len(ids_to_delete),
            )
        return len(ids_to_delete)

    def _filter_changed(self, file_paths: List[str]) -> List[str]:
        """对比文件 mtime_ns 与 ChromaDB 中存储的索引时间，返回需要重新索引的文件列表。

        判断逻辑：如果文件在 ChromaDB 中不存在任何 chunk，或文件 mtime
        晚于最早一条 chunk 的 source_mtime_ns（兼容旧 indexed_at），则认为需要重新索引。
        """
        changed: List[str] = []
        for fp in file_paths:
            mtime_ns = _file_mtime_ns(fp)

            # 查询 ChromaDB 中是否有该文件的 chunk
            existing = self._collection.get(
                where={"source_file": fp},
                limit=1,
            )

            if not existing["ids"]:
                # 该文件从未被索引
                try:
                    with open(fp, "r", encoding="utf-8") as source_file:
                        if not source_file.read().strip():
                            # 空文件没有 chunk；跳过但保留“未索引”状态，
                            # 文件未来写入内容后仍会被重新发现。
                            continue
                except OSError:
                    # 交给 _index_file 记录读取错误。
                    pass
                changed.append(fp)
                continue

            metadata = existing["metadatas"][0]

            # 优先使用当前版本写入的精确纳秒级 mtime。
            source_mtime_ns = metadata.get(_META_SOURCE_MTIME_NS)
            if source_mtime_ns is not None:
                try:
                    if mtime_ns > int(source_mtime_ns):
                        self._delete_file_chunks(fp)
                        changed.append(fp)
                    continue
                except (TypeError, ValueError):
                    # 旧元数据无法解析时，回退到 legacy 时间戳。
                    pass

            # legacy 索引只有 ISO 时间戳，最多精确到微秒；文件系统 mtime
            # 可能保留纳秒，因此增加极小容差，避免每次重启都误判为变更。
            indexed_at_str = metadata.get(_META_INDEXED_AT, "")
            if indexed_at_str:
                try:
                    indexed_at = datetime.fromisoformat(indexed_at_str).timestamp()
                    if mtime_ns / 1_000_000_000 > indexed_at + _LEGACY_MTIME_TOLERANCE_SEC:
                        # 文件有更新
                        self._delete_file_chunks(fp)
                        changed.append(fp)
                    # else: 文件未变化，跳过
                except (ValueError, TypeError):
                    changed.append(fp)
            else:
                changed.append(fp)

        return changed

    def _delete_file_chunks(self, file_path: str) -> None:
        """删除 ChromaDB 中某个文件的所有 chunk。"""
        existing = self._collection.get(
            where={"source_file": file_path},
        )
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
            logger.debug("已删除文件 %s 的 %d 个旧 chunk。", file_path, len(existing["ids"]))

    def _index_file(self, file_path: str, mtime_ns: int) -> int:
        """索引单个文件，返回生成的 chunk 数量。"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            logger.error("读取文件 %s 失败: %s", file_path, exc)
            return 0

        if not content.strip():
            return 0

        # Context Guard: 在分块前清洗内容（移除 prompt injection、截断超长行）
        guard_result = self._context_guard.sanitize(content)
        content = guard_result.content
        guard_metadata = guard_result.metadata

        # 分块
        chunks = _chunk_text(content, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0

        # 向量化
        vectors = self.embedder.encode(chunks)

        # 构建 ChromaDB 记录
        ids: List[str] = []
        metadatas: List[Dict] = []
        indexed_at_iso = datetime.fromtimestamp(
            mtime_ns / 1_000_000_000, tz=timezone.utc,
        ).isoformat()

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{file_path}::{i}"
            ids.append(chunk_id)
            metadatas.append(
                {
                    _META_INDEXED_AT: indexed_at_iso,
                    _META_SOURCE_MTIME_NS: str(mtime_ns),
                    "source_file": file_path,
                    "chunk_index": i,
                    "content": chunk_text,
                    # Context Guard 元数据标记
                    "source_type": guard_metadata.get("source_type", "external"),
                    "sanitized": guard_metadata.get("sanitized", True),
                }
            )

        # 写入 ChromaDB
        try:
            self._collection.add(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=chunks,
            )
            logger.debug("文件 %s 已索引: %d 个 chunk。", file_path, len(chunks))
        except Exception as exc:
            logger.error("写入 ChromaDB 失败 (文件: %s): %s", file_path, exc)
            return 0

        # Knowledge Graph: 从文档中抽取实体和关系
        if self._knowledge_graph is not None:
            self._extract_entities_from_chunks(chunks, file_path)

        return len(chunks)

    def _extract_entities_from_chunks(self, chunks: List[str], file_path: str) -> None:
        """从文档分块中抽取实体并添加到知识图谱."""
        if not self._entity_extractor:
            from .knowledge_graph import EntityExtractor
            self._entity_extractor = EntityExtractor()

        total_entities = 0
        total_relations = 0
        for chunk in chunks:
            try:
                result = self._entity_extractor.extract(chunk, source_file=file_path)
                if result.entities or result.relations:
                    self._knowledge_graph.add_extraction(result)
                    total_entities += len(result.entities)
                    total_relations += len(result.relations)
            except Exception as exc:
                logger.debug("entity extraction failed for chunk in %s: %s", file_path, exc)

        if total_entities or total_relations:
            logger.debug(
                "kg_extract: %d entities, %d relations from %s",
                total_entities, total_relations, file_path,
            )
