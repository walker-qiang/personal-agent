"""DocumentIndexer: 扫描目录，分块，向量化并写入 ChromaDB。"""

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import chromadb

from .embedder import LocalEmbedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 支持的文件扩展名
_SUPPORTED_EXTS = {".md", ".txt", ".yaml", ".yml"}

# 分块参数
_CHUNK_SIZE = 500  # 字符数
_CHUNK_OVERLAP = 100  # 字符数

# ChromaDB 集合名
_COLLECTION_NAME = "documents"

# 元数据键名
_META_INDEXED_AT = ".indexed_at"

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _file_mtime(path: str) -> float:
    """获取文件最后修改时间（Unix 时间戳）。"""
    return os.path.getmtime(path)


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
    ) -> None:
        """
        Args:
            embedder: LocalEmbedder 实例，若为 None 则自动创建。
            persist_dir: ChromaDB 持久化目录，若为 None 则使用默认临时目录。
            chunk_size: 分块大小（字符数）。
            chunk_overlap: 块间重叠字符数。
        """
        self.embedder = embedder or LocalEmbedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化 ChromaDB 客户端
        if persist_dir is None:
            persist_dir = os.path.join(
                os.path.expanduser("~"), ".personal-agent", "chromadb"
            )
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "DocumentIndexer 已初始化, persist_dir=%s, collection=%s",
            persist_dir,
            _COLLECTION_NAME,
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def index_directory(self, docs_path: str) -> int:
        """扫描目录并索引所有支持的文件。

        增量索引：对比文件 mtime 与 ChromaDB 中记录的时间戳，只处理变更文件。

        Args:
            docs_path: 文档目录路径。

        Returns:
            本次索引新增/更新的 chunk 数量。
        """
        if not os.path.isdir(docs_path):
            raise ValueError(f"目录不存在: {docs_path}")

        # 收集所有需要索引的文件
        files_to_index = self._find_files(docs_path)
        changed_files = self._filter_changed(files_to_index)

        if not changed_files:
            logger.info("没有文件需要更新索引。")
            return 0

        total_chunks = 0
        for file_path in changed_files:
            mtime = _file_mtime(file_path)
            chunk_count = self._index_file(file_path, mtime)
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
        for root, _dirs, filenames in os.walk(docs_path):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in _SUPPORTED_EXTS:
                    files.append(os.path.join(root, fname))
        logger.debug("扫描到 %d 个支持的文件。", len(files))
        return files

    def _filter_changed(self, file_paths: List[str]) -> List[str]:
        """对比文件 mtime 与 ChromaDB 中存储的索引时间，返回需要重新索引的文件列表。

        判断逻辑：如果文件在 ChromaDB 中不存在任何 chunk，或文件 mtime
        晚于最早一条 chunk 的 indexed_at，则认为需要重新索引。
        """
        changed: List[str] = []
        for fp in file_paths:
            mtime = _file_mtime(fp)

            # 查询 ChromaDB 中是否有该文件的 chunk
            existing = self._collection.get(
                where={"source_file": fp},
                limit=1,
            )

            if not existing["ids"]:
                # 该文件从未被索引
                changed.append(fp)
                continue

            # 检查 indexed_at 时间戳
            indexed_at_str = existing["metadatas"][0].get(_META_INDEXED_AT, "")
            if indexed_at_str:
                try:
                    indexed_at = datetime.fromisoformat(indexed_at_str).timestamp()
                    if mtime > indexed_at:
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

    def _index_file(self, file_path: str, mtime: float) -> int:
        """索引单个文件，返回生成的 chunk 数量。"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            logger.error("读取文件 %s 失败: %s", file_path, exc)
            return 0

        if not content.strip():
            return 0

        # 分块
        chunks = _chunk_text(content, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0

        # 向量化
        vectors = self.embedder.encode(chunks)

        # 构建 ChromaDB 记录
        ids: List[str] = []
        metadatas: List[Dict] = []
        indexed_at_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{file_path}::{i}"
            ids.append(chunk_id)
            metadatas.append(
                {
                    _META_INDEXED_AT: indexed_at_iso,
                    "source_file": file_path,
                    "chunk_index": i,
                    "content": chunk_text,
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

        return len(chunks)