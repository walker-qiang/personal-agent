"""HybridRetriever: 组合 ChromaDB 向量检索 + BM25 关键词检索，RRF 融合。"""

import logging
import os
from typing import Dict, List, Optional

try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False
    chromadb = None  # type: ignore

from .bm25 import BM25Retriever
from .embedder import LocalEmbedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_COLLECTION_NAME = "documents"
_RRF_K = 60

# 检索候选数
_VECTOR_TOP_K = 20
_BM25_TOP_K = 20


# ---------------------------------------------------------------------------
# RRF 融合
# ---------------------------------------------------------------------------


def _rrf_fuse(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = _RRF_K,
) -> List[Dict]:
    """使用 Reciprocal Rank Fusion 融合两路检索结果。

    Args:
        vector_results: 向量检索结果，每项需含 ``id`` 和 ``score``。
        bm25_results: BM25 检索结果，每项需含 ``id`` 和 ``score``。
        k: RRF 平滑参数。

    Returns:
        融合后的结果列表，按融合分数降序排列。
    """
    # 按 id 合并分数
    fused: Dict[str, Dict] = {}

    for rank, item in enumerate(vector_results):
        doc_id = item["id"]
        rrf_score = 1.0 / (k + rank + 1)
        if doc_id not in fused:
            fused[doc_id] = {**item, "score": 0.0}
        fused[doc_id]["score"] += rrf_score

    for rank, item in enumerate(bm25_results):
        doc_id = item["id"]
        rrf_score = 1.0 / (k + rank + 1)
        if doc_id not in fused:
            fused[doc_id] = {**item, "score": 0.0}
        fused[doc_id]["score"] += rrf_score

    # 按融合分数降序排列
    sorted_results = sorted(
        fused.values(), key=lambda x: x["score"], reverse=True
    )
    return sorted_results


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """混合检索器：向量检索（ChromaDB） + 关键词检索（BM25），RRF 融合。

    使用方式::

        retriever = HybridRetriever(persist_dir="/path/to/chromadb")
        retriever.index(docs_path="/path/to/docs")
        results = retriever.query("如何使用 RAG？", top_k=5)
    """

    def __init__(
        self,
        embedder: Optional[LocalEmbedder] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            embedder: LocalEmbedder 实例，若为 None 则自动创建。
            persist_dir: ChromaDB 持久化目录。
        """
        self.embedder = embedder or LocalEmbedder()
        self.bm25 = BM25Retriever()

        if not _HAS_CHROMADB:
            raise ImportError("chromadb 未安装，无法使用 HybridRetriever。请运行: pip install chromadb")

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

        # 将 ChromaDB 中已有的文档加载到 BM25 索引
        self._load_bm25_from_chromadb()

        logger.info(
            "HybridRetriever 已初始化, persist_dir=%s", persist_dir
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        """混合检索：向量 + BM25，RRF 融合后返回 top-k 结果。

        Args:
            text: 查询文本。
            top_k: 返回结果数量。

        Returns:
            结果列表，每项包含 ``id``, ``title``, ``content``, ``score``。
        """
        # 1. 向量检索
        vector_results = self._vector_search(text, top_k=_VECTOR_TOP_K)

        # 2. BM25 检索
        bm25_results = self.bm25.search(text, top_k=_BM25_TOP_K)

        # 3. RRF 融合
        fused = _rrf_fuse(vector_results, bm25_results, k=_RRF_K)

        # 4. 取 top_k
        top_results = fused[:top_k]

        # 5. 统一输出格式
        output: List[Dict] = []
        for item in top_results:
            output.append(
                {
                    "id": item.get("id", ""),
                    "title": item.get("title", item.get("source_file", "")),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                }
            )
        return output

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _vector_search(self, text: str, top_k: int = _VECTOR_TOP_K) -> List[Dict]:
        """通过 ChromaDB 进行向量检索。"""
        try:
            query_vec = self.embedder.encode_single(text)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
            )
        except Exception as exc:
            logger.error("向量检索失败: %s", exc)
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        output: List[Dict] = []
        ids_list = results["ids"][0]
        docs_list = results["documents"][0] if results["documents"] else [""] * len(ids_list)
        metas_list = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids_list)
        dists_list = results["distances"][0] if results["distances"] else [0.0] * len(ids_list)

        for i, doc_id in enumerate(ids_list):
            meta = metas_list[i] if i < len(metas_list) else {}
            output.append(
                {
                    "id": doc_id,
                    "content": docs_list[i] if i < len(docs_list) else "",
                    "source_file": meta.get("source_file", ""),
                    "title": meta.get("source_file", ""),
                    "score": 1.0 - min(dists_list[i] if i < len(dists_list) else 0.0, 1.0),
                }
            )
        return output

    def _load_bm25_from_chromadb(self) -> None:
        """将 ChromaDB 中已有文档加载到 BM25 索引中。"""
        try:
            all_data = self._collection.get()
            if not all_data["ids"]:
                return

            docs: List[Dict] = []
            ids_list = all_data["ids"]
            docs_list = all_data["documents"] or [""] * len(ids_list)
            metas_list = all_data["metadatas"] or [{}] * len(ids_list)

            for i, doc_id in enumerate(ids_list):
                docs.append(
                    {
                        "id": doc_id,
                        "content": docs_list[i] if i < len(docs_list) else "",
                        "source_file": metas_list[i].get("source_file", "") if i < len(metas_list) else "",
                    }
                )

            self.bm25.add_documents(docs)
            logger.info("已从 ChromaDB 加载 %d 条文档到 BM25 索引。", len(docs))
        except Exception as exc:
            logger.error("从 ChromaDB 加载 BM25 索引失败: %s", exc)