"""BM25Retriever: 基于 rank_bm25 的关键词检索引擎。"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 尝试导入 rank_bm25
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi

    _HAS_RANK_BM25 = True
except ImportError:  # pragma: no cover
    _HAS_RANK_BM25 = False
    logger.warning("rank_bm25 未安装，BM25 检索不可用。")

# ---------------------------------------------------------------------------
# 尝试导入 jieba 分词
# ---------------------------------------------------------------------------
try:
    import jieba

    _HAS_JIEBA = True
except ImportError:  # pragma: no cover
    _HAS_JIEBA = False
    logger.warning("jieba 未安装，将使用按字符分词。")

# ---------------------------------------------------------------------------
# 分词工具
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """对文本进行分词。

    优先使用 jieba 分词；如果 jieba 不可用，回退到按字符拆分。
    """
    if _HAS_JIEBA:
        return list(jieba.cut(text))
    else:
        return list(text)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """基于 BM25 算法的关键词检索器。

    使用 rank_bm25 库实现，支持中文 jieba 分词。
    """

    def __init__(self) -> None:
        self._docs: List[Dict] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional["BM25Okapi"] = None
        self._doc_id_to_index: Dict[str, int] = {}
        self._initialized = _HAS_RANK_BM25

    @property
    def is_available(self) -> bool:
        """BM25 是否可用（依赖 rank_bm25 是否已安装）。"""
        return self._initialized

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def add_documents(self, docs: List[Dict]) -> None:
        """添加文档到 BM25 索引。

        Args:
            docs: 文档列表，每个文档需包含 ``id`` 和 ``content`` 字段。
        """
        if not self._initialized:
            logger.warning("BM25 不可用，跳过文档添加。")
            return

        for doc in docs:
            doc_id = doc["id"]
            content = doc.get("content", "")
            tokens = _tokenize(content)

            if doc_id in self._doc_id_to_index:
                # 替换已有文档
                idx = self._doc_id_to_index[doc_id]
                self._docs[idx] = doc
                self._tokenized_corpus[idx] = tokens
            else:
                self._doc_id_to_index[doc_id] = len(self._docs)
                self._docs.append(doc)
                self._tokenized_corpus.append(tokens)

        # 重建 BM25 索引
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
            logger.debug(
                "BM25 索引已更新，文档数: %d", len(self._docs)
            )

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """使用 BM25 检索与查询最相关的文档。

        Args:
            query: 查询文本。
            top_k: 返回结果数量上限。

        Returns:
            结果列表，每个结果包含 ``id``, ``content``, ``score`` 字段。
        """
        if not self._initialized or self._bm25 is None:
            logger.warning("BM25 索引为空或不可用，返回空结果。")
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # 按分数降序排列，取 top_k
        indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: List[Dict] = []
        for idx in indices:
            if scores[idx] <= 0:
                continue
            doc = self._docs[idx]
            results.append(
                {
                    "id": doc["id"],
                    "content": doc.get("content", ""),
                    "score": float(scores[idx]),
                }
            )
        return results