"""LocalEmbedder: 使用 sentence-transformers 加载本地模型，支持降级到伪向量。"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 尝试导入 sentence-transformers
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    _HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "sentence-transformers 未安装，将使用伪向量（哈希嵌入）作为降级方案。"
    )

# ---------------------------------------------------------------------------
# 降级：简单的哈希嵌入（仅用于开发测试）
# ---------------------------------------------------------------------------

_PSEUDO_DIM = 256  # 伪向量维度


def _hash_embedding(text: str, dim: int = _PSEUDO_DIM) -> List[float]:
    """基于文本哈希生成固定维度的伪向量，仅用于开发测试。"""
    import hashlib

    # 使用 SHA-256 生成确定性哈希，再扩展 / 截断到 dim 维
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec: List[float] = []
    for i in range(dim):
        byte_val = h[i % len(h)]
        # 归一化到 [-1, 1]
        vec.append((byte_val / 127.5) - 1.0)
    return vec


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------


class LocalEmbedder:
    """使用本地 sentence-transformers 模型生成文本嵌入向量。

    默认模型为 ``BAAI/bge-small-zh-v1.5``。如果 sentence-transformers
    不可用，自动降级为基于哈希的伪向量，仅用于开发测试。
    """

    DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self._model: Optional["SentenceTransformer"] = None

        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(
                    self.model_name, device=self.device
                )
                logger.info(
                    "LocalEmbedder 已加载模型: %s (device=%s)",
                    self.model_name,
                    str(self._model.device),
                )
            except Exception as exc:
                logger.error("加载模型 %s 失败: %s", self.model_name, exc)
                self._model = None
        else:
            logger.info("LocalEmbedder 运行在降级模式（伪向量）。")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本为向量。

        Args:
            texts: 待编码的文本列表。

        Returns:
            与 texts 一一对应的向量列表，每个向量为 float 列表。
        """
        if not texts:
            return []

        if self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts, normalize_embeddings=True
                )
                return embeddings.tolist()  # type: ignore[union-attr]
            except Exception as exc:
                logger.error("模型编码失败: %s，降级到伪向量。", exc)

        # 降级路径
        return [_hash_embedding(t) for t in texts]

    def encode_single(self, text: str) -> List[float]:
        """编码单个文本为向量。

        Args:
            text: 待编码的文本。

        Returns:
            向量（float 列表）。
        """
        results = self.encode([text])
        return results[0]