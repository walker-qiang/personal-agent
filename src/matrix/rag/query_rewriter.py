"""QueryRewriter: LLM-driven query rewriting for better retrieval.

Transforms user's natural language query into retrieval-optimized form:
- HyDE-style hypothetical document generation
- Sub-query decomposition for multi-hop questions
- Keyword extraction for BM25-friendly matching
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """你是一个查询重写助手。你的任务是将用户的自然语言查询改写为更适合知识库检索的形式。

规则：
1. 提取核心实体和关键词
2. 将口语化表述转为检索友好的关键词组合
3. 对于复合问题，分解为2-3个子查询
4. 保持中文，不要翻译
5. 输出JSON格式：{"rewritten": "改写后的查询", "sub_queries": ["子查询1", "子查询2"]}

示例：
- "我的持仓中科技股占比多少" → {"rewritten": "持仓 科技股 占比 配置", "sub_queries": ["科技股持仓比例", "持仓行业分布"]}
- "最近有什么值得关注的投资机会" → {"rewritten": "投资机会 推荐 2026", "sub_queries": ["投资热点", "新兴赛道分析"]}
- "帮我看看腾讯和阿里哪家更值得买" → {"rewritten": "腾讯 阿里 投资 对比", "sub_queries": ["腾讯基本面分析", "阿里巴巴基本面分析", "腾讯阿里估值对比"]}
"""


class QueryRewriter:
    """Rewrites user queries for better retrieval using LLM.

    When no LLM is available, returns the original query unchanged (graceful degradation).
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    @property
    def available(self) -> bool:
        """Whether LLM-based rewriting is available."""
        return self._llm is not None

    def rewrite(self, query: str) -> dict[str, Any]:
        """Rewrite a query for better retrieval.

        Args:
            query: Original user query.

        Returns:
            Dict with keys:
                - "rewritten": str — primary rewritten query
                - "sub_queries": list[str] — sub-queries for multi-hop retrieval
                - "original": str — original query preserved
        """
        if self._llm is None:
            return {
                "rewritten": query,
                "sub_queries": [],
                "original": query,
            }

        try:
            result = self._llm.complete_json(
                _REWRITE_SYSTEM,
                [{"role": "user", "content": f"改写以下查询：\n{query}"}],
                temperature=0.1,
            )

            rewritten = str(result.get("rewritten", query)).strip() or query
            sub_queries = [
                sq.strip()
                for sq in result.get("sub_queries", [])
                if isinstance(sq, str) and sq.strip()
            ]

            # Always include original as fallback
            if rewritten != query and query not in sub_queries:
                sub_queries.append(query)

            return {
                "rewritten": rewritten,
                "sub_queries": sub_queries[:3],  # max 3 sub-queries
                "original": query,
            }

        except (LLMError, Exception) as exc:
            logger.warning("query rewrite failed, using original: %s", exc)
            return {
                "rewritten": query,
                "sub_queries": [],
                "original": query,
            }
