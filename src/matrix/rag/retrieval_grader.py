"""RetrievalGrader: LLM-based relevance grading for retrieved documents.

Implements CRAG (Corrective RAG) style document grading:
- Grade each document's relevance to the query (relevant/irrelevant/partially)
- Filter out irrelevant documents
- Return assessment for multi-step retrieval decisions
"""

from __future__ import annotations

import logging
from typing import Any

from ..llm import LLMClient, LLMError

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

_GRADE_SYSTEM = """你是一个文档相关性评估员。判断每个检索到的文档片段是否与用户查询相关。

评分标准：
- "relevant" — 文档直接包含回答查询所需的信息
- "partially" — 文档包含部分相关信息，但不完整
- "irrelevant" — 文档与查询无关

输出JSON格式：
{
  "graded_docs": [
    {"index": 0, "relevance": "relevant", "reason": "简要说明"},
    {"index": 1, "relevance": "irrelevant", "reason": "简要说明"}
  ],
  "overall_assessment": "sufficient" | "insufficient" | "partial",
  "missing_info": "如果信息不足，说明缺少什么（简短描述）"
}

评估原则：
- 宁可严格也不要宽松（误判相关比误判不相关危害更大）
- 如果文档只是碰巧包含相同关键词但讨论的是不同主题，判为 irrelevant
- overall_assessment 为 "sufficient" 当且仅当至少有1个 relevant 文档
"""


class RetrievalGrader:
    """Grades retrieved documents for relevance using LLM.

    When no LLM is available, returns all documents as relevant (graceful degradation).
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    @property
    def available(self) -> bool:
        """Whether LLM-based grading is available."""
        return self._llm is not None

    def grade(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Grade documents for relevance to the query.

        Args:
            query: The user's (possibly rewritten) query.
            documents: List of retrieved documents, each with at least "content" key.

        Returns:
            Dict with keys:
                - "relevant_docs": list[dict] — documents graded as relevant or partially
                - "irrelevant_count": int — number of irrelevant documents
                - "assessment": str — "sufficient" | "insufficient" | "partial"
                - "missing_info": str — description of what's missing (if insufficient)
        """
        if not documents:
            return {
                "relevant_docs": [],
                "irrelevant_count": 0,
                "assessment": "insufficient",
                "missing_info": "没有检索到任何文档",
            }

        if self._llm is None:
            # Graceful degradation: return all as relevant
            return {
                "relevant_docs": documents,
                "irrelevant_count": 0,
                "assessment": "sufficient",
                "missing_info": "",
            }

        try:
            # Build document list for grading
            doc_list = "\n\n".join(
                f"[文档{i}] {doc.get('title', '无标题')}\n{doc.get('content', '')[:500]}"
                for i, doc in enumerate(documents)
            )

            result = self._llm.complete_json(
                _GRADE_SYSTEM,
                [{"role": "user", "content": f"查询：{query}\n\n文档列表：\n{doc_list}"}],
                temperature=0.0,
            )

            graded = result.get("graded_docs", [])
            assessment = str(result.get("overall_assessment", "partial"))
            missing_info = str(result.get("missing_info", ""))

            # Filter relevant and partially relevant docs
            relevant_indices: set[int] = set()
            for item in graded:
                idx = item.get("index", -1)
                relevance = str(item.get("relevance", "irrelevant")).lower()
                if relevance in ("relevant", "partially"):
                    relevant_indices.add(idx)

            relevant_docs = [
                doc
                for i, doc in enumerate(documents)
                if i in relevant_indices
            ]

            # If all filtered out, keep top 2 by score (don't return empty)
            if not relevant_docs and documents:
                relevant_docs = documents[:2]
                assessment = "partial"
                logger.warning("grader filtered all docs, keeping top 2 by score as fallback")

            return {
                "relevant_docs": relevant_docs,
                "irrelevant_count": len(documents) - len(relevant_docs),
                "assessment": assessment,
                "missing_info": missing_info,
            }

        except (LLMError, Exception) as exc:
            logger.warning("retrieval grading failed, returning all docs: %s", exc)
            return {
                "relevant_docs": documents,
                "irrelevant_count": 0,
                "assessment": "sufficient",
                "missing_info": "",
            }
