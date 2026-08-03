"""knowledge_search — 搜索个人知识库中的文档。

按需调用，由 LLM 在 ReAct 循环中决定何时检索。
检索器由 register_all 注入，避免全局状态。

支持 Agentic RAG 模式：
- 查询重写：将用户自然语言查询改写为检索友好的形式
- 检索评分：LLM 评估每篇文档的相关性，过滤不相关结果
- 多步检索：对复合问题分解子查询，迭代检索补充信息
- CRAG 纠错：检索质量不足时返回降级建议

当 pipeline LLM 不可用时，优雅降级为普通 HybridRetriever 检索。
"""

from __future__ import annotations

from typing import Any

from ..base import ToolDefinition

tool_definition = ToolDefinition(
    name="knowledge_search",
    description="搜索个人知识库中的文档。用于：用户问「我的笔记里有没有…」「之前记录过什么…」「知识库中关于…的内容」。返回匹配度最高的文档片段。仅当用户明确询问个人知识库内容时才调用。",
    capabilities=["knowledge_base"],
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，用自然语言描述要找的内容",
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数，默认 5，最大 10",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    handler=None,  # replaced at registration time
)


def knowledge_search(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search personal knowledge base using Agentic RAG pipeline.

    When AgenticSearch is available (pipeline LLM configured), performs:
    1. Query rewriting for better retrieval
    2. Hybrid retrieval (vector + BM25)
    3. LLM-based relevance grading
    4. Multi-step retrieval if initial results insufficient

    Falls back to simple HybridRetriever.query() when AgenticSearch is not configured.
    """
    # Try Agentic RAG first
    agentic = _get_agentic_search()
    if agentic is not None:
        try:
            result = agentic.search(query, top_k=min(top_k, 10))

            # If CRAG determined web fallback is needed, note it
            response: dict[str, Any] = {
                "results": [
                    {
                        "title": d.get("title", ""),
                        "content": d.get("content", ""),
                        "score": d.get("score", 0),
                    }
                    for d in result.get("results", [])
                ],
                "query": query,
                "rewritten_query": result.get("rewritten_query", query),
                "assessment": result.get("assessment", "sufficient"),
            }

            if result.get("needs_web_fallback"):
                response["suggestion"] = "知识库中未找到足够信息，建议使用 web_search 搜索网络获取补充信息。"
                response["web_fallback_results"] = result.get("web_fallback_results", [])

            if result.get("missing_info"):
                response["missing_info"] = result["missing_info"]

            return response

        except Exception as exc:
            # Fall back to simple retrieval on any error
            import logging
            logging.getLogger(__name__).warning(
                "agentic search failed, falling back to simple retrieval: %s", exc
            )

    # Fallback: simple HybridRetriever
    _retriever = _get_retriever()
    if _retriever is None:
        return {"results": [], "message": "知识库检索器未初始化，请检查 RAG 配置。"}
    try:
        docs = _retriever.query(query, top_k=min(top_k, 10))
        results = []
        for d in docs:
            results.append({
                "title": d.get("title", ""),
                "content": d.get("content", ""),
                "score": d.get("score", 0),
            })
        return {"results": results, "query": query}
    except Exception as exc:
        return {"results": [], "error": f"知识库检索失败: {exc}"}


# ---- Retriever injection (no global state) ----

_retriever: Any = None
_agentic_search: Any = None


def _get_retriever() -> Any:
    return _retriever


def _get_agentic_search() -> Any:
    return _agentic_search


def set_retriever(retriever: Any) -> None:
    """Inject the retriever instance. Called by register_all at startup."""
    global _retriever
    _retriever = retriever


def set_agentic_search(agentic: Any) -> None:
    """Inject the AgenticSearch instance. Called by register_all at startup.

    When set, knowledge_search will use the agentic RAG pipeline (query rewriting +
    retrieval grading + multi-step retrieval + CRAG fallback) instead of simple retrieval.
    """
    global _agentic_search
    _agentic_search = agentic
