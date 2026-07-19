"""knowledge_search — 搜索个人知识库中的文档。

按需调用，由 LLM 在 ReAct 循环中决定何时检索。
检索器由 register_all 注入，避免全局状态。
"""

from __future__ import annotations

from typing import Any

from ..base import ToolDefinition

tool_definition = ToolDefinition(
    name="knowledge_search",
    description="搜索个人知识库中的文档。用于：用户问「我的笔记里有没有…」「之前记录过什么…」「知识库中关于…的内容」。返回匹配度最高的文档片段。仅当用户明确询问个人知识库内容时才调用。",
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
    """Search personal knowledge base. Handler is injected by register_all."""
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


def _get_retriever() -> Any:
    return _retriever


def set_retriever(retriever: Any) -> None:
    """Inject the retriever instance. Called by register_all at startup."""
    global _retriever
    _retriever = retriever