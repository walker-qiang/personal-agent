"""RAG / knowledge base search tools."""

from __future__ import annotations

from typing import Any

from ..registry import ToolRegistry
from . import knowledge


def register_all(
    registry: ToolRegistry,
    retriever: Any = None,
    agentic_search: Any = None,
) -> None:
    """Register RAG tools in the given registry.

    If retriever is provided, injects it into the knowledge_search handler.
    If agentic_search is provided, injects it for Agentic RAG (query rewriting + grading).
    If both are None, registers the tool anyway (returns error on call).
    """
    from ..base import ToolDefinition

    if retriever is not None:
        knowledge.set_retriever(retriever)

    if agentic_search is not None:
        knowledge.set_agentic_search(agentic_search)

    registry.register(
        ToolDefinition(
            name=knowledge.tool_definition.name,
            description=knowledge.tool_definition.description,
            input_schema=knowledge.tool_definition.input_schema,
            handler=knowledge.knowledge_search,
            capabilities=knowledge.tool_definition.capabilities,
        )
    )


__all__ = [
    "register_all",
    "knowledge",
]
