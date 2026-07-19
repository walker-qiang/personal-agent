"""RAG / knowledge base search tools."""

from __future__ import annotations

from typing import Any

from ..registry import ToolRegistry
from . import knowledge


def register_all(registry: ToolRegistry, retriever: Any = None) -> None:
    """Register RAG tools in the given registry.

    If retriever is provided, injects it into the knowledge_search handler.
    If retriever is None, registers the tool anyway (returns error on call).
    """
    from ..base import ToolDefinition

    if retriever is not None:
        knowledge.set_retriever(retriever)

    registry.register(
        ToolDefinition(
            name=knowledge.tool_definition.name,
            description=knowledge.tool_definition.description,
            input_schema=knowledge.tool_definition.input_schema,
            handler=knowledge.knowledge_search,
        )
    )


__all__ = [
    "register_all",
    "knowledge",
]