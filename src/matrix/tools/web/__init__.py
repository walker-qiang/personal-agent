"""Web search and fetch tools."""

from __future__ import annotations

from ..registry import ToolRegistry
from . import fetch, search


def register_all(registry: ToolRegistry) -> None:
    """Register all web tools in the given registry."""
    from ..base import ToolDefinition

    registry.register(
        ToolDefinition(
            name=search.tool_definition.name,
            description=search.tool_definition.description,
            input_schema=search.tool_definition.input_schema,
            handler=search.web_search,
        )
    )
    registry.register(
        ToolDefinition(
            name=fetch.tool_definition.name,
            description=fetch.tool_definition.description,
            input_schema=fetch.tool_definition.input_schema,
            handler=fetch.web_fetch,
        )
    )


__all__ = [
    "register_all",
    "fetch",
    "search",
]