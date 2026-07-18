"""Web search, fetch, news, and weather tools."""

from __future__ import annotations

from ..registry import ToolRegistry
from . import fetch, news_search, search, weather


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
            name=news_search.tool_definition.name,
            description=news_search.tool_definition.description,
            input_schema=news_search.tool_definition.input_schema,
            handler=news_search.news_search,
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
    registry.register(
        ToolDefinition(
            name=weather.tool_definition.name,
            description=weather.tool_definition.description,
            input_schema=weather.tool_definition.input_schema,
            handler=weather.weather,
        )
    )


__all__ = [
    "register_all",
    "fetch",
    "news_search",
    "search",
    "weather",
]