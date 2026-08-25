"""Web search, fetch, news, finance, and weather tools."""

from __future__ import annotations

from ..registry import ToolRegistry
from . import fetch, finance, news_search, search, weather


def register_all(registry: ToolRegistry) -> None:
    """Register all web tools in the given registry."""
    from ..base import ToolDefinition

    registry.register(
        ToolDefinition(
            name=search.tool_definition.name,
            description=search.tool_definition.description,
            input_schema=search.tool_definition.input_schema,
            handler=search.web_search,
            capabilities=search.tool_definition.capabilities,
        )
    )
    registry.register(
        ToolDefinition(
            name=news_search.tool_definition.name,
            description=news_search.tool_definition.description,
            input_schema=news_search.tool_definition.input_schema,
            handler=news_search.news_search,
            capabilities=news_search.tool_definition.capabilities,
        )
    )
    registry.register(
        ToolDefinition(
            name=finance.tool_definition.name,
            description=finance.tool_definition.description,
            input_schema=finance.tool_definition.input_schema,
            handler=finance.finance_query,
            capabilities=finance.tool_definition.capabilities,
        )
    )
    registry.register(
        ToolDefinition(
            name=fetch.tool_definition.name,
            description=fetch.tool_definition.description,
            input_schema=fetch.tool_definition.input_schema,
            handler=fetch.web_fetch,
            capabilities=fetch.tool_definition.capabilities,
        )
    )
    registry.register(
        ToolDefinition(
            name=weather.tool_definition.name,
            description=weather.tool_definition.description,
            input_schema=weather.tool_definition.input_schema,
            handler=weather.weather,
            capabilities=weather.tool_definition.capabilities,
        )
    )


__all__ = [
    "register_all",
    "fetch",
    "finance",
    "news_search",
    "search",
    "weather",
]
