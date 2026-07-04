"""Finance read-only tools for the personal-os SQLite cache."""

from __future__ import annotations

from pathlib import Path

from ..registry import ToolRegistry
from . import allocation, assets, holdings, snapshots


def register_all(registry: ToolRegistry, cache_path: Path) -> None:
    """Register all finance tools in the given registry, bound to cache_path."""
    from ..base import ToolDefinition

    # Bind each tool handler to the configured cache path
    registry.register(
        ToolDefinition(
            name=holdings.tool_definition.name,
            description=holdings.tool_definition.description,
            input_schema=holdings.tool_definition.input_schema,
            handler=lambda **kwargs: holdings.holdings_summary(
                cache_path=str(cache_path), **kwargs
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name=assets.tool_definition.name,
            description=assets.tool_definition.description,
            input_schema=assets.tool_definition.input_schema,
            handler=lambda **kwargs: assets.asset_lookup(
                cache_path=str(cache_path), **kwargs
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name=snapshots.history_tool.name,
            description=snapshots.history_tool.description,
            input_schema=snapshots.history_tool.input_schema,
            handler=lambda **kwargs: snapshots.snapshot_history(
                cache_path=str(cache_path), **kwargs
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name=snapshots.recent_tool.name,
            description=snapshots.recent_tool.description,
            input_schema=snapshots.recent_tool.input_schema,
            handler=lambda **kwargs: snapshots.recent_snapshots(
                cache_path=str(cache_path), **kwargs
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name=allocation.tool_definition.name,
            description=allocation.tool_definition.description,
            input_schema=allocation.tool_definition.input_schema,
            handler=lambda **kwargs: allocation.bucket_allocation(
                cache_path=str(cache_path), **kwargs
            ),
        )
    )


__all__ = [
    "register_all",
    "allocation",
    "assets",
    "holdings",
    "snapshots",
]