"""Agnes tools: image and video generation."""

from __future__ import annotations

from ..registry import ToolRegistry
from .generation import image_tool, video_tool


def register_all(registry: ToolRegistry) -> None:
    """Register all Agnes generation tools in the given registry."""
    from ..base import ToolDefinition

    registry.register(
        ToolDefinition(
            name=image_tool.name,
            description=image_tool.description,
            input_schema=image_tool.input_schema,
            handler=image_tool.handler,
        )
    )
    registry.register(
        ToolDefinition(
            name=video_tool.name,
            description=video_tool.description,
            input_schema=video_tool.input_schema,
            handler=video_tool.handler,
        )
    )


__all__ = [
    "register_all",
    "image_tool",
    "video_tool",
]