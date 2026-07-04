"""Tool system base types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Handler: receives arguments dict and returns result dict
ToolHandler = Callable[..., dict[str, Any]]


class FinanceToolError(Exception):
    """Raised for invalid read-only finance tool calls."""


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable definition of a registered tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the tool definition in the format expected by LLM planners."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }