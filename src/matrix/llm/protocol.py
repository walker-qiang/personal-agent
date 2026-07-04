"""LLM client protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class LLMClient(Protocol):
    """Protocol for LLM provider clients."""

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class ToolCall:
    """A parsed tool call from the planner LLM response."""

    name: str
    arguments: dict[str, Any]