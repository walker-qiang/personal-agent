"""LLM client protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


class LLMClient(Protocol):
    """Protocol for LLM provider clients."""

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        ...

    def stream_complete(self, system: str, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream completion tokens one by one. Yields content chunks."""
        ...

    def function_call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> FunctionCallResult:
        """Call LLM with native function/tool calling support.

        Supports multi-turn tool messages:
        - role="tool" with tool_call_id + content
        - role="assistant" with tool_calls[] + content
        """
        ...


@dataclass(frozen=True)
class ToolCall:
    """A parsed tool call from the LLM response."""

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCallResult:
    """Result of a function calling LLM invocation."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"