"""Model provider port; concrete LLM clients live outside Runtime Core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

from ..domain.messages import Message, ToolCall
from ..domain.tools import ToolSpec


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    messages: list[Message]
    tools: list[ToolSpec]
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvent:
    kind: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: dict[str, Any] = field(default_factory=dict)


class ModelPort(Protocol):
    """The minimum model interface required by the future loop."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        ...

    def stream(self, request: ModelRequest) -> Iterator[ModelEvent]:
        ...
