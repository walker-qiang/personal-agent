"""Provider-neutral messages used by the runtime contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


MessageContent = str | list[dict[str, Any]]


@dataclass(frozen=True)
class ToolCall:
    """A model-requested tool call."""

    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """A normalized conversation message."""

    role: str
    content: MessageContent = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role}")
