"""Public request values for starting and resuming an operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import RuntimeValidationError
from .messages import Message
from .tools import ToolSpec


@dataclass(frozen=True)
class ExecutionOptions:
    """Boundaries for one single-Agent execution."""

    max_turns: int = 8
    max_tool_calls: int = 32
    timeout_seconds: float = 300.0
    max_model_retries: int = 2
    thinking_level: str = "normal"
    tool_execution_mode: str = "sequential"

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise RuntimeValidationError("max_turns must be at least 1")
        if self.max_tool_calls < 0:
            raise RuntimeValidationError("max_tool_calls cannot be negative")
        if self.timeout_seconds <= 0:
            raise RuntimeValidationError("timeout_seconds must be positive")
        if self.max_model_retries < 0:
            raise RuntimeValidationError("max_model_retries cannot be negative")
        if self.tool_execution_mode != "sequential":
            raise RuntimeValidationError("only sequential tool execution is supported in WP1")


@dataclass(frozen=True)
class RunRequest:
    """Fully resolved input for one Runtime operation."""

    owner_id: str
    session_id: str
    agent_id: str
    messages: list[Message] = field(default_factory=list)
    system_prompt: str = ""
    model: str = ""
    tools: list[ToolSpec] = field(default_factory=list)
    tool_context: dict[str, Any] = field(default_factory=dict)
    execution_options: ExecutionOptions = field(default_factory=ExecutionOptions)
    metadata: dict[str, Any] = field(default_factory=dict)
    orchestration_run_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("owner_id", "session_id", "agent_id"):
            if not getattr(self, field_name).strip():
                raise RuntimeValidationError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class ResumeInput:
    """Explicit input used to continue a suspended operation."""

    kind: str
    decision: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise RuntimeValidationError("resume input kind must not be empty")
