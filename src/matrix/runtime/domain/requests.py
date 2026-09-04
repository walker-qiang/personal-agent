"""Public request values for starting and resuming an operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import RuntimeValidationError
from .messages import Message
from .tools import ToolSpec


@dataclass(frozen=True)
class ExecutionPolicy:
    """Resolved capability policy supplied by the application layer.

    The Runtime does not resolve user-facing presets.  It only enforces this
    already-resolved, provider-neutral policy at the tool boundary.
    """

    mode: str = "read_only"
    preset: str = "default"
    allow_external_effects: bool = False
    require_approval: bool = True
    approval_mode: str = "manual"
    auto_approve_operations: tuple[str, ...] = ()
    debug_trace: bool = False
    output_style: str = "default"

    def __post_init__(self) -> None:
        if self.mode not in {"read_only", "writeback"}:
            raise RuntimeValidationError(
                f"unsupported agent mode: {self.mode}; expected read_only or writeback"
            )
        if not self.preset.strip():
            raise RuntimeValidationError("agent preset must not be empty")
        if self.mode == "read_only" and self.allow_external_effects:
            raise RuntimeValidationError("read_only mode cannot allow external effects")
        if self.mode == "writeback" and not self.require_approval:
            raise RuntimeValidationError("writeback mode requires approval")
        if self.approval_mode not in {"manual", "auto_allowlist"}:
            raise RuntimeValidationError(
                "unsupported approval_mode; expected manual or auto_allowlist"
            )
        if self.approval_mode == "auto_allowlist" and self.mode != "writeback":
            raise RuntimeValidationError("auto_allowlist requires writeback mode")


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
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
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
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise RuntimeValidationError("resume input kind must not be empty")
