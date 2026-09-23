"""Public request values for starting and resuming an operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import RuntimeValidationError
from .messages import Message
from .tools import ToolSpec


RUNTIME_STATE_SCHEMA_VERSION = 2


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
class RuntimeRequestSnapshot:
    """Serializable request contract persisted with an operation.

    ``OperationState.state`` also contains mutable execution progress such as
    runtime messages and pending tool calls.  This value keeps the immutable
    request contract separate so approval recovery never depends on a caller
    reconstructing the original request from memory.
    """

    system_prompt: str = ""
    model: str = ""
    messages: tuple[Message, ...] = ()
    tools: tuple[ToolSpec, ...] = ()
    tool_context: dict[str, Any] = field(default_factory=dict)
    execution_options: ExecutionOptions = field(default_factory=ExecutionOptions)
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)
    orchestration_run_id: str = ""

    @classmethod
    def from_request(cls, request: RunRequest) -> "RuntimeRequestSnapshot":
        return cls(
            system_prompt=request.system_prompt,
            model=request.model,
            messages=tuple(request.messages),
            tools=tuple(request.tools),
            tool_context=dict(request.tool_context),
            execution_options=request.execution_options,
            execution_policy=request.execution_policy,
            metadata=dict(request.metadata),
            orchestration_run_id=request.orchestration_run_id,
        )

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RuntimeRequestSnapshot":
        """Load the canonical request snapshot from durable operation state."""

        raw = state.get("request_snapshot")
        if not isinstance(raw, dict):
            raise RuntimeValidationError(
                "operation state is missing request_snapshot"
            )
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeRequestSnapshot":
        try:
            schema_version = int(value["schema_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeValidationError(
                "request_snapshot schema_version is required"
            ) from exc
        if schema_version != RUNTIME_STATE_SCHEMA_VERSION:
            raise RuntimeValidationError(
                "unsupported request_snapshot schema_version: "
                f"{schema_version}; expected {RUNTIME_STATE_SCHEMA_VERSION}"
            )
        return cls(
            system_prompt=str(value.get("system_prompt", "")),
            model=str(value.get("model", "")),
            messages=tuple(_messages_from_dict(value.get("messages", []))),
            tools=tuple(_tools_from_dict(value.get("tools", []))),
            tool_context=dict(value.get("tool_context", {}))
            if isinstance(value.get("tool_context", {}), dict) else {},
            execution_options=_execution_options_from_dict(
                value.get("execution_options", {}),
            ),
            execution_policy=_execution_policy_from_dict(
                value.get("execution_policy", {}),
            ),
            metadata=dict(value.get("metadata", {}))
            if isinstance(value.get("metadata", {}), dict) else {},
            orchestration_run_id=str(value.get("orchestration_run_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_STATE_SCHEMA_VERSION,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "messages": _messages_to_dict(self.messages),
            "tools": _tools_to_dict(self.tools),
            "tool_context": dict(self.tool_context),
            "execution_options": {
                "max_turns": self.execution_options.max_turns,
                "max_tool_calls": self.execution_options.max_tool_calls,
                "timeout_seconds": self.execution_options.timeout_seconds,
                "max_model_retries": self.execution_options.max_model_retries,
                "thinking_level": self.execution_options.thinking_level,
                "tool_execution_mode": self.execution_options.tool_execution_mode,
            },
            "execution_policy": {
                "mode": self.execution_policy.mode,
                "preset": self.execution_policy.preset,
                "allow_external_effects": self.execution_policy.allow_external_effects,
                "require_approval": self.execution_policy.require_approval,
                "approval_mode": self.execution_policy.approval_mode,
                "auto_approve_operations": list(
                    self.execution_policy.auto_approve_operations
                ),
                "debug_trace": self.execution_policy.debug_trace,
                "output_style": self.execution_policy.output_style,
            },
            "metadata": dict(self.metadata),
            "orchestration_run_id": self.orchestration_run_id,
        }

    def to_state(self) -> dict[str, Any]:
        """Return the canonical request contract for durable state."""

        return {"request_snapshot": self.to_dict()}

    def to_request(
        self,
        *,
        owner_id: str,
        session_id: str,
        agent_id: str,
        messages: list[Message] | None = None,
        orchestration_run_id: str | None = None,
    ) -> RunRequest:
        return RunRequest(
            owner_id=owner_id,
            session_id=session_id,
            agent_id=agent_id,
            messages=list(self.messages if messages is None else messages),
            system_prompt=self.system_prompt,
            model=self.model,
            tools=list(self.tools),
            tool_context=dict(self.tool_context),
            execution_options=self.execution_options,
            execution_policy=self.execution_policy,
            metadata=dict(self.metadata),
            orchestration_run_id=(
                self.orchestration_run_id
                if orchestration_run_id is None else orchestration_run_id
            ),
        )


@dataclass(frozen=True)
class ResumeInput:
    """Explicit input used to continue a suspended operation."""

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise RuntimeValidationError("resume input kind must not be empty")


def _messages_to_dict(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.role,
            "content": message.content,
            "tool_call_id": message.tool_call_id,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": dict(call.arguments),
                }
                for call in message.tool_calls
            ],
        }
        for message in messages
    ]


def _messages_from_dict(value: Any) -> list[Message]:
    if not isinstance(value, list):
        return []
    messages: list[Message] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        messages.append(Message(
            role=str(item.get("role", "user")),
            content=item.get("content", ""),
            tool_call_id=str(item.get("tool_call_id", "")),
            tool_calls=tuple(
                _tool_call_from_dict(call)
                for call in item.get("tool_calls", [])
                if isinstance(call, dict)
            ),
        ))
    return messages


def _tool_call_from_dict(value: dict[str, Any]):
    from .messages import ToolCall

    return ToolCall(
        call_id=str(value.get("call_id", "")),
        name=str(value.get("name", "")),
        arguments=dict(value.get("arguments", {}))
        if isinstance(value.get("arguments", {}), dict) else {},
    )


def _tools_to_dict(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": dict(tool.input_schema),
            "recovery_policy": tool.recovery_policy.value,
            "requires_approval": tool.requires_approval,
            "side_effect": tool.side_effect,
            "policy_class": tool.policy_class.value,
        }
        for tool in tools
    ]


def _tools_from_dict(value: Any) -> list[ToolSpec]:
    if not isinstance(value, list):
        return []
    return [
        ToolSpec(
            name=str(item.get("name", "")),
            description=str(item.get("description", "")),
            input_schema=dict(item.get("input_schema", {}))
            if isinstance(item.get("input_schema", {}), dict) else {},
            recovery_policy=item.get("recovery_policy", "manual"),
            requires_approval=bool(item.get("requires_approval", False)),
            side_effect=bool(item.get("side_effect", False)),
            policy_class=item.get("policy_class", "read_only"),
        )
        for item in value
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    ]


def _execution_options_from_dict(value: Any) -> ExecutionOptions:
    if not isinstance(value, dict):
        return ExecutionOptions()
    allowed = {
        "max_turns", "max_tool_calls", "timeout_seconds",
        "max_model_retries", "thinking_level", "tool_execution_mode",
    }
    return ExecutionOptions(**{
        key: item for key, item in value.items() if key in allowed
    })


def _execution_policy_from_dict(value: Any) -> ExecutionPolicy:
    if not isinstance(value, dict):
        return ExecutionPolicy()
    allowed = {
        "mode", "preset", "allow_external_effects", "require_approval",
        "approval_mode", "auto_approve_operations", "debug_trace", "output_style",
    }
    data = {key: item for key, item in value.items() if key in allowed}
    if "auto_approve_operations" in data:
        data["auto_approve_operations"] = tuple(data["auto_approve_operations"])
    return ExecutionPolicy(**data)
