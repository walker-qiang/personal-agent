"""Tool execution values and recovery metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RecoveryPolicy(str, Enum):
    """How an unfinished external tool effect may be recovered."""

    REPLAYABLE = "replayable"
    IDEMPOTENT = "idempotent"
    MANUAL = "manual"


@dataclass(frozen=True)
class ToolSpec:
    """The resolved tool contract handed to a runtime operation."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    recovery_policy: RecoveryPolicy = RecoveryPolicy.MANUAL
    requires_approval: bool = False
    side_effect: bool = False


@dataclass(frozen=True)
class ToolRequest:
    """A single tool invocation independent of ToolRegistry."""

    operation_id: str
    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""


@dataclass(frozen=True)
class ToolResult:
    """A normalized tool result returned to the model loop."""

    call_id: str
    name: str
    result: Any = None
    error: str = ""
    is_error: bool = False
