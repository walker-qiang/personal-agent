"""Tool execution values and recovery metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .policy import EffectGrant


class RecoveryPolicy(str, Enum):
    """How an unfinished external tool effect may be recovered."""

    REPLAYABLE = "replayable"
    IDEMPOTENT = "idempotent"
    MANUAL = "manual"


class ToolPolicyClass(str, Enum):
    """Capability class used by the unified execution policy boundary."""

    UNCLASSIFIED = "unclassified"
    READ_ONLY = "read_only"
    EXTERNAL_READ = "external_read"
    CODE_EXECUTION = "code_execution"
    DURABLE_WRITE = "durable_write"
    AGENT_DELEGATION = "agent_delegation"


@dataclass(frozen=True)
class ToolSpec:
    """The resolved tool contract handed to a runtime operation."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    recovery_policy: RecoveryPolicy = RecoveryPolicy.MANUAL
    requires_approval: bool = False
    side_effect: bool = False
    policy_class: ToolPolicyClass = ToolPolicyClass.READ_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "recovery_policy", RecoveryPolicy(self.recovery_policy))
        object.__setattr__(self, "policy_class", ToolPolicyClass(self.policy_class))


@dataclass(frozen=True)
class ToolRequest:
    """A single tool invocation independent of ToolRegistry."""

    operation_id: str
    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    approval_grant: "EffectGrant | None" = None


@dataclass(frozen=True)
class ToolResult:
    """A normalized tool result returned to the model loop."""

    call_id: str
    name: str
    result: Any = None
    error: str = ""
    is_error: bool = False
