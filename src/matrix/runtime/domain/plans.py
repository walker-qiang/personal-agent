"""Runtime execution plan values."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .policy import PolicyDecision, ToolExecutionContext
from .tools import ToolRequest, ToolSpec


class PlanStatus(str, Enum):
    """Whether a compiled plan can be handed to the execution gateway."""

    READY = "ready"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class PlanStep:
    """One policy-evaluated tool invocation in an execution plan."""

    step_id: str
    request: ToolRequest
    tool: ToolSpec
    decision: PolicyDecision


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable plan produced before any tool handler is invoked."""

    operation_id: str
    source: str
    context: ToolExecutionContext
    steps: tuple[PlanStep, ...]
    status: PlanStatus
    reason: str = ""

    @property
    def ready(self) -> bool:
        return self.status is PlanStatus.READY

    @property
    def requires_approval(self) -> bool:
        return self.status is PlanStatus.REQUIRE_APPROVAL
