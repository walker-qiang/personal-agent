"""Human approval values owned by the Runtime domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    SKIPPED = "skipped"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    SKIP = "skip"


class ApprovalSetStatus(str, Enum):
    PENDING = "pending"
    PARTIALLY_DECIDED = "partially_decided"
    APPROVED = "approved"
    SKIPPED = "skipped"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ApprovalSet:
    """Durable decision boundary for one suspended operation or workflow step."""

    approval_set_id: str
    owner_id: str
    operation_id: str
    orchestration_run_id: str = ""
    status: ApprovalSetStatus = ApprovalSetStatus.PENDING
    version: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    last_idempotency_key: str = ""


@dataclass(frozen=True)
class Approval:
    approval_id: str
    owner_id: str
    operation_id: str
    tool_call_id: str
    tool_name: str
    sanitized_arguments: dict[str, Any] = field(default_factory=dict)
    risk: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    decision: ApprovalDecision | None = None
    expires_at: float | None = None
    version: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    approval_set_id: str = ""
    decided_by: str = ""
    decided_at: float | None = None
    decision_source: str = ""
    idempotency_key: str = ""


# The Runtime uses Approval for the persisted request shape.  Keep the
# domain name available for callers that prefer the workflow terminology.
ApprovalRequest = Approval
