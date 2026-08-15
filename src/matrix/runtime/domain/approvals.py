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
