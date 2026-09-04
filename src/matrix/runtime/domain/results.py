"""Results returned by a Runtime operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .tools import ToolResult


class RunOutcome(str, Enum):
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    FAILED = "failed"
    ABORTED = "aborted"
    RECOVERY_REQUIRED = "recovery_required"


@dataclass(frozen=True)
class Suspension:
    """Information needed by an adapter to display a pending suspension."""

    reason: str
    approval_id: str = ""
    approval_set_id: str = ""
    approval_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    """Stable result shape shared by in-memory and persistent runtimes."""

    outcome: RunOutcome
    operation_id: str
    final_message: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    suspension: Suspension | None = None
