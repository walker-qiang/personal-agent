"""Durable, ordered Runtime events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimeEventType(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    MESSAGE_START = "message_start"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_END = "message_end"
    TOOL_START = "tool_start"
    TOOL_UPDATE = "tool_update"
    TOOL_END = "tool_end"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_DECIDED = "approval_decided"
    RETRY_SCHEDULED = "retry_scheduled"
    RETRY_START = "retry_start"
    RETRY_END = "retry_end"
    RUN_SUSPENDED = "run_suspended"
    RUN_RESUMED = "run_resumed"
    RUN_FAILED = "run_failed"
    RUN_ABORTED = "run_aborted"
    RECOVERY_REQUIRED = "recovery_required"


@dataclass(frozen=True)
class RuntimeEvent:
    """An event envelope whose sequence is scoped to one operation."""

    event_id: str
    owner_id: str
    operation_id: str
    session_id: str
    sequence: int
    event_type: RuntimeEventType
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("runtime event sequence must start at 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "owner_id": self.owner_id,
            "operation_id": self.operation_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
