"""Operation lifecycle and atomic transition values."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .events import RuntimeEvent


class OperationPhase(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    REQUESTING_MODEL = "requesting_model"
    EXECUTING_TOOLS = "executing_tools"
    PREPARING_NEXT_TURN = "preparing_next_turn"
    WAITING_APPROVAL = "waiting_approval"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    RECOVERY_REQUIRED = "recovery_required"


TERMINAL_PHASES = frozenset({
    OperationPhase.COMPLETED,
    OperationPhase.FAILED,
    OperationPhase.ABORTED,
    OperationPhase.RECOVERY_REQUIRED,
})


@dataclass(frozen=True)
class OperationState:
    """Complete operation snapshot; it is not reconstructed from events."""

    operation_id: str
    owner_id: str
    session_id: str
    agent_id: str
    phase: OperationPhase = OperationPhase.CREATED
    turn_index: int = 0
    version: int = 0
    state_schema_version: int = 1
    last_event_sequence: int = 0
    state: dict[str, Any] = field(default_factory=dict)
    orchestration_run_id: str = ""
    operation_scope: str = "top_level"
    step_id: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.phase in TERMINAL_PHASES


@dataclass(frozen=True)
class StateTransition:
    """Atomic store command containing the next snapshot and new events."""

    previous_version: int
    new_state: OperationState
    events: tuple[RuntimeEvent, ...] = ()


def with_phase(state: OperationState, phase: OperationPhase) -> OperationState:
    """Return a new snapshot with a monotonically increasing version."""

    return replace(state, phase=phase, version=state.version + 1)
