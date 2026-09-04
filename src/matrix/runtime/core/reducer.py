"""Pure operation transition rules."""

from __future__ import annotations

from dataclasses import replace

from ..domain.errors import RuntimeValidationError
from ..domain.operations import OperationPhase, OperationState


_ALLOWED_TRANSITIONS: dict[OperationPhase, frozenset[OperationPhase]] = {
    OperationPhase.CREATED: frozenset({OperationPhase.PREPARING, OperationPhase.ABORTED, OperationPhase.RECOVERY_REQUIRED}),
    OperationPhase.PREPARING: frozenset({
        OperationPhase.REQUESTING_MODEL,
        OperationPhase.FAILED,
        OperationPhase.ABORTED,
        OperationPhase.RECOVERY_REQUIRED,
    }),
    OperationPhase.REQUESTING_MODEL: frozenset({
        OperationPhase.EXECUTING_TOOLS,
        OperationPhase.PREPARING_NEXT_TURN,
        OperationPhase.COMPLETED,
        OperationPhase.FAILED,
        OperationPhase.ABORTED,
        OperationPhase.RECOVERY_REQUIRED,
    }),
    OperationPhase.EXECUTING_TOOLS: frozenset({
        OperationPhase.PREPARING_NEXT_TURN,
        OperationPhase.WAITING_APPROVAL,
        OperationPhase.RECOVERY_REQUIRED,
        OperationPhase.FAILED,
        OperationPhase.ABORTED,
    }),
    OperationPhase.PREPARING_NEXT_TURN: frozenset({
        OperationPhase.REQUESTING_MODEL,
        OperationPhase.COMPLETED,
        OperationPhase.ABORTED,
        OperationPhase.RECOVERY_REQUIRED,
    }),
    OperationPhase.WAITING_APPROVAL: frozenset({
        OperationPhase.WAITING_APPROVAL,
        OperationPhase.RESUMING,
        OperationPhase.ABORTED,
    }),
    OperationPhase.RESUMING: frozenset({
        OperationPhase.EXECUTING_TOOLS,
        OperationPhase.PREPARING_NEXT_TURN,
        OperationPhase.FAILED,
        OperationPhase.ABORTED,
        OperationPhase.RECOVERY_REQUIRED,
    }),
    OperationPhase.COMPLETED: frozenset(),
    OperationPhase.FAILED: frozenset(),
    OperationPhase.ABORTED: frozenset(),
    OperationPhase.RECOVERY_REQUIRED: frozenset(),
}


def validate_transition(current: OperationPhase, target: OperationPhase) -> None:
    """Raise when a phase transition is not part of the Runtime state machine."""

    if target not in _ALLOWED_TRANSITIONS[current]:
        raise RuntimeValidationError(
            f"invalid operation transition: {current.value} -> {target.value}"
        )


def with_next_phase(state: OperationState, target: OperationPhase) -> OperationState:
    """Apply one legal phase transition and increment the snapshot version."""

    validate_transition(state.phase, target)
    return replace(state, phase=target, version=state.version + 1)
