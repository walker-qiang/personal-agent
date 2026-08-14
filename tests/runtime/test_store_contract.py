from __future__ import annotations

import pytest

from matrix.runtime.core.reducer import with_next_phase
from matrix.runtime.domain.errors import OperationConflictError
from matrix.runtime.domain.operations import OperationPhase, OperationState, StateTransition
from matrix.runtime.testing.memory_store import MemoryOperationStore


def _operation(operation_id: str = "op-1", owner_id: str = "user-a") -> OperationState:
    return OperationState(
        operation_id=operation_id,
        owner_id=owner_id,
        session_id="session-1",
        agent_id="assistant",
    )


def test_memory_store_enforces_owner_and_single_active_operation() -> None:
    store = MemoryOperationStore()
    operation = _operation()
    store.create(operation)

    assert store.load("other-user", operation.operation_id) is None
    with pytest.raises(OperationConflictError, match="active operation"):
        store.create(_operation("op-2"))


def test_memory_store_commit_is_compare_and_set() -> None:
    store = MemoryOperationStore()
    operation = _operation()
    store.create(operation)
    next_state = with_next_phase(operation, OperationPhase.PREPARING)

    store.commit(StateTransition(previous_version=0, new_state=next_state))
    assert store.load("user-a", "op-1").phase is OperationPhase.PREPARING

    with pytest.raises(OperationConflictError, match="version conflict"):
        store.commit(StateTransition(previous_version=0, new_state=next_state))


def test_memory_store_lists_only_incomplete_operations() -> None:
    store = MemoryOperationStore()
    operation = _operation()
    store.create(operation)
    failed = OperationState(
        operation_id="op-2",
        owner_id="user-a",
        session_id="session-2",
        agent_id="assistant",
        phase=OperationPhase.FAILED,
    )
    store.create(failed)

    assert [item.operation_id for item in store.list_incomplete()] == ["op-1"]
