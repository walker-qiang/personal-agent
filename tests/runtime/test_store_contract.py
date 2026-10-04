from __future__ import annotations

from dataclasses import replace
import time

import pytest

from matrix.runtime import AgentRuntime, ResumeInput, RunRequest
from matrix.runtime.adapters.sqlite_store import SQLiteRuntimeStore
from matrix.runtime.core.reducer import with_next_phase
from matrix.runtime.domain.approvals import Approval, ApprovalSet, ApprovalStatus
from matrix.runtime.domain.errors import OperationConflictError
from matrix.runtime.domain.events import RuntimeEventType
from matrix.runtime.domain.messages import Message
from matrix.runtime.domain.operations import OperationPhase, OperationState, StateTransition
from matrix.runtime.domain.results import RunOutcome
from matrix.runtime.domain.tools import RecoveryPolicy, ToolSpec
from matrix.runtime.ports.model import ModelResponse
from matrix.runtime.testing.fake_model import FakeModel, tool_call
from matrix.runtime.testing.fake_tools import FakeToolExecutor
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


def test_memory_store_recovers_interrupted_operation_but_preserves_approval_wait() -> None:
    store = MemoryOperationStore()
    interrupted = _operation("op-interrupted")
    waiting = _operation("op-waiting", owner_id="user-a")
    waiting = replace(waiting, session_id="session-waiting", phase=OperationPhase.WAITING_APPROVAL)
    store.create(interrupted)
    store.create(waiting)

    recovered = store.recover_incomplete()

    assert [item.operation_id for item in recovered] == ["op-interrupted"]
    assert store.load("user-a", "op-interrupted").phase is OperationPhase.RECOVERY_REQUIRED
    assert store.load("user-a", "op-waiting").phase is OperationPhase.WAITING_APPROVAL
    assert store.event_list("op-interrupted")[-1].event_type is RuntimeEventType.RECOVERY_REQUIRED


def test_sqlite_store_recovers_interrupted_operation_durably(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime-recovery.db")
    operation = _operation()
    store.create(operation)
    store.commit(StateTransition(
        previous_version=operation.version,
        new_state=with_next_phase(operation, OperationPhase.PREPARING),
    ))

    recovered = store.recover_incomplete(reason="test_restart")

    assert [item.operation_id for item in recovered] == ["op-1"]
    current = store.load("user-a", "op-1")
    assert current.phase is OperationPhase.RECOVERY_REQUIRED
    assert current.state["runtime_recovery"]["reason"] == "test_restart"
    events = store.list_events("user-a", "op-1")
    assert events[-1].event_type is RuntimeEventType.RECOVERY_REQUIRED
    assert events[-1].payload["previous_phase"] == "preparing"
    assert store.recover_incomplete() == []
    store.close()


def _suspend_sqlite(store: SQLiteRuntimeStore) -> tuple[str, str]:
    runtime = AgentRuntime(
        store,
        model=FakeModel([
            ModelResponse(
                tool_calls=(tool_call("sqlite-call", "write", {"value": 1}),),
                finish_reason="tool_calls",
            ),
        ]),
        tools=FakeToolExecutor({"write": lambda args: {"ok": True}}),
    )
    handle = runtime.start(RunRequest(
        owner_id="owner-a",
        session_id="sqlite-approval",
        agent_id="assistant",
        messages=[Message(role="user", content="write")],
        tools=[ToolSpec(
            name="write",
            requires_approval=True,
            recovery_policy=RecoveryPolicy.MANUAL,
        )],
    ))
    assert handle.result().outcome is RunOutcome.SUSPENDED
    operation = store.load("owner-a", handle.operation_id)
    approval_id = operation.state["pending_tool_calls"][0]["approval_id"]
    return handle.operation_id, approval_id


def test_sqlite_approval_resume_is_durable_and_effect_is_settled(tmp_path) -> None:
    db_path = tmp_path / "runtime.db"
    store = SQLiteRuntimeStore(db_path)
    operation_id, approval_id = _suspend_sqlite(store)
    store.close()

    store = SQLiteRuntimeStore(db_path)
    with pytest.raises(OperationConflictError, match="owner mismatch"):
        AgentRuntime(store, FakeModel(), FakeToolExecutor()).resume(
            "owner-b",
            operation_id,
            ResumeInput(
                kind="approval",
                payload={
                    "approval_set_id": store.get_approval("owner-a", approval_id).approval_set_id,
                    "decisions": {approval_id: "approve"},
                },
            ),
        )
    tools = FakeToolExecutor({"write": lambda args: {"ok": True}})
    result = AgentRuntime(
        store,
        FakeModel([ModelResponse(content="done")]),
        tools,
    ).resume(
        "owner-a",
        operation_id,
        ResumeInput(
            kind="approval",
            payload={
                "approval_set_id": store.get_approval("owner-a", approval_id).approval_set_id,
                "decisions": {approval_id: "approve"},
            },
        ),
    ).result()

    effect = store._get_conn().execute(
        "SELECT status, recovery_policy FROM runtime_tool_effects "
        "WHERE operation_id=? AND tool_call_id=?",
        (operation_id, "sqlite-call"),
    ).fetchone()
    assert result.outcome is RunOutcome.COMPLETED
    assert len(tools.requests) == 1
    assert tuple(effect) == ("settled", "manual")
    assert store.get_approval("owner-a", approval_id).status is ApprovalStatus.APPROVED
    with pytest.raises(OperationConflictError, match="not waiting"):
        AgentRuntime(store, FakeModel(), FakeToolExecutor()).resume(
            "owner-a",
            operation_id,
            ResumeInput(
                kind="approval",
                payload={
                    "approval_set_id": store.get_approval("owner-a", approval_id).approval_set_id,
                    "decisions": {approval_id: "approve"},
                },
            ),
        )
    store.close()


def test_sqlite_multi_approval_resume_survives_restart_and_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "runtime-multi-approval.db"
    initial_store = SQLiteRuntimeStore(db_path)
    initial_model = FakeModel([
        ModelResponse(
            tool_calls=(
                tool_call("call-a", "write-a", {"value": "a"}),
                tool_call("call-b", "write-b", {"value": "b"}),
            ),
            finish_reason="tool_calls",
        ),
    ])
    initial_tools = FakeToolExecutor({
        "write-a": lambda args: {"ok": "a"},
        "write-b": lambda args: {"ok": "b"},
    })
    request = RunRequest(
        owner_id="owner-a",
        session_id="sqlite-multi-approval",
        agent_id="assistant",
        messages=[Message(role="user", content="write both")],
        system_prompt="persisted system prompt",
        model="persisted-model",
        tools=[
            ToolSpec(
                name="write-a",
                requires_approval=True,
                recovery_policy=RecoveryPolicy.MANUAL,
            ),
            ToolSpec(
                name="write-b",
                requires_approval=True,
                recovery_policy=RecoveryPolicy.MANUAL,
            ),
        ],
    )

    suspended = AgentRuntime(
        initial_store, initial_model, initial_tools,
    ).start(request).result()
    assert suspended.outcome is RunOutcome.SUSPENDED
    assert suspended.suspension is not None
    operation = initial_store.load("owner-a", suspended.operation_id)
    assert operation is not None
    approval_set_id = suspended.suspension.approval_set_id
    approvals = initial_store.list_approval_set("owner-a", approval_set_id)
    assert len(approvals) == 2
    assert operation.state["request_snapshot"]["system_prompt"] == "persisted system prompt"
    assert operation.state["request_snapshot"]["model"] == "persisted-model"
    initial_store.close()

    partial_store = SQLiteRuntimeStore(db_path)
    partial_operation = partial_store.load("owner-a", suspended.operation_id)
    assert partial_operation is not None
    partial_set = partial_store.get_approval_set("owner-a", approval_set_id)
    assert partial_set is not None
    partial = AgentRuntime(
        partial_store, FakeModel(), FakeToolExecutor(),
    ).resume(
        "owner-a",
        suspended.operation_id,
        ResumeInput(
            kind="approval",
            payload={
                "approval_set_id": approval_set_id,
                "approval_ids": [approvals[0].approval_id],
                "decisions": {approvals[0].approval_id: "approve"},
                "expected_operation_version": partial_operation.version,
                "expected_approval_set_version": partial_set.version,
                "idempotency_key": "partial-approval",
            },
        ),
    ).result()
    assert partial.outcome is RunOutcome.SUSPENDED
    assert [
        item.status for item in partial_store.list_approval_set(
            "owner-a", approval_set_id,
        )
    ] == [ApprovalStatus.APPROVED, ApprovalStatus.PENDING]
    partial_operation = partial_store.load("owner-a", suspended.operation_id)
    partial_set = partial_store.get_approval_set("owner-a", approval_set_id)
    assert partial_operation is not None
    assert partial_set is not None
    partial_store.close()

    replay_store = SQLiteRuntimeStore(db_path)
    replay = AgentRuntime(
        replay_store, FakeModel(), FakeToolExecutor(),
    ).resume(
        "owner-a",
        suspended.operation_id,
        ResumeInput(
            kind="approval",
            payload={
                "approval_set_id": approval_set_id,
                "approval_ids": [approvals[0].approval_id],
                "decisions": {approvals[0].approval_id: "approve"},
                "expected_operation_version": partial_operation.version,
                "expected_approval_set_version": partial_set.version,
                "idempotency_key": "partial-approval",
            },
        ),
    ).result()
    assert replay.outcome is RunOutcome.SUSPENDED
    assert replay.suspension is not None
    assert replay.suspension.approval_ids == [approvals[1].approval_id]
    replay_operation = replay_store.load("owner-a", suspended.operation_id)
    replay_set = replay_store.get_approval_set("owner-a", approval_set_id)
    assert replay_operation is not None
    assert replay_set is not None
    replay_store.close()

    final_store = SQLiteRuntimeStore(db_path)
    final_model = FakeModel([ModelResponse(content="done")])
    final_tools = FakeToolExecutor({
        "write-a": lambda args: {"ok": "a"},
        "write-b": lambda args: {"ok": "b"},
    })
    final = AgentRuntime(
        final_store, final_model, final_tools,
    ).resume(
        "owner-a",
        suspended.operation_id,
        ResumeInput(
            kind="approval",
            payload={
                "approval_set_id": approval_set_id,
                "approval_ids": [approvals[1].approval_id],
                "decisions": {approvals[1].approval_id: "skip"},
                "expected_operation_version": replay_operation.version,
                "expected_approval_set_version": replay_set.version,
                "idempotency_key": "final-approval",
            },
        ),
    ).result()
    assert final.outcome is RunOutcome.COMPLETED
    assert final.final_message == "done"
    assert [request.call_id for request in final_tools.requests] == ["call-a"]
    assert final_model.requests[0].system_prompt == "persisted system prompt"
    assert final_model.requests[0].model == "persisted-model"
    assert final_store.load("owner-a", suspended.operation_id).phase is OperationPhase.COMPLETED
    final_store.close()


def test_sqlite_lists_all_dag_waiting_approval_operations_for_session(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime-dag-approvals.db")
    for step_id in ("1", "2"):
        operation = replace(
            _operation(f"dag-op-{step_id}"),
            session_id="dag-session",
            phase=OperationPhase.WAITING_APPROVAL,
            orchestration_run_id="dag-run",
            operation_scope="dag_step",
            step_id=step_id,
        )
        store.create(operation)
        approval_set_id = f"set-{step_id}"
        store.create_approval_set(
            ApprovalSet(
                approval_set_id=approval_set_id,
                owner_id="user-a",
                operation_id=operation.operation_id,
                orchestration_run_id="dag-run",
            )
        )
        store.create_approval(
            Approval(
                approval_id=f"approval-{step_id}",
                owner_id="user-a",
                operation_id=operation.operation_id,
                tool_call_id=f"call-{step_id}",
                tool_name="write",
                approval_set_id=approval_set_id,
            )
        )

    waiting = store.list_waiting_approvals("user-a", "dag-session")

    assert {item.operation_id for item in waiting} == {"dag-op-1", "dag-op-2"}
    assert all(item.operation_scope == "dag_step" for item in waiting)
    store.close()

    reopened = SQLiteRuntimeStore(tmp_path / "runtime-dag-approvals.db")
    assert reopened.recover_incomplete(reason="test_restart") == []
    restored = reopened.list_waiting_approvals("user-a", "dag-session")
    assert {item.operation_id for item in restored} == {"dag-op-1", "dag-op-2"}
    reopened.close()


def test_sqlite_expired_approval_never_executes_effect(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "runtime-expired.db")
    operation_id, approval_id = _suspend_sqlite(store)
    store._get_conn().execute(
        "UPDATE runtime_approvals SET expires_at=? WHERE approval_id=?",
        (time.time() - 1, approval_id),
    )
    store._get_conn().commit()
    tools = FakeToolExecutor({"write": lambda args: {"ok": True}})

    result = AgentRuntime(
        store,
        FakeModel([ModelResponse(content="must not run")]),
        tools,
    ).resume(
        "owner-a",
        operation_id,
        ResumeInput(
            kind="approval",
            payload={
                "approval_set_id": store.get_approval("owner-a", approval_id).approval_set_id,
                "decisions": {approval_id: "approve"},
            },
        ),
    ).result()

    effect_count = store._get_conn().execute(
        "SELECT count(*) FROM runtime_tool_effects WHERE operation_id=?",
        (operation_id,),
    ).fetchone()[0]
    assert result.outcome is RunOutcome.ABORTED
    assert tools.requests == []
    assert effect_count == 0
    assert store.get_approval("owner-a", approval_id).status is ApprovalStatus.EXPIRED
    assert store.load("owner-a", operation_id).phase is OperationPhase.ABORTED
    store.close()


def test_sqlite_branch_summary_entry_is_idempotent_and_status_filtered(tmp_path) -> None:
    store = SQLiteRuntimeStore(tmp_path / "branch-summary.db")

    entry_id, payload = store.ensure_branch_summary_entry(
        "owner-a", "session-a", "from-1", "leaf-1", 2,
    )
    duplicate_id, duplicate_payload = store.ensure_branch_summary_entry(
        "owner-a", "session-a", "from-1", "leaf-1", 99,
    )

    assert duplicate_id == entry_id
    assert duplicate_payload["message_count"] == 2
    assert payload["status"] == "scheduled"
    assert [item["entry_id"] for item in store.list_pending_session_entries(
        "branch_summary", ("scheduled", "running"),
    )] == [entry_id]

    payload["status"] = "completed"
    assert store.update_session_entry("owner-a", entry_id, payload)
    assert store.list_pending_session_entries(
        "branch_summary", ("scheduled", "running"),
    ) == []
    store.close()
