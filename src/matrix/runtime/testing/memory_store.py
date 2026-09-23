"""In-memory operation store used by contract tests and future loop tests."""

from __future__ import annotations

from dataclasses import replace
import time
import uuid

from ..domain.approvals import (
    Approval,
    ApprovalDecision,
    ApprovalSet,
    ApprovalSetStatus,
    ApprovalStatus,
)
from ..domain.events import RuntimeEvent, RuntimeEventType
from ..domain.tools import RecoveryPolicy, ToolRequest, ToolResult

from ..domain.errors import OperationConflictError
from ..domain.operations import (
    OperationPhase,
    OperationState,
    StateTransition,
    aggregate_orchestration_status,
)


class MemoryOperationStore:
    def __init__(self) -> None:
        self.operations: dict[str, OperationState] = {}
        self.events: dict[str, list[object]] = {}
        self.approvals: dict[str, Approval] = {}
        self.approval_sets: dict[str, ApprovalSet] = {}
        self.orchestration_runs: dict[str, dict] = {}
        self.orchestration_steps: dict[tuple[str, str], dict] = {}
        self.effects: dict[tuple[str, str], dict] = {}
        self.session_entries: list[dict] = []
        self._operation_order: dict[str, int] = {}
        self._next_operation_order = 0

    def append_session_entry(self, owner_id, session_id, entry_type, payload):
        entry = {
            "entry_id": uuid.uuid4().hex,
            "owner_id": owner_id,
            "session_id": session_id,
            "entry_type": entry_type,
            "payload": dict(payload),
            "created_at": time.time(),
        }
        self.session_entries.append(entry)
        return entry["entry_id"]

    def list_session_entries(self, owner_id, session_id, entry_type="", limit=20):
        values = [
            item for item in self.session_entries
            if item["owner_id"] == owner_id and item["session_id"] == session_id
            and (not entry_type or item["entry_type"] == entry_type)
        ]
        values.sort(key=lambda item: item["created_at"], reverse=True)
        return values[:limit]

    def delete_session_entries(self, owner_id, session_id):
        before = len(self.session_entries)
        self.session_entries = [
            item for item in self.session_entries
            if not (item["owner_id"] == owner_id and item["session_id"] == session_id)
        ]
        return before - len(self.session_entries)

    def create(self, operation: OperationState) -> None:
        if operation.operation_id in self.operations:
            raise OperationConflictError(f"operation already exists: {operation.operation_id}")
        if operation.operation_scope == "top_level" and self.has_active(operation.owner_id, operation.session_id):
            raise OperationConflictError(
                f"session already has an active operation: {operation.session_id}"
            )
        if operation.operation_scope == "dag_step" and operation.orchestration_run_id:
            if any(
                item.operation_scope == "dag_step"
                and item.orchestration_run_id == operation.orchestration_run_id
                and item.step_id == operation.step_id
                for item in self.operations.values()
            ):
                raise OperationConflictError("DAG step operation already exists")
        self.operations[operation.operation_id] = operation
        self.events[operation.operation_id] = []
        self._operation_order[operation.operation_id] = self._next_operation_order
        self._next_operation_order += 1

    def load(self, owner_id: str, operation_id: str) -> OperationState | None:
        operation = self.operations.get(operation_id)
        if operation is None or operation.owner_id != owner_id:
            return None
        return operation

    def commit(self, transition: StateTransition) -> None:
        current = self.operations.get(transition.new_state.operation_id)
        if current is None:
            raise OperationConflictError(
                f"operation does not exist: {transition.new_state.operation_id}"
            )
        if current.version != transition.previous_version:
            raise OperationConflictError(
                f"operation version conflict: expected {transition.previous_version}, "
                f"actual {current.version}"
            )
        if transition.new_state.version != transition.previous_version + 1:
            raise OperationConflictError("operation version must increase by one")
        if transition.new_state.owner_id != current.owner_id:
            raise OperationConflictError("operation owner cannot change")
        self.operations[current.operation_id] = transition.new_state
        self.events[current.operation_id].extend(transition.events)
        self._sync_orchestration_projection(transition.new_state)

    def list_incomplete(self) -> list[OperationState]:
        return [operation for operation in self.operations.values() if not operation.is_terminal]

    def recover_incomplete(self, reason: str = "process_restart") -> list[OperationState]:
        recovered: list[OperationState] = []
        for operation in list(self.operations.values()):
            unresolved_effects = [
                {
                    "tool_call_id": call_id,
                    "tool_name": effect.get("name", ""),
                    "recovery_policy": effect.get("policy", ""),
                    "idempotency_key": effect.get("idempotency_key", ""),
                }
                for (operation_id, call_id), effect in self.effects.items()
                if operation_id == operation.operation_id
                and effect.get("status") == "executing"
            ]
            if (
                (operation.is_terminal and not unresolved_effects)
                or (
                    operation.phase is OperationPhase.WAITING_APPROVAL
                    and not unresolved_effects
                )
            ):
                continue
            for (operation_id, _), effect in self.effects.items():
                if (
                    operation_id == operation.operation_id
                    and effect.get("status") == "executing"
                ):
                    effect["status"] = "recovery_required"
                    effect["error"] = (
                        f"tool effect interrupted during {reason}; "
                        "external outcome is unknown"
                    )
            state = dict(operation.state)
            state["runtime_recovery"] = {
                "reason": reason,
                "previous_phase": operation.phase.value,
                "unresolved_tool_effects": unresolved_effects,
            }
            event = RuntimeEvent(
                event_id=uuid.uuid4().hex,
                owner_id=operation.owner_id,
                operation_id=operation.operation_id,
                session_id=operation.session_id,
                sequence=operation.last_event_sequence + 1,
                event_type=RuntimeEventType.RECOVERY_REQUIRED,
                timestamp=time.time(),
                payload={"reason": reason, "previous_phase": operation.phase.value},
            )
            recovered_operation = replace(
                operation,
                phase=OperationPhase.RECOVERY_REQUIRED,
                version=operation.version + 1,
                last_event_sequence=event.sequence,
                state=state,
            )
            self.operations[operation.operation_id] = recovered_operation
            self.events[operation.operation_id].append(event)
            self._sync_orchestration_projection(recovered_operation)
            recovered.append(recovered_operation)
        return recovered

    def list_operations(
        self,
        owner_id: str,
        session_id: str | None = None,
        phase: str | None = None,
        limit: int = 50,
    ) -> list[OperationState]:
        values = [
            operation for operation in self.operations.values()
            if operation.owner_id == owner_id
            and (session_id is None or operation.session_id == session_id)
            and (phase is None or operation.phase.value == phase)
        ]
        values.sort(key=lambda operation: operation.updated_at, reverse=True)
        return values[:limit]

    def list_approvals(
        self,
        owner_id: str,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Approval]:
        values = [
            approval for approval in self.approvals.values()
            if approval.owner_id == owner_id
            and (status is None or approval.status.value == status)
            and (
                session_id is None
                or self.operations.get(approval.operation_id) is not None
                and self.operations[approval.operation_id].session_id == session_id
            )
        ]
        values.sort(key=lambda approval: approval.updated_at, reverse=True)
        return values[:limit]

    def has_active(self, owner_id: str, session_id: str) -> bool:
        return any(
            operation.owner_id == owner_id
            and operation.session_id == session_id
            and not operation.is_terminal
            for operation in self.operations.values()
        )

    def find_active(self, owner_id: str, session_id: str) -> OperationState | None:
        for operation in self.operations.values():
            if (
                operation.owner_id == owner_id
                and operation.session_id == session_id
                and operation.operation_scope == "top_level"
                and not operation.is_terminal
            ):
                return operation
        return None

    def find_waiting_approval(self, owner_id: str, session_id: str) -> OperationState | None:
        values = self.list_waiting_approvals(owner_id, session_id, limit=1)
        return values[0] if values else None

    def list_waiting_approvals(self, owner_id: str, session_id: str, limit: int = 50):
        candidates = [
            operation for operation in self.operations.values()
            if operation.owner_id == owner_id
            and operation.session_id == session_id
            and operation.phase is OperationPhase.WAITING_APPROVAL
            and any(
                approval.operation_id == operation.operation_id
                and approval.status is ApprovalStatus.PENDING
                for approval in self.approvals.values()
            )
        ]
        candidates.sort(key=lambda operation: operation.updated_at, reverse=True)
        return candidates[:max(1, min(limit, 200))]

    def ensure_orchestration_run(
        self, run_id, owner_id, session_id, graph_thread_id="", metadata=None,
    ):
        if not run_id:
            return
        current = self.orchestration_runs.get(run_id)
        if current is not None:
            self.orchestration_runs[run_id] = {
                **current,
                "owner_id": owner_id,
                "session_id": session_id,
                "graph_thread_id": graph_thread_id or current.get("graph_thread_id", ""),
                "metadata": (
                    dict(metadata)
                    if metadata is not None
                    else dict(current.get("metadata", {}))
                ),
            }
            return
        self.orchestration_runs[run_id] = {
            "run_id": run_id, "owner_id": owner_id, "session_id": session_id,
            "graph_thread_id": graph_thread_id, "status": "running",
            "metadata": dict(metadata or {}),
        }

    def get_orchestration_run(self, owner_id, run_id):
        value = self.orchestration_runs.get(run_id)
        if value is None or value.get("owner_id") != owner_id:
            return None
        return dict(value)

    def list_orchestration_steps(self, owner_id, run_id):
        if self.get_orchestration_run(owner_id, run_id) is None:
            return []
        values = [
            dict(value)
            for (candidate_run_id, _), value in self.orchestration_steps.items()
            if candidate_run_id == run_id
        ]
        values.sort(key=lambda item: item.get("step_id", ""))
        return values

    def reconcile_orchestration_runs(self) -> list[dict]:
        """Rebuild in-memory run and step projections from operations."""

        summaries: list[dict] = []
        for run_id, run in self.orchestration_runs.items():
            step_operations: dict[str, OperationState] = {}
            top_level_operations: list[OperationState] = []
            for operation in self.operations.values():
                if operation.orchestration_run_id != run_id:
                    continue
                if operation.operation_scope == "dag_step" and operation.step_id:
                    current = step_operations.get(operation.step_id)
                    if (
                        current is None
                        or _operation_projection_key(
                            operation,
                            self._operation_order.get(operation.operation_id, -1),
                        )
                        > _operation_projection_key(
                            current,
                            self._operation_order.get(current.operation_id, -1),
                        )
                    ):
                        step_operations[operation.step_id] = operation
                elif operation.operation_scope == "top_level":
                    top_level_operations.append(operation)

            projected_keys = {
                key for key in self.orchestration_steps
                if key[0] == run_id
            }
            for step_id, operation in step_operations.items():
                key = (run_id, step_id)
                metadata = _operation_projection_metadata(operation)
                current = self.orchestration_steps.get(key)
                if current is None:
                    self.orchestration_steps[key] = {
                        "run_id": run_id,
                        "step_id": step_id,
                        "operation_id": operation.operation_id,
                        "status": operation.phase.value,
                        "version": 0,
                        "metadata": metadata,
                    }
                elif (
                    current["operation_id"] != operation.operation_id
                    or current["status"] != operation.phase.value
                    or current.get("metadata", {}) != metadata
                ):
                    self.orchestration_steps[key] = {
                        **current,
                        "operation_id": operation.operation_id,
                        "status": operation.phase.value,
                        "version": current["version"] + 1,
                        "metadata": metadata,
                    }
            for key in projected_keys - {
                (run_id, step_id) for step_id in step_operations
            }:
                del self.orchestration_steps[key]

            step_statuses = [
                operation.phase.value for operation in step_operations.values()
            ]
            if step_statuses:
                run_status = aggregate_orchestration_status(step_statuses)
            elif top_level_operations:
                latest_top_level = max(
                    top_level_operations,
                    key=lambda operation: _operation_projection_key(
                        operation,
                        self._operation_order.get(operation.operation_id, -1),
                    ),
                )
                run_status = aggregate_orchestration_status(
                    [latest_top_level.phase.value],
                )
            else:
                run_status = str(run.get("status") or "running")
            run["status"] = run_status
            summaries.append({
                "run_id": run_id,
                "owner_id": run.get("owner_id", ""),
                "session_id": run.get("session_id", ""),
                "status": run_status,
                "step_count": len(step_operations),
            })
        return summaries

    def upsert_orchestration_step(
        self, run_id, step_id, operation_id, status, metadata=None,
    ):
        if not run_id or not step_id:
            return
        key = (run_id, step_id)
        current = self.orchestration_steps.get(key, {"version": -1})
        self.orchestration_steps[key] = {
            "run_id": run_id, "step_id": step_id, "operation_id": operation_id,
            "status": status, "version": current["version"] + 1,
            "metadata": dict(metadata or {}),
        }

    def _sync_orchestration_projection(self, operation: OperationState) -> None:
        run_id = operation.orchestration_run_id
        if not run_id:
            return
        if operation.step_id:
            self.upsert_orchestration_step(
                run_id,
                operation.step_id,
                operation.operation_id,
                operation.phase.value,
            )
        terminal = {
            OperationPhase.COMPLETED.value,
            OperationPhase.FAILED.value,
            OperationPhase.ABORTED.value,
            OperationPhase.RECOVERY_REQUIRED.value,
        }
        run = self.orchestration_runs.get(run_id)
        if run is None:
            return
        if operation.operation_scope == "top_level" and operation.phase.value in terminal:
            run["status"] = operation.phase.value
            return
        if operation.operation_scope != "dag_step":
            return
        steps = [
            item for (candidate_run_id, _), item in self.orchestration_steps.items()
            if candidate_run_id == run_id
        ]
        if steps:
            run["status"] = aggregate_orchestration_status(
                item["status"] for item in steps
            )

    def event_list(self, operation_id: str) -> list[object]:
        return list(self.events.get(operation_id, []))

    def list_events(
        self,
        owner_id: str,
        operation_id: str,
        limit: int = 200,
        after_sequence: int = 0,
    ) -> list[object]:
        operation = self.operations.get(operation_id)
        if operation is None or operation.owner_id != owner_id:
            return []
        return [
            event for event in self.events.get(operation_id, [])
            if getattr(event, "sequence", 0) > after_sequence
        ][:limit]

    def create_approval(self, approval: Approval) -> None:
        if not approval.approval_set_id:
            raise OperationConflictError("approval_set_id is required")
        if approval.approval_id in self.approvals:
            raise OperationConflictError("approval already exists")
        self.approvals[approval.approval_id] = approval

    def create_approval_set(self, approval_set: ApprovalSet) -> None:
        if approval_set.approval_set_id in self.approval_sets:
            raise OperationConflictError("approval set already exists")
        self.approval_sets[approval_set.approval_set_id] = approval_set

    def get_approval(self, owner_id: str, approval_id: str) -> Approval | None:
        approval = self.approvals.get(approval_id)
        if approval is None or approval.owner_id != owner_id:
            return None
        return approval

    def get_approval_set(self, owner_id: str, approval_set_id: str) -> ApprovalSet | None:
        approval_set = self.approval_sets.get(approval_set_id)
        return (
            approval_set
            if approval_set is not None and approval_set.owner_id == owner_id
            else None
        )

    def list_approval_set(self, owner_id: str, approval_set_id: str) -> list[Approval]:
        return [
            approval for approval in self.approvals.values()
            if approval.owner_id == owner_id
            and approval.approval_set_id == approval_set_id
        ]

    def delete_session_runtime(self, owner_id: str, session_id: str) -> None:
        operation_ids = {
            operation.operation_id
            for operation in self.operations.values()
            if operation.owner_id == owner_id and operation.session_id == session_id
        }
        run_ids = {
            operation.orchestration_run_id
            for operation in self.operations.values()
            if operation.operation_id in operation_ids and operation.orchestration_run_id
        }
        self.operations = {
            key: value for key, value in self.operations.items()
            if key not in operation_ids
        }
        self._operation_order = {
            key: value for key, value in self._operation_order.items()
            if key not in operation_ids
        }
        self.events = {
            key: value for key, value in self.events.items()
            if key not in operation_ids
        }
        self.approvals = {
            key: value for key, value in self.approvals.items()
            if value.operation_id not in operation_ids
        }
        self.approval_sets = {
            key: value for key, value in self.approval_sets.items()
            if value.operation_id not in operation_ids
        }
        self.effects = {
            key: value for key, value in self.effects.items()
            if key[0] not in operation_ids
        }
        self.orchestration_steps = {
            key: value for key, value in self.orchestration_steps.items()
            if key[0] not in run_ids
        }
        self.orchestration_runs = {
            key: value for key, value in self.orchestration_runs.items()
            if key not in run_ids
        }
        self.delete_session_entries(owner_id, session_id)

    def resolve_approval_set(
        self,
        owner_id: str,
        approval_set_id: str,
        decisions: dict[str, ApprovalDecision],
        decided_at: float,
        decided_by: str,
        decision_source: str,
        idempotency_key: str,
        expected_approval_set_version: int | None,
        transition: StateTransition | None,
    ) -> tuple[list[Approval], ApprovalSet]:
        if not decisions:
            raise OperationConflictError("at least one approval decision is required")
        approval_set = self.get_approval_set(owner_id, approval_set_id)
        if approval_set is None:
            raise KeyError(approval_set_id)
        if (
            expected_approval_set_version is not None
            and approval_set.version != expected_approval_set_version
        ):
            raise OperationConflictError(
                "approval set version conflict: "
                f"expected {expected_approval_set_version}, actual {approval_set.version}"
            )
        members = {
            approval.approval_id: approval
            for approval in self.list_approval_set(owner_id, approval_set_id)
        }
        if not members or any(item not in members for item in decisions):
            raise OperationConflictError("approval does not belong to approval set")
        operation = self.operations.get(approval_set.operation_id)
        if operation is None or operation.owner_id != owner_id:
            raise OperationConflictError("approval operation not found or owner mismatch")
        if operation.phase is not OperationPhase.WAITING_APPROVAL:
            raise OperationConflictError("operation is not waiting for approval")
        if transition is not None and operation.version != transition.previous_version:
            raise OperationConflictError(
                f"operation version conflict: expected {transition.previous_version}, "
                f"actual {operation.version}"
            )
        changed = False
        for approval_id, decision in decisions.items():
            approval = members[approval_id]
            if approval.status is not ApprovalStatus.PENDING:
                if approval.status is ApprovalStatus.EXPIRED:
                    continue
                if approval.decision is not decision:
                    raise OperationConflictError("approval is no longer pending")
                if (
                    idempotency_key
                    and approval.idempotency_key
                    and approval.idempotency_key != idempotency_key
                ):
                    raise OperationConflictError("idempotency key conflicts with prior decision")
                continue
            if approval.expires_at is not None and approval.expires_at <= decided_at:
                updated = replace(
                    approval,
                    status=ApprovalStatus.EXPIRED,
                    version=approval.version + 1,
                    decided_by=decided_by,
                    decided_at=decided_at,
                    decision_source=decision_source,
                    idempotency_key=idempotency_key,
                    updated_at=decided_at,
                )
            else:
                updated = replace(
                    approval,
                    status=(
                        ApprovalStatus.APPROVED
                        if decision is ApprovalDecision.APPROVE
                        else ApprovalStatus.SKIPPED
                    ),
                    decision=decision,
                    version=approval.version + 1,
                    decided_by=decided_by,
                    decided_at=decided_at,
                    decision_source=decision_source,
                    idempotency_key=idempotency_key,
                    updated_at=decided_at,
                )
            self.approvals[approval_id] = updated
            changed = True
        updated_approvals = self.list_approval_set(owner_id, approval_set_id)
        next_set = replace(
            approval_set,
            status=_approval_set_status(tuple(updated_approvals)),
            **(
                {
                    "version": approval_set.version + 1,
                    "updated_at": decided_at,
                    "last_idempotency_key": idempotency_key,
                }
                if changed
                else {}
            ),
        )
        if changed:
            self.approval_sets[approval_set_id] = next_set
        if transition is not None:
            if not changed:
                raise OperationConflictError("approval decision is already applied")
            self.commit(transition)
        return updated_approvals, next_set

    def begin_tool_effect(self, request: ToolRequest, policy: RecoveryPolicy) -> None:
        key = (request.operation_id, request.call_id)
        if key in self.effects:
            raise OperationConflictError("tool effect already exists")
        self.effects[key] = {
            "name": request.name, "policy": policy.value, "status": "executing",
            "idempotency_key": request.idempotency_key,
        }

    def settle_tool_effect(self, request: ToolRequest, result: ToolResult) -> None:
        key = (request.operation_id, request.call_id)
        if key not in self.effects or self.effects[key].get("status") != "executing":
            raise OperationConflictError("tool effect intent does not exist")
        self.effects[key].update({
            "status": "failed" if result.is_error else "settled",
            "result": result.result, "error": result.error,
        })

    def list_tool_effects(self, owner_id, operation_id):
        operation = self.operations.get(operation_id)
        if operation is None or operation.owner_id != owner_id:
            return []
        return [
            {
                "operation_id": operation_id,
                "tool_call_id": call_id,
                "owner_id": owner_id,
                "tool_name": effect.get("name", ""),
                "recovery_policy": effect.get("policy", ""),
                "status": effect.get("status", ""),
                "idempotency_key": effect.get("idempotency_key", ""),
                "result": effect.get("result"),
                "error": effect.get("error", ""),
            }
            for (candidate_operation_id, call_id), effect in self.effects.items()
            if candidate_operation_id == operation_id
        ]


def _approval_set_status(approvals: tuple[Approval, ...]) -> ApprovalSetStatus:
    if not approvals:
        return ApprovalSetStatus.PENDING
    statuses = {item.status for item in approvals}
    if ApprovalStatus.PENDING in statuses:
        return (
            ApprovalSetStatus.PARTIALLY_DECIDED
            if len(statuses) > 1 else ApprovalSetStatus.PENDING
        )
    if ApprovalStatus.EXPIRED in statuses:
        return ApprovalSetStatus.EXPIRED
    if ApprovalStatus.CANCELLED in statuses:
        return ApprovalSetStatus.CANCELLED
    if all(item.status is ApprovalStatus.APPROVED for item in approvals):
        return ApprovalSetStatus.APPROVED
    return ApprovalSetStatus.SKIPPED


def _operation_projection_key(
    operation: OperationState,
    order: int,
) -> tuple[float, float, int, str, int]:
    return (
        float(operation.updated_at),
        float(operation.created_at),
        operation.version,
        order,
        operation.operation_id,
    )


def _operation_projection_metadata(operation: OperationState) -> dict:
    snapshot = operation.state.get("request_snapshot")
    if not isinstance(snapshot, dict):
        return {}
    metadata = snapshot.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}
