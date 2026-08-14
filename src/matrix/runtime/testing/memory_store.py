"""In-memory operation store used by contract tests and future loop tests."""

from __future__ import annotations

from dataclasses import replace
from ..domain.approvals import Approval, ApprovalDecision, ApprovalStatus
from ..domain.tools import RecoveryPolicy, ToolRequest, ToolResult

from ..domain.errors import OperationConflictError
from ..domain.operations import OperationPhase, OperationState, StateTransition


class MemoryOperationStore:
    def __init__(self) -> None:
        self.operations: dict[str, OperationState] = {}
        self.events: dict[str, list[object]] = {}
        self.approvals: dict[str, Approval] = {}
        self.effects: dict[tuple[str, str], dict] = {}

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

    def list_incomplete(self) -> list[OperationState]:
        return [operation for operation in self.operations.values() if not operation.is_terminal]

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

    def event_list(self, operation_id: str) -> list[object]:
        return list(self.events.get(operation_id, []))

    def create_approval(self, approval: Approval) -> None:
        if approval.approval_id in self.approvals:
            raise OperationConflictError("approval already exists")
        self.approvals[approval.approval_id] = approval

    def get_approval(self, owner_id: str, approval_id: str) -> Approval | None:
        approval = self.approvals.get(approval_id)
        if approval is None or approval.owner_id != owner_id:
            return None
        return approval

    def resolve_approval(
        self,
        owner_id: str,
        approval_id: str,
        decision: ApprovalDecision,
        decided_at: float,
        transition: StateTransition,
    ) -> Approval:
        approval = self.get_approval(owner_id, approval_id)
        if approval is None:
            raise KeyError(approval_id)
        operation = self.operations.get(approval.operation_id)
        if operation is None or operation.owner_id != owner_id:
            raise OperationConflictError("approval operation not found or owner mismatch")
        if transition.new_state.operation_id != operation.operation_id:
            raise OperationConflictError("approval operation does not match transition")
        if operation.phase is not OperationPhase.WAITING_APPROVAL:
            raise OperationConflictError("operation is not waiting for approval")
        if operation.version != transition.previous_version:
            raise OperationConflictError(
                f"operation version conflict: expected {transition.previous_version}, "
                f"actual {operation.version}"
            )
        if approval.status is ApprovalStatus.EXPIRED:
            return approval
        if approval.status is not ApprovalStatus.PENDING:
            raise OperationConflictError("approval is no longer pending")
        if approval.expires_at is not None and approval.expires_at <= decided_at:
            expired = replace(
                approval,
                status=ApprovalStatus.EXPIRED,
                version=approval.version + 1,
            )
            self.approvals[approval_id] = expired
            return expired
        self.commit(transition)
        updated = replace(
            approval,
            status=(ApprovalStatus.APPROVED if decision == ApprovalDecision.APPROVE else ApprovalStatus.SKIPPED),
            decision=decision,
            version=approval.version + 1,
        )
        self.approvals[approval_id] = updated
        return updated

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
        if key not in self.effects:
            raise OperationConflictError("tool effect intent does not exist")
        self.effects[key].update({
            "status": "failed" if result.is_error else "settled",
            "result": result.result, "error": result.error,
        })
