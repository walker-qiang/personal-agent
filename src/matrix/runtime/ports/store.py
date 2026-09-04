"""Operation persistence port."""

from __future__ import annotations

from typing import Protocol

from ..domain.operations import OperationState, StateTransition
from ..domain.approvals import Approval, ApprovalDecision, ApprovalSet
from ..domain.events import RuntimeEvent
from ..domain.tools import ToolRequest, ToolResult, RecoveryPolicy


class OperationStorePort(Protocol):
    def create(self, operation: OperationState) -> None:
        ...

    def load(self, owner_id: str, operation_id: str) -> OperationState | None:
        ...

    def commit(self, transition: StateTransition) -> None:
        ...

    def list_incomplete(self) -> list[OperationState]:
        ...

    def recover_incomplete(self, reason: str = "process_restart") -> list[OperationState]:
        """Fail closed operations left mid-flight by a process restart.

        WAITING_APPROVAL is intentionally preserved because it has a durable,
        user-mediated resume path. Other non-terminal phases cannot be safely
        replayed without knowing whether an external effect already ran.
        """
        ...

    def list_operations(
        self,
        owner_id: str,
        session_id: str | None = None,
        phase: str | None = None,
        limit: int = 50,
    ) -> list[OperationState]:
        ...

    def list_approvals(
        self,
        owner_id: str,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Approval]:
        ...

    def list_events(
        self,
        owner_id: str,
        operation_id: str,
        limit: int = 200,
    ) -> list[RuntimeEvent]:
        ...

    def has_active(self, owner_id: str, session_id: str) -> bool:
        ...

    def find_active(self, owner_id: str, session_id: str) -> OperationState | None:
        ...

    def find_waiting_approval(self, owner_id: str, session_id: str) -> OperationState | None:
        ...

    def list_waiting_approvals(
        self, owner_id: str, session_id: str, limit: int = 50,
    ) -> list[OperationState]:
        ...

    def ensure_orchestration_run(
        self,
        run_id: str,
        owner_id: str,
        session_id: str,
        graph_thread_id: str = "",
        metadata: dict | None = None,
    ) -> None:
        ...

    def upsert_orchestration_step(
        self,
        run_id: str,
        step_id: str,
        operation_id: str,
        status: str,
        metadata: dict | None = None,
    ) -> None:
        ...

    def create_approval(self, approval: Approval) -> None:
        ...

    def create_approval_set(self, approval_set: ApprovalSet) -> None:
        ...

    def get_approval(self, owner_id: str, approval_id: str) -> Approval | None:
        ...

    def get_approval_set(self, owner_id: str, approval_set_id: str) -> ApprovalSet | None:
        ...

    def list_approval_set(
        self, owner_id: str, approval_set_id: str,
    ) -> list[Approval]:
        ...

    def delete_session_runtime(self, owner_id: str, session_id: str) -> None:
        """Delete all Runtime/workflow data owned by one chat session."""
        ...

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
        """Resolve one or more approvals and optionally commit an operation transition atomically."""
        ...

    def begin_tool_effect(self, request: ToolRequest, policy: RecoveryPolicy) -> None:
        ...

    def settle_tool_effect(self, request: ToolRequest, result: ToolResult) -> None:
        ...
