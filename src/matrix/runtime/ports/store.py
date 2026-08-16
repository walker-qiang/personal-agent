"""Operation persistence port."""

from __future__ import annotations

from typing import Protocol

from ..domain.operations import OperationState, StateTransition
from ..domain.approvals import Approval, ApprovalDecision
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

    def create_approval(self, approval: Approval) -> None:
        ...

    def get_approval(self, owner_id: str, approval_id: str) -> Approval | None:
        ...

    def resolve_approval(
        self,
        owner_id: str,
        approval_id: str,
        decision: ApprovalDecision,
        decided_at: float,
        transition: StateTransition,
    ) -> Approval:
        """Consume a pending approval and commit the resume transition atomically."""
        ...

    def begin_tool_effect(self, request: ToolRequest, policy: RecoveryPolicy) -> None:
        ...

    def settle_tool_effect(self, request: ToolRequest, result: ToolResult) -> None:
        ...
