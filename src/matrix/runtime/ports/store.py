"""Operation persistence port."""

from __future__ import annotations

from typing import Protocol

from ..domain.operations import OperationState, StateTransition
from ..domain.approvals import Approval, ApprovalDecision
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

    def has_active(self, owner_id: str, session_id: str) -> bool:
        ...

    def find_active(self, owner_id: str, session_id: str) -> OperationState | None:
        ...

    def create_approval(self, approval: Approval) -> None:
        ...

    def get_approval(self, owner_id: str, approval_id: str) -> Approval | None:
        ...

    def decide_approval(
        self, owner_id: str, approval_id: str, decision: ApprovalDecision,
    ) -> Approval:
        ...

    def begin_tool_effect(self, request: ToolRequest, policy: RecoveryPolicy) -> None:
        ...

    def settle_tool_effect(self, request: ToolRequest, result: ToolResult) -> None:
        ...
