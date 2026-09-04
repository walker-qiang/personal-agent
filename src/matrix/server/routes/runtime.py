"""Read-only Runtime status and approval history endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from ...runtime.domain.approvals import Approval
from ...runtime.domain.operations import OperationPhase, OperationState

router = APIRouter(prefix="/api/runtime", tags=["runtime"])


def _store(request: Request):
    return request.app.state.runtime_store


@router.get("/operations")
async def list_operations(
    request: Request,
    session_id: str | None = Query(None),
    phase: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """List Runtime operation snapshots owned by the authenticated user."""
    operations = _store(request).list_operations(
        _user_id(request), session_id=session_id, phase=phase, limit=limit,
    )
    return {"operations": [_operation_dict(operation) for operation in operations]}


@router.get("/approvals")
async def list_approvals(
    request: Request,
    session_id: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """List approval history owned by the authenticated user."""
    store = _store(request)
    approvals = store.list_approvals(
        _user_id(request), session_id=session_id, status=status, limit=limit,
    )
    return {
        "approvals": [
            _approval_dict(
                approval,
                store.load(_user_id(request), approval.operation_id),
                store.get_approval_set(_user_id(request), approval.approval_set_id)
                if approval.approval_set_id else None,
            )
            for approval in approvals
        ]
    }


@router.get("/approval-sets/{approval_set_id}")
async def approval_set_detail(request: Request, approval_set_id: str):
    """Return one approval set with its member decisions and CAS version."""
    store = _store(request)
    owner_id = _user_id(request)
    approval_set = store.get_approval_set(owner_id, approval_set_id)
    if approval_set is None:
        raise HTTPException(status_code=404, detail="runtime approval set not found")
    operation = store.load(owner_id, approval_set.operation_id)
    return {
        "approval_set": {
            "approval_set_id": approval_set.approval_set_id,
            "operation_id": approval_set.operation_id,
            "orchestration_run_id": approval_set.orchestration_run_id,
            "status": approval_set.status.value,
            "version": approval_set.version,
            "created_at": approval_set.created_at,
            "updated_at": approval_set.updated_at,
        },
        "approvals": [
            _approval_dict(item, operation, approval_set)
            for item in store.list_approval_set(owner_id, approval_set_id)
        ],
    }


@router.get("/operations/{operation_id}/events")
async def operation_events(
    request: Request,
    operation_id: str,
    limit: int = Query(200, ge=1, le=1000),
):
    """Return ordered durable events for one user-owned operation."""
    store = _store(request)
    operation = store.load(_user_id(request), operation_id)
    if operation is None:
        raise HTTPException(status_code=404, detail="runtime operation not found")
    return {
        "operation_id": operation_id,
        "events": [
            {key: value for key, value in event.to_dict().items() if key != "owner_id"}
            for event in store.list_events(_user_id(request), operation_id, limit=limit)
        ],
    }


@router.get("/operations/{operation_id}/retry-context")
async def retry_context(request: Request, operation_id: str):
    """Return a safe user prompt for an explicit retry of a recovered run.

    Retrying never resumes or replays the old operation. The client submits
    this context as a new chat request after the user explicitly confirms.
    """
    store = _store(request)
    operation = store.load(_user_id(request), operation_id)
    if operation is None:
        raise HTTPException(status_code=404, detail="runtime operation not found")
    if operation.phase is not OperationPhase.RECOVERY_REQUIRED:
        raise HTTPException(
            status_code=409,
            detail="only recovery_required operations can be retried",
        )
    if operation.operation_scope != "top_level":
        raise HTTPException(
            status_code=409,
            detail="DAG step operations must be retried by their parent workflow",
        )
    session_entries = store.list_session_entries(
        _user_id(request), operation.session_id, entry_type="user", limit=1,
    )
    user_message = ""
    if session_entries:
        payload = session_entries[0].get("payload", {})
        if isinstance(payload, dict):
            user_message = str(payload.get("content", "")).strip()
    messages = operation.state.get("runtime_messages", [])
    user_message = user_message or next(
        (
            str(message.get("content", "")).strip()
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and str(message.get("content", "")).strip()
        ),
        "",
    )
    if not user_message:
        raise HTTPException(status_code=409, detail="operation has no retryable user message")
    policy = operation.state.get("execution_policy", {})
    return {
        "operation_id": operation.operation_id,
        "session_id": operation.session_id,
        "message": user_message,
        "mode": str(policy.get("mode", "read_only")),
        "preset": str(policy.get("preset", "default")),
    }


def _user_id(request: Request) -> str:
    return getattr(request.state, "user_id", "default")


def _operation_dict(operation: OperationState) -> dict:
    return {
        "operation_id": operation.operation_id,
        "session_id": operation.session_id,
        "agent_id": operation.agent_id,
        "orchestration_run_id": operation.orchestration_run_id,
        "operation_scope": operation.operation_scope,
        "step_id": operation.step_id,
        "phase": operation.phase.value,
        "turn_index": operation.turn_index,
        "version": operation.version,
        "last_event_sequence": operation.last_event_sequence,
        "created_at": operation.created_at,
        "updated_at": operation.updated_at,
        "terminal": operation.is_terminal,
    }


def _approval_dict(
    approval: Approval,
    operation: OperationState | None,
    approval_set=None,
) -> dict:
    return {
        "approval_id": approval.approval_id,
        "approval_set_id": approval.approval_set_id,
        "approval_set_status": approval_set.status.value if approval_set else None,
        "approval_set_version": approval_set.version if approval_set else None,
        "operation_id": approval.operation_id,
        "session_id": operation.session_id if operation else "",
        "tool_call_id": approval.tool_call_id,
        "tool_name": approval.tool_name,
        "sanitized_arguments": approval.sanitized_arguments,
        "risk": approval.risk,
        "status": approval.status.value,
        "decision": approval.decision.value if approval.decision else None,
        "expires_at": approval.expires_at,
        "version": approval.version,
        "created_at": approval.created_at,
        "updated_at": approval.updated_at,
        "decided_by": approval.decided_by,
        "decided_at": approval.decided_at,
        "decision_source": approval.decision_source,
        "idempotency_key": approval.idempotency_key,
    }
