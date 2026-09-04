"""Public Runtime facade.

WP1 intentionally exposes the shape without starting a production execution
loop.  The model/tool loop is implemented in WP2 after these contracts are
validated by tests and adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
import uuid
from typing import Callable, Iterator

from ..domain.errors import OperationConflictError, RuntimeNotImplementedError
from ..domain.approvals import ApprovalDecision, ApprovalStatus
from ..domain.events import RuntimeEvent, RuntimeEventType
from ..domain.operations import OperationPhase, OperationState, StateTransition
from ..domain.requests import ExecutionOptions, ExecutionPolicy, ResumeInput, RunRequest
from ..domain.messages import Message
from ..domain.tools import RecoveryPolicy, ToolSpec
from ..domain.results import RunOutcome, RunResult, Suspension
from ..ports.model import ModelPort
from ..ports.context import ContextPort
from ..ports.store import OperationStorePort
from ..ports.tools import ToolExecutorPort
from .loop import execute_operation
from .debug import EphemeralDebugTrace
from .reducer import with_next_phase


@dataclass
class RunHandle:
    """Handle shape returned by start/resume in the completed Runtime."""

    operation_id: str
    _run: Callable[["RunHandle"], tuple[list[RuntimeEvent], RunResult]] | None = field(
        default=None, repr=False
    )
    _events: list[RuntimeEvent] = field(default_factory=list, repr=False)
    _result: RunResult | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _cancel_requested: bool = field(default=False, repr=False)
    _debug: EphemeralDebugTrace | None = field(default=None, repr=False)

    def events(self) -> Iterator[RuntimeEvent]:
        if not self._started:
            self._started = True
            if self._run is None:
                raise RuntimeNotImplementedError("Runtime execution is not configured")
            self._events, self._result = self._run(self)
        yield from self._events

    def result(self) -> RunResult:
        if not self._started:
            list(self.events())
        if self._result is None:
            raise RuntimeNotImplementedError("Runtime did not produce a result")
        return self._result

    def cancel(self, reason: str = "") -> None:
        del reason
        self._cancel_requested = True

    def debug_trace(self) -> list[dict[str, object]]:
        """Return the current run's ephemeral diagnostics, if enabled."""
        if self._debug is None:
            return []
        return [event.to_dict() for event in self._debug.snapshot()]

    def clear_debug_trace(self) -> None:
        """Release diagnostic payloads held by this handle."""
        if self._debug is not None:
            self._debug.clear()


class AgentRuntime:
    """Dependency-inverted single-Agent Runtime."""

    def __init__(
        self,
        store: OperationStorePort,
        model: ModelPort | None = None,
        tools: ToolExecutorPort | None = None,
        context: ContextPort | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.tools = tools
        self.context = context

    def start(self, request: RunRequest) -> RunHandle:
        if self.model is None or self.tools is None:
            raise RuntimeNotImplementedError("model and tools are required for Runtime execution")
        operation_id = uuid.uuid4().hex
        operation = OperationState(
            operation_id=operation_id,
            owner_id=request.owner_id,
            session_id=request.session_id,
            agent_id=request.agent_id,
            orchestration_run_id=request.orchestration_run_id,
            operation_scope=str(request.metadata.get("operation_scope", "top_level")),
            step_id=str(request.metadata.get("step_id", "")),
            state={
                "system_prompt": request.system_prompt,
                "model": request.model,
                "tools": [
                    {
                        "name": tool.name, "description": tool.description,
                        "input_schema": tool.input_schema,
                        "recovery_policy": tool.recovery_policy.value,
                        "requires_approval": tool.requires_approval,
                        "side_effect": tool.side_effect,
                    }
                    for tool in request.tools
                ],
                "execution_options": {
                    "max_turns": request.execution_options.max_turns,
                    "max_tool_calls": request.execution_options.max_tool_calls,
                    "timeout_seconds": request.execution_options.timeout_seconds,
                    "max_model_retries": request.execution_options.max_model_retries,
                },
                "execution_policy": {
                    "mode": request.execution_policy.mode,
                    "preset": request.execution_policy.preset,
                    "allow_external_effects": request.execution_policy.allow_external_effects,
                    "require_approval": request.execution_policy.require_approval,
                    "approval_mode": request.execution_policy.approval_mode,
                    "auto_approve_operations": list(request.execution_policy.auto_approve_operations),
                    "debug_trace": request.execution_policy.debug_trace,
                    "output_style": request.execution_policy.output_style,
                },
                "metadata": request.metadata,
                "runtime_messages": [
                    {
                        "role": message.role, "content": message.content,
                        "tool_call_id": message.tool_call_id,
                        "tool_calls": [
                            {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                            for call in message.tool_calls
                        ],
                    }
                    for message in request.messages
                ],
            },
        )
        self.store.create(operation)
        handle = RunHandle(
            operation_id=operation_id,
            _debug=EphemeralDebugTrace() if request.execution_policy.debug_trace else None,
        )
        handle._run = lambda current_handle: execute_operation(
            request=request,
            operation=operation,
            handle=current_handle,
            store=self.store,
            model=self.model,
            tools=self.tools,
            context=self.context,
            debug_trace=current_handle._debug,
        )
        return handle

    def resume(
        self,
        owner_id: str,
        operation_id: str,
        resume_input: ResumeInput,
    ) -> RunHandle:
        if self.model is None or self.tools is None:
            raise RuntimeNotImplementedError("model and tools are required for Runtime execution")
        operation = self.store.load(owner_id, operation_id)
        if operation is None:
            raise OperationConflictError("operation not found or owner mismatch")
        if operation.phase is not OperationPhase.WAITING_APPROVAL:
            raise OperationConflictError("operation is not waiting for approval")
        payload = resume_input.payload
        approval_set_id = str(payload.get("approval_set_id", ""))
        raw_approval_ids = payload.get("approval_ids", [])
        approval_ids = [
            str(item) for item in raw_approval_ids
            if str(item)
        ] if isinstance(raw_approval_ids, (list, tuple)) else []
        if not approval_set_id:
            raise OperationConflictError("approval_set_id is required")
        approvals = self.store.list_approval_set(owner_id, approval_set_id)
        if not approvals:
            raise OperationConflictError("approval set has no approval requests")
        approval_ids = approval_ids or [item.approval_id for item in approvals]
        decisions_payload = payload.get("decisions")
        decisions: dict[str, str] = {}
        if isinstance(decisions_payload, dict):
            decisions = {
                str(key): str(value)
                for key, value in decisions_payload.items()
                if str(key) and str(value) in {"approve", "skip"}
            }
        if not decisions:
            raise OperationConflictError("decisions are required")
        if any(item not in {approval.approval_id for approval in approvals} for item in decisions):
            raise OperationConflictError("approval does not belong to approval set")
        expected_version = payload.get("expected_operation_version")
        if expected_version is not None and int(expected_version) != operation.version:
            raise OperationConflictError(
                f"operation version conflict: expected {expected_version}, "
                f"actual {operation.version}"
            )
        approval_set = self.store.get_approval_set(owner_id, approval_set_id)
        if approval_set is None or approval_set.operation_id != operation_id:
            raise OperationConflictError("approval set does not belong to operation")
        policy_data = operation.state.get("execution_policy", {})
        policy = ExecutionPolicy(**{
            key: value for key, value in policy_data.items()
            if key in {
                "mode", "preset", "allow_external_effects", "require_approval",
                "approval_mode", "auto_approve_operations", "debug_trace", "output_style",
            }
        })
        handle = RunHandle(
            operation_id=operation_id,
            _debug=EphemeralDebugTrace() if policy.debug_trace else None,
        )
        handle._run = lambda current_handle: self._resume_operation(
            current_handle,
            owner_id=owner_id,
            operation_id=operation_id,
            approval_set_id=approval_set_id,
            approval_ids=approval_ids or [item.approval_id for item in approvals],
            decisions=decisions,
            policy=policy,
            expected_version=operation.version,
            expected_approval_set_version=(
                approval_set.version
                if payload.get("expected_approval_set_version") is None
                else int(payload["expected_approval_set_version"])
            ),
            decided_by=owner_id,
            decision_source=str(payload.get("decision_source", "chat")) or "chat",
            idempotency_key=str(payload.get("idempotency_key", "")),
        )
        return handle

    def _resume_operation(
        self,
        handle: RunHandle,
        *,
        owner_id: str,
        operation_id: str,
        approval_set_id: str,
        approval_ids: list[str],
        decisions: dict[str, str],
        policy: ExecutionPolicy,
        expected_version: int,
        expected_approval_set_version: int,
        decided_by: str,
        decision_source: str,
        idempotency_key: str,
    ) -> tuple[list[RuntimeEvent], RunResult]:
        operation = self.store.load(owner_id, operation_id)
        if operation is None:
            raise OperationConflictError("operation not found or owner mismatch")
        if operation.phase is not OperationPhase.WAITING_APPROVAL:
            raise OperationConflictError("operation is not waiting for approval")
        if operation.version != expected_version:
            raise OperationConflictError(
                f"operation version conflict: expected {expected_version}, "
                f"actual {operation.version}"
            )
        pending_items = _pending_tool_items(operation.state)
        pending_set_id = str(
            pending_items[0].get("approval_set_id", "") if pending_items else ""
        )
        if pending_set_id and pending_set_id != approval_set_id:
            raise OperationConflictError("approval set does not match suspended tool call")
        decided_at = time.time()
        approvals = self.store.list_approval_set(owner_id, approval_set_id)
        if not approvals:
            raise OperationConflictError("approval set has no approval requests")
        pending_approval_ids = {
            item.approval_id for item in approvals
            if item.status is ApprovalStatus.PENDING
        }
        if not pending_approval_ids.intersection(decisions):
            replayed = all(
                item.approval_id in decisions
                and item.decision is not None
                and item.decision.value == decisions[item.approval_id]
                and (
                    not idempotency_key
                    or item.idempotency_key == idempotency_key
                )
                for item in approvals
                if item.approval_id in decisions
            )
            if replayed:
                pending = [
                    _approval_payload(item)
                    for item in approvals
                    if item.status is ApprovalStatus.PENDING
                ]
                return [], RunResult(
                    outcome=RunOutcome.SUSPENDED,
                    operation_id=operation_id,
                    error="approval required",
                    suspension=Suspension(
                        reason="approval required",
                        approval_id=pending[0]["approval_id"] if pending else "",
                        approval_set_id=approval_set_id,
                        approval_ids=[item["approval_id"] for item in pending],
                        payload={
                            "actions": pending,
                            "approval_set_version": (
                                self.store.get_approval_set(
                                    owner_id, approval_set_id
                                ).version
                            ),
                        },
                    ),
                )
            raise OperationConflictError("no pending approval was selected")
        decision_values = {
            key: ApprovalDecision.APPROVE if value == "approve" else ApprovalDecision.SKIP
            for key, value in decisions.items()
        }
        decided_at = time.time()
        all_decided = pending_approval_ids.issubset(decision_values)
        event_type = (
            RuntimeEventType.RUN_RESUMED if all_decided
            else RuntimeEventType.APPROVAL_DECIDED
        )
        resume_event = RuntimeEvent(
            event_id=uuid.uuid4().hex,
            owner_id=operation.owner_id,
            operation_id=operation.operation_id,
            session_id=operation.session_id,
            sequence=operation.last_event_sequence + 1,
            event_type=event_type,
            timestamp=decided_at,
            payload={
                "approval_set_id": approval_set_id,
                "approval_ids": list(decisions),
                "decisions": decisions,
                "owner_id": owner_id,
                "expected_operation_version": expected_version,
                "expected_approval_set_version": expected_approval_set_version,
            },
        )
        target_phase = OperationPhase.RESUMING if all_decided else OperationPhase.WAITING_APPROVAL
        resumed_state = replace(
            operation,
            phase=target_phase,
            version=operation.version + 1,
            last_event_sequence=resume_event.sequence,
        )
        resolved, resolved_set = self.store.resolve_approval_set(
            owner_id=owner_id,
            approval_set_id=approval_set_id,
            decisions=decision_values,
            decided_at=decided_at,
            decided_by=decided_by,
            decision_source=decision_source,
            idempotency_key=idempotency_key,
            expected_approval_set_version=expected_approval_set_version,
            transition=StateTransition(
                previous_version=operation.version,
                new_state=resumed_state,
                events=(resume_event,),
            ),
        )
        if not all_decided:
            pending = [
                _approval_payload(item)
                for item in resolved
                if item.status is ApprovalStatus.PENDING
            ]
            return [resume_event], RunResult(
                outcome=RunOutcome.SUSPENDED,
                operation_id=operation_id,
                error="approval required",
                suspension=Suspension(
                    reason="approval required",
                    approval_id=pending[0]["approval_id"] if pending else "",
                    approval_set_id=approval_set_id,
                    approval_ids=[item["approval_id"] for item in pending],
                    payload={"actions": pending, "approval_set_version": resolved_set.version},
                ),
            )
        if any(item.status is ApprovalStatus.EXPIRED for item in resolved):
            latest = self.store.load(owner_id, operation_id) or resumed_state
            abort_event = RuntimeEvent(
                event_id=uuid.uuid4().hex,
                owner_id=operation.owner_id,
                operation_id=operation.operation_id,
                session_id=operation.session_id,
                sequence=latest.last_event_sequence + 1,
                event_type=RuntimeEventType.RUN_ABORTED,
                timestamp=time.time(),
                payload={"outcome": RunOutcome.ABORTED.value, "error": "approval expired"},
            )
            aborted_state = replace(
                with_next_phase(latest, OperationPhase.ABORTED),
                last_event_sequence=abort_event.sequence,
            )
            self.store.commit(StateTransition(
                previous_version=latest.version,
                new_state=aborted_state,
                events=(abort_event,),
            ))
            return [resume_event, abort_event], RunResult(
                outcome=RunOutcome.ABORTED,
                operation_id=operation.operation_id,
                error="approval expired",
            )

        state = resumed_state.state
        request = RunRequest(
            owner_id=resumed_state.owner_id,
            session_id=resumed_state.session_id,
            agent_id=resumed_state.agent_id,
            messages=[],
            system_prompt=state.get("system_prompt", ""),
            model=state.get("model", ""),
            tools=[ToolSpec(
                name=item["name"],
                description=item.get("description", ""),
                input_schema=item.get("input_schema", {}),
                recovery_policy=RecoveryPolicy(item.get("recovery_policy", "manual")),
                requires_approval=bool(item.get("requires_approval", False)),
                side_effect=bool(item.get("side_effect", False)),
            ) for item in state.get("tools", [])],
            execution_options=ExecutionOptions(**{
                key: value
                for key, value in state.get("execution_options", {}).items()
                if key in {"max_turns", "max_tool_calls", "timeout_seconds", "max_model_retries"}
            }),
            execution_policy=policy,
            metadata=dict(state.get("metadata", {})),
            orchestration_run_id=resumed_state.orchestration_run_id,
        )
        return execute_operation(
            request=request,
            operation=resumed_state,
            handle=handle,
            store=self.store,
            model=self.model,
            tools=self.tools,
            context=self.context,
            debug_trace=handle._debug,
            resume_approval_id=next(
                item.approval_id for item in resolved
                if item.status is not ApprovalStatus.EXPIRED
            ),
            resume_decision="approve",
            resume_decisions={
                item.approval_id: item.decision.value
                for item in resolved
                if item.decision is not None
            },
            committed_events=(resume_event,),
        )


def _pending_tool_items(state: dict) -> list[dict]:
    items = state.get("pending_tool_calls", [])
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _approval_payload(approval) -> dict:
    return {
        "approval_id": approval.approval_id,
        "approval_set_id": approval.approval_set_id,
        "tool_call_id": approval.tool_call_id,
        "tool_name": approval.tool_name,
        "name": approval.tool_name,
        "arguments": dict(approval.sanitized_arguments),
        "args": dict(approval.sanitized_arguments),
        "risk": approval.risk,
        "status": approval.status.value,
    }
