"""Synchronous single-Agent model/tool loop.

This module deliberately owns only the execution mechanics.  Planning,
domain-agent lookup, reflection and LangGraph routing stay above it.
"""

from __future__ import annotations

from dataclasses import replace
import time
import uuid
from typing import Any

from ..domain.events import RuntimeEvent, RuntimeEventType
from ..domain.approvals import Approval, ApprovalDecision, ApprovalStatus
from ..domain.messages import Message, ToolCall
from ..domain.operations import OperationPhase, OperationState, StateTransition
from ..domain.requests import RunRequest
from ..domain.results import RunOutcome, RunResult, Suspension
from ..domain.tools import RecoveryPolicy, ToolRequest, ToolResult
from ..ports.model import ModelPort, ModelRequest, ModelResponse
from ..ports.context import ContextPort
from ..ports.store import OperationStorePort
from ..ports.tools import ToolExecutorPort
from .reducer import with_next_phase
from .debug import EphemeralDebugTrace


def execute_operation(
    *,
    request: RunRequest,
    operation: OperationState,
    handle: Any,
    store: OperationStorePort,
    model: ModelPort,
    tools: ToolExecutorPort,
    context: ContextPort | None = None,
    resume_approval_id: str = "",
    resume_decision: str = "",
    committed_events: tuple[RuntimeEvent, ...] = (),
    debug_trace: EphemeralDebugTrace | None = None,
) -> tuple[list[RuntimeEvent], RunResult]:
    """Execute one operation and return committed events plus its result."""

    current = operation
    events: list[RuntimeEvent] = list(committed_events)
    messages = _messages_from_state(operation.state) or list(request.messages)
    tool_results: list[ToolResult] = []
    total_tool_calls = 0
    usage: dict[str, Any] = {}
    started_at = time.monotonic()

    if resume_approval_id:
        if current.phase is not OperationPhase.RESUMING:
            raise ValueError("resumed operation must already be in resuming phase")
        pending = current.state.get("pending_tool_call", {})
        if pending.get("approval_id") != resume_approval_id:
            raise ValueError("approval does not match suspended tool call")
        pending_call = ToolCall(
            call_id=pending["call_id"], name=pending["name"],
            arguments=dict(pending.get("arguments", {})),
        )
        if resume_decision == "approve":
            current = _commit_phase(store, current, OperationPhase.EXECUTING_TOOLS, None, events)
            current = _commit_state(
                store,
                current,
                _event(current, RuntimeEventType.TOOL_START, {
                    "call_id": pending_call.call_id,
                    "name": pending_call.name,
                }),
                events,
            )
            tool_request = ToolRequest(
                operation_id=current.operation_id, call_id=pending_call.call_id,
                name=pending_call.name, arguments=pending_call.arguments,
            )
            tool_spec = next(
                (spec for spec in request.tools if spec.name == pending_call.name), None
            )
            result = _execute_tool_effect(
                store,
                tools,
                tool_request,
                tool_spec.recovery_policy if tool_spec else RecoveryPolicy.MANUAL,
            )
            _check_timeout(started_at, request.execution_options.timeout_seconds)
            _debug(debug_trace, "tool_result", {
                "name": pending_call.name, "call_id": pending_call.call_id,
                "result": result.result, "error": result.error,
            })
        else:
            result = ToolResult(
                call_id=pending_call.call_id, name=pending_call.name,
                error="tool call skipped by user", is_error=True,
            )
        tool_results.append(result)
        messages.append(Message(role="tool", content=_tool_content(result), tool_call_id=result.call_id))
        current = _commit_state(
            store, replace(current, state={**current.state, "runtime_messages": _messages_to_state(messages), "pending_tool_call": {}}),
            _event(current, RuntimeEventType.TOOL_END, {
                "call_id": result.call_id, "name": result.name,
                "is_error": result.is_error, "error": result.error,
            }), events,
        )
        current = _commit_phase(store, current, OperationPhase.PREPARING_NEXT_TURN, None, events)
    else:
        current = _commit_phase(
            store, current, OperationPhase.PREPARING,
            _event(current, RuntimeEventType.RUN_START, {"agent_id": request.agent_id}),
            events,
        )

    try:
        for turn_index in range(request.execution_options.max_turns):
            _check_timeout(started_at, request.execution_options.timeout_seconds)
            if handle._cancel_requested:
                return _finish(
                    store, current, events, RunOutcome.ABORTED,
                    "operation cancelled", tool_results,
                    RuntimeEventType.RUN_ABORTED,
                )

            current = _commit_state(
                store, replace(current, turn_index=turn_index),
                _event(current, RuntimeEventType.TURN_START, {"turn": turn_index}),
                events,
            )
            current = _commit_phase(store, current, OperationPhase.REQUESTING_MODEL, None, events)

            response, current = _complete_with_retry(
                request=request,
                messages=messages,
                model=model,
                handle=handle,
                store=store,
                operation=current,
                events=events,
                debug_trace=debug_trace,
                started_at=started_at,
                timeout_seconds=request.execution_options.timeout_seconds,
                context=context,
            )
            _check_timeout(started_at, request.execution_options.timeout_seconds)
            if response is None:
                return _finish(
                    store, current, events, RunOutcome.ABORTED,
                    "operation cancelled", tool_results,
                    RuntimeEventType.RUN_ABORTED,
                )

            usage = response.usage
            assistant = Message(
                role="assistant",
                content=response.content,
                tool_calls=tuple(response.tool_calls),
            )
            messages.append(assistant)
            current = _commit_snapshot(
                store, current,
                {"runtime_messages": _messages_to_state(messages)},
            )

            if not response.tool_calls:
                current = _commit_state(
                    store, current,
                    _event(current, RuntimeEventType.MESSAGE_START, {"turn": turn_index}),
                    events,
                )
                if response.content:
                    current = _commit_state(
                        store, current,
                        _event(current, RuntimeEventType.MESSAGE_DELTA, {"content": response.content}),
                        events,
                    )
                current = _commit_state(
                    store, current,
                    _event(current, RuntimeEventType.MESSAGE_END, {"finish_reason": response.finish_reason}),
                    events,
                )
                current = _commit_state(
                    store, current,
                    _event(current, RuntimeEventType.TURN_END, {"turn": turn_index}),
                    events,
                )
                return _finish(
                    store, current, events, RunOutcome.COMPLETED,
                    response.content, tool_results, RuntimeEventType.RUN_END, usage,
                )

            if total_tool_calls + len(response.tool_calls) > request.execution_options.max_tool_calls:
                return _finish(
                    store, current, events, RunOutcome.FAILED,
                    "maximum tool calls exceeded", tool_results,
                    RuntimeEventType.RUN_FAILED, usage,
                )

            current = _commit_phase(store, current, OperationPhase.EXECUTING_TOOLS, None, events)
            for tool_call in response.tool_calls:
                if handle._cancel_requested:
                    return _finish(
                        store, current, events, RunOutcome.ABORTED,
                        "operation cancelled", tool_results,
                        RuntimeEventType.RUN_ABORTED, usage,
                    )
                total_tool_calls += 1
                tool_spec = next(
                    (spec for spec in request.tools if spec.name == tool_call.name), None
                )
                requires_approval = bool(
                    tool_spec is not None and (
                        tool_spec.requires_approval
                        or (tool_spec.side_effect and request.execution_policy.require_approval)
                    )
                )
                if tool_spec is not None and tool_spec.side_effect and not request.execution_policy.allow_external_effects:
                    result = ToolResult(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        error=(
                            f"tool {tool_call.name} is blocked by agent mode "
                            f"{request.execution_policy.mode}; use an approved writeback operation"
                        ),
                        is_error=True,
                    )
                    _debug(debug_trace, "tool_blocked", {
                        "name": tool_call.name, "call_id": tool_call.call_id,
                        "mode": request.execution_policy.mode,
                    })
                    current = _commit_state(
                        store, current,
                        _event(current, RuntimeEventType.TOOL_START, {
                            "call_id": tool_call.call_id, "name": tool_call.name,
                        }), events,
                    )
                    current = _commit_state(
                        store, current,
                        _event(current, RuntimeEventType.TOOL_END, {
                            "call_id": result.call_id, "name": result.name,
                            "is_error": True, "error": result.error,
                        }), events,
                    )
                    tool_results.append(result)
                    messages.append(Message(
                        role="tool", content=_tool_content(result), tool_call_id=result.call_id,
                    ))
                    continue
                auto_approved = _can_auto_approve(
                    tool_call.name, tool_call.arguments, request.execution_policy,
                )
                if requires_approval and not auto_approved:
                    approval = Approval(
                        approval_id=uuid.uuid4().hex,
                        owner_id=current.owner_id,
                        operation_id=current.operation_id,
                        tool_call_id=tool_call.call_id,
                        tool_name=tool_call.name,
                        sanitized_arguments=dict(tool_call.arguments),
                        risk="tool_requires_approval",
                    )
                    store.create_approval(approval)
                    _debug(debug_trace, "approval_required", {
                        "approval_id": approval.approval_id,
                        "name": tool_call.name,
                        "mode": request.execution_policy.mode,
                    })
                    current = replace(
                        current,
                        state={
                            **current.state,
                            "runtime_messages": _messages_to_state(messages),
                            "pending_tool_call": {
                                "approval_id": approval.approval_id,
                                "call_id": tool_call.call_id,
                                "name": tool_call.name,
                                "arguments": dict(tool_call.arguments),
                            },
                        },
                    )
                    current = _commit_phase(
                        store, current, OperationPhase.WAITING_APPROVAL,
                        _event(current, RuntimeEventType.APPROVAL_REQUIRED, {
                            "approval_id": approval.approval_id,
                            "call_id": tool_call.call_id,
                            "name": tool_call.name,
                            "arguments": dict(tool_call.arguments),
                        }), events,
                    )
                    current = _commit_state(
                        store, current,
                        _event(current, RuntimeEventType.RUN_SUSPENDED, {
                            "approval_id": approval.approval_id,
                        }), events,
                    )
                    return events, RunResult(
                        outcome=RunOutcome.SUSPENDED,
                        operation_id=current.operation_id,
                        tool_results=list(tool_results),
                        error="approval required",
                        suspension=Suspension(
                            reason="approval required", approval_id=approval.approval_id,
                            payload={"tool_name": tool_call.name, "arguments": dict(tool_call.arguments)},
                        ),
                    )
                if auto_approved:
                    approval = Approval(
                        approval_id=uuid.uuid4().hex,
                        owner_id=current.owner_id,
                        operation_id=current.operation_id,
                        tool_call_id=tool_call.call_id,
                        tool_name=tool_call.name,
                        sanitized_arguments=dict(tool_call.arguments),
                        risk="auto_allowlist",
                        status=ApprovalStatus.APPROVED,
                        decision=ApprovalDecision.APPROVE,
                    )
                    store.create_approval(approval)
                    _debug(debug_trace, "approval_auto_approved", {
                        "approval_id": approval.approval_id,
                        "name": tool_call.name,
                        "operation": _approval_operation(tool_call.name, tool_call.arguments),
                    })
                    current = _commit_state(
                        store, current,
                        _event(current, RuntimeEventType.APPROVAL_DECIDED, {
                            "approval_id": approval.approval_id,
                            "decision": "approve",
                            "source": "policy",
                            "operation": _approval_operation(tool_call.name, tool_call.arguments),
                        }), events,
                    )
                current = _commit_state(
                    store, current,
                    _event(current, RuntimeEventType.TOOL_START, {
                        "call_id": tool_call.call_id,
                        "name": tool_call.name,
                    }),
                    events,
                )
                tool_request = ToolRequest(
                    operation_id=current.operation_id,
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                _debug(debug_trace, "tool_request", {
                    "name": tool_request.name,
                    "call_id": tool_request.call_id,
                    "arguments": tool_request.arguments,
                })
                result = _execute_tool_effect(
                    store,
                    tools,
                    tool_request,
                    tool_spec.recovery_policy if tool_spec else RecoveryPolicy.MANUAL,
                )
                _check_timeout(started_at, request.execution_options.timeout_seconds)
                _debug(debug_trace, "tool_result", {
                    "name": tool_call.name, "call_id": tool_call.call_id,
                    "result": result.result, "error": result.error,
                })
                tool_results.append(result)
                messages.append(Message(
                    role="tool",
                    content=_tool_content(result),
                    tool_call_id=tool_call.call_id,
                ))
                current = _commit_state(
                    store, current,
                    _event(current, RuntimeEventType.TOOL_END, {
                        "call_id": result.call_id,
                        "name": result.name,
                        "is_error": result.is_error,
                        "error": result.error,
                    }),
                    events,
                )

            current = _commit_phase(store, current, OperationPhase.PREPARING_NEXT_TURN, None, events)

        return _finish(
            store, current, events, RunOutcome.FAILED,
            "maximum turns exceeded", tool_results,
            RuntimeEventType.RUN_FAILED, usage,
        )
    except TimeoutError as exc:
        latest = store.load(operation.owner_id, operation.operation_id)
        if latest is not None:
            current = latest
        return _finish(
            store, current, events, RunOutcome.FAILED,
            str(exc), tool_results, RuntimeEventType.RUN_FAILED, usage,
        )
    except Exception as exc:
        # Model retries commit RETRY_SCHEDULED transitions inside
        # _complete_with_retry().  If the final attempt fails, that helper
        # raises before returning its newer snapshot, so `current` can lag
        # behind the durable SQLite version.  Reload before the terminal
        # failure transition to avoid committing from a stale version.
        latest = store.load(operation.owner_id, operation.operation_id)
        if latest is not None:
            current = latest
        return _finish(
            store, current, events, RunOutcome.FAILED,
            f"runtime execution failed: {type(exc).__name__}: {exc}",
            tool_results, RuntimeEventType.RUN_FAILED, usage,
        )


def _complete_with_retry(
    *,
    request: RunRequest,
    messages: list[Message],
    model: ModelPort,
    handle: Any,
    store: OperationStorePort,
    operation: OperationState,
    events: list[RuntimeEvent],
    debug_trace: EphemeralDebugTrace | None = None,
    started_at: float | None = None,
    timeout_seconds: float = 300.0,
    context: ContextPort | None = None,
) -> tuple[ModelResponse | None, OperationState]:
    attempts = request.execution_options.max_model_retries + 1
    for attempt in range(attempts):
        if started_at is not None:
            _check_timeout(started_at, timeout_seconds)
        if handle._cancel_requested:
            return None, operation
        try:
            model_messages = (
                context.fit(request.system_prompt, messages)
                if context is not None
                else list(messages)
            )
            model_request = ModelRequest(
                system_prompt=request.system_prompt,
                messages=model_messages,
                tools=request.tools,
                model=request.model,
                metadata={
                    **request.metadata,
                    "agent_mode": request.execution_policy.mode,
                    "agent_preset": request.execution_policy.preset,
                    "output_style": request.execution_policy.output_style,
                },
            )
            _debug(debug_trace, "model_request", {
                "model": model_request.model,
                "system_prompt": model_request.system_prompt,
                "messages": [_message_debug_value(message) for message in model_request.messages],
                "tools": [tool.name for tool in model_request.tools],
                "metadata": model_request.metadata,
                "attempt": attempt,
            })
            response = model.complete(model_request)
            if started_at is not None:
                _check_timeout(started_at, timeout_seconds)
            _debug(debug_trace, "model_response", {
                "content": response.content,
                "tool_calls": [
                    {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                    for call in response.tool_calls
                ],
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            })
            return response, operation
        except Exception as exc:
            _debug(debug_trace, "model_error", {
                "attempt": attempt, "error": f"{type(exc).__name__}: {exc}",
            })
            if attempt + 1 >= attempts:
                raise
            operation = _commit_state(
                store, operation,
                _event(operation, RuntimeEventType.RETRY_SCHEDULED, {
                    "attempt": attempt + 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }),
                events,
            )


def _execute_tool_effect(
    store: OperationStorePort,
    tools: ToolExecutorPort,
    request: ToolRequest,
    recovery_policy: RecoveryPolicy,
) -> ToolResult:
    """Execute one external tool call through the persistent effect journal."""

    store.begin_tool_effect(request, recovery_policy)
    try:
        result = tools.execute(request)
    except Exception as exc:
        # Tool adapters should normally normalize failures, but Runtime is
        # the last durable boundary. Never leave an effect in ``executing``
        # merely because an adapter violated that contract.
        result = ToolResult(
            call_id=request.call_id,
            name=request.name,
            error=f"tool execution failed: {type(exc).__name__}: {exc}",
            is_error=True,
        )
    store.settle_tool_effect(request, result)
    return result


def _approval_operation(tool_name: str, arguments: dict[str, Any]) -> str:
    """Resolve the business operation represented by an approval tool call."""
    if tool_name == "writeback.execute_plan":
        plan = arguments.get("plan", {})
        if isinstance(plan, dict):
            return str(plan.get("operation", ""))
    return tool_name


def _check_timeout(started_at: float, timeout_seconds: float) -> None:
    """Enforce the operation wall-clock budget between blocking calls."""
    elapsed = time.monotonic() - started_at
    if elapsed >= timeout_seconds:
        raise TimeoutError(
            f"operation timed out after {timeout_seconds:g} seconds"
        )


def _can_auto_approve(tool_name: str, arguments: dict[str, Any], policy: Any) -> bool:
    if policy.approval_mode != "auto_allowlist" or not policy.allow_external_effects:
        return False
    operation = _approval_operation(tool_name, arguments)
    return bool(operation and operation in set(policy.auto_approve_operations))


def _debug(trace: EphemeralDebugTrace | None, kind: str, payload: dict[str, Any]) -> None:
    if trace is not None:
        trace.emit(kind, payload)


def _message_debug_value(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "tool_calls": [
            {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
    }


def _finish(
    store: OperationStorePort,
    operation: OperationState,
    events: list[RuntimeEvent],
    outcome: RunOutcome,
    message: str,
    tool_results: list[ToolResult],
    event_type: RuntimeEventType,
    usage: dict[str, Any] | None = None,
) -> tuple[list[RuntimeEvent], RunResult]:
    phase = {
        RunOutcome.COMPLETED: OperationPhase.COMPLETED,
        RunOutcome.FAILED: OperationPhase.FAILED,
        RunOutcome.ABORTED: OperationPhase.ABORTED,
    }[outcome]
    current = _commit_phase(
        store, operation, phase,
        _event(operation, event_type, {"outcome": outcome.value, "error": message if outcome != RunOutcome.COMPLETED else ""}),
        events,
    )
    del current
    return events, RunResult(
        outcome=outcome,
        operation_id=operation.operation_id,
        final_message=message if outcome == RunOutcome.COMPLETED else "",
        tool_results=list(tool_results),
        usage=usage or {},
        error=message if outcome != RunOutcome.COMPLETED else "",
    )


def _event(operation: OperationState, event_type: RuntimeEventType, payload: dict[str, Any]) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=uuid.uuid4().hex,
        owner_id=operation.owner_id,
        operation_id=operation.operation_id,
        session_id=operation.session_id,
        sequence=operation.last_event_sequence + 1,
        event_type=event_type,
        timestamp=time.time(),
        payload=payload,
    )


def _messages_to_state(messages: list[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.role,
            "content": message.content,
            "tool_call_id": message.tool_call_id,
            "tool_calls": [
                {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
                for call in message.tool_calls
            ],
        }
        for message in messages
    ]


def _messages_from_state(state: dict[str, Any]) -> list[Message]:
    values = state.get("runtime_messages", [])
    result: list[Message] = []
    for value in values if isinstance(values, list) else []:
        result.append(Message(
            role=value.get("role", "user"), content=value.get("content", ""),
            tool_call_id=value.get("tool_call_id", ""),
            tool_calls=tuple(
                ToolCall(
                    call_id=item.get("call_id", ""), name=item.get("name", ""),
                    arguments=dict(item.get("arguments", {})),
                )
                for item in value.get("tool_calls", [])
            ),
        ))
    return result


def _commit_phase(
    store: OperationStorePort,
    operation: OperationState,
    phase: OperationPhase,
    event: RuntimeEvent | None,
    events: list[RuntimeEvent],
) -> OperationState:
    from .reducer import validate_transition

    validate_transition(operation.phase, phase)
    next_state = replace(
        operation,
        phase=phase,
        version=operation.version + 1,
        last_event_sequence=event.sequence if event else operation.last_event_sequence,
    )
    store.commit(StateTransition(
        previous_version=operation.version,
        new_state=next_state,
        events=(event,) if event else (),
    ))
    if event:
        events.append(event)
    return next_state


def _commit_state(
    store: OperationStorePort,
    operation: OperationState,
    event: RuntimeEvent,
    events: list[RuntimeEvent],
) -> OperationState:
    next_state = replace(
        operation,
        version=operation.version + 1,
        last_event_sequence=event.sequence,
    )
    store.commit(StateTransition(
        previous_version=operation.version,
        new_state=next_state,
        events=(event,),
    ))
    events.append(event)
    return next_state


def _commit_snapshot(
    store: OperationStorePort,
    operation: OperationState,
    state: dict[str, Any],
) -> OperationState:
    next_state = replace(operation, state=dict(state), version=operation.version + 1)
    store.commit(StateTransition(previous_version=operation.version, new_state=next_state))
    return next_state


def _tool_content(result: ToolResult) -> str:
    if result.is_error:
        return result.error or "tool execution failed"
    return str(result.result)
