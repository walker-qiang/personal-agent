"""Adapters for external agents that own their own tool loop.

An external agent is intentionally not executed by ``AgentRuntime``.  The
adapter only gives its lifecycle a durable Runtime operation and maps the
agent's high-level events to Runtime events.  Codex therefore keeps ownership
of its CLI process and internal tool loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
import uuid
from typing import Any, Iterator, Protocol

from ..domain.events import RuntimeEvent, RuntimeEventType
from ..domain.operations import OperationPhase, OperationState, StateTransition
from ..domain.results import RunOutcome
from ..ports.store import OperationStorePort


class ExternalAgentClient(Protocol):
    def stream_agent(
        self, system: str, messages: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        ...


@dataclass(frozen=True)
class ExternalAgentEvent:
    runtime_event: RuntimeEvent
    ui_event: dict[str, Any]


@dataclass(frozen=True)
class ExternalAgentResult:
    outcome: RunOutcome
    operation_id: str
    final_message: str = ""
    error: str = ""


class ExternalAgentHandle:
    def __init__(self, adapter: "ExternalAgentAdapter", operation: OperationState,
                 system: str, messages: list[dict[str, Any]]) -> None:
        self._adapter = adapter
        self._operation = operation
        self._system = system
        self._messages = messages
        self._started = False
        self._cancel_requested = False
        self._events: list[ExternalAgentEvent] = []
        self._result: ExternalAgentResult | None = None

    @property
    def operation_id(self) -> str:
        return self._operation.operation_id

    def cancel(self, reason: str = "") -> None:
        self._cancel_requested = True
        self._cancel_reason = reason or "operation cancelled"

    def events(self) -> Iterator[ExternalAgentEvent]:
        if self._started:
            yield from self._events
            return
        self._started = True
        yield from self._run()

    def result(self) -> ExternalAgentResult:
        if not self._started:
            list(self.events())
        assert self._result is not None
        return self._result

    def _run(self) -> Iterator[ExternalAgentEvent]:
        current = self._operation
        try:
            item = self._adapter._commit(
                current,
                phase=OperationPhase.REQUESTING_MODEL,
                event_type=RuntimeEventType.RUN_START,
                payload={"agent_id": current.agent_id, "external_agent": True},
                ui_event={"type": "classify", "intent": "codex-direct"},
            )
            current, emitted = item
            self._events.append(emitted)
            yield emitted

            if self._cancel_requested:
                current, event, result = self._adapter._finish(
                    current, RunOutcome.ABORTED,
                    self._cancel_reason,
                    RuntimeEventType.RUN_ABORTED,
                    {"outcome": RunOutcome.ABORTED.value,
                     "error": self._cancel_reason},
                    {"type": "error", "message": self._cancel_reason},
                )
                wrapped = ExternalAgentEvent(
                    event, {"type": "error", "message": self._cancel_reason}
                )
                self._events.append(wrapped)
                self._result = result
                yield wrapped
                return

            stream_agent = getattr(self._adapter.client, "stream_agent", None)
            if callable(stream_agent):
                source = stream_agent(self._system, self._messages)
            else:
                source = (
                    {"type": "message", "content": token}
                    for token in self._adapter.client.stream_complete(
                        self._system, self._messages
                    )
                )

            answer_parts: list[str] = []
            for raw_event in source:
                if self._cancel_requested:
                    emitted = self._adapter._finish(
                        current, RunOutcome.ABORTED,
                        self._cancel_reason,
                        RuntimeEventType.RUN_ABORTED,
                        {"outcome": RunOutcome.ABORTED.value,
                         "error": self._cancel_reason},
                        {"type": "error", "message": self._cancel_reason},
                    )
                    current, event, result = emitted
                    wrapped = ExternalAgentEvent(event, {"type": "error", "message": self._cancel_reason})
                    self._events.append(wrapped)
                    self._result = result
                    yield wrapped
                    return

                raw = dict(raw_event or {})
                event_type = str(raw.get("type") or "progress")
                if event_type == "done":
                    continue
                if event_type == "message":
                    content = str(raw.get("content") or "")
                    if content:
                        answer_parts.append(content)
                    ui_event = raw
                    runtime_type = RuntimeEventType.MESSAGE_DELTA
                elif event_type == "error":
                    message = str(raw.get("message") or "external agent failed")
                    current, event, result = self._adapter._finish(
                        current, RunOutcome.FAILED, message,
                        RuntimeEventType.RUN_FAILED,
                        {"outcome": RunOutcome.FAILED.value, "error": message,
                         "external_event": raw},
                        raw,
                    )
                    wrapped = ExternalAgentEvent(event, raw)
                    self._events.append(wrapped)
                    self._result = result
                    yield wrapped
                    return
                else:
                    ui_event = raw
                    runtime_type = RuntimeEventType.TOOL_UPDATE

                current, event = self._adapter._commit(
                    current,
                    phase=OperationPhase.EXECUTING_TOOLS
                    if event_type in {"tool_call", "tool_result"}
                    else OperationPhase.REQUESTING_MODEL,
                    event_type=runtime_type,
                    payload={"external_event": raw},
                    ui_event=ui_event,
                )
                wrapped = event
                self._events.append(wrapped)
                yield wrapped

            answer = "".join(answer_parts).strip()
            current, event, result = self._adapter._finish(
                current, RunOutcome.COMPLETED, answer,
                RuntimeEventType.RUN_END,
                {"outcome": RunOutcome.COMPLETED.value, "message": answer},
                {"type": "done"},
            )
            wrapped = ExternalAgentEvent(event, {"type": "done"})
            self._events.append(wrapped)
            self._result = result
            yield wrapped
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            current, event, result = self._adapter._finish(
                current, RunOutcome.FAILED, message,
                RuntimeEventType.RUN_FAILED,
                {"outcome": RunOutcome.FAILED.value, "error": message},
                {"type": "error", "message": message},
            )
            wrapped = ExternalAgentEvent(event, {"type": "error", "message": message})
            self._events.append(wrapped)
            self._result = result
            yield wrapped


class ExternalAgentAdapter:
    """Durably wrap an external agent without taking over its inner loop."""

    def __init__(self, store: OperationStorePort, client: ExternalAgentClient) -> None:
        self.store = store
        self.client = client

    def start(
        self,
        *,
        owner_id: str,
        session_id: str,
        agent_id: str,
        system: str,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> ExternalAgentHandle:
        operation = OperationState(
            operation_id=uuid.uuid4().hex,
            owner_id=owner_id,
            session_id=session_id,
            agent_id=agent_id,
            operation_scope="top_level",
            state={
                "external_agent": True,
                "system_prompt": system,
                "messages": messages,
                "metadata": metadata or {},
            },
        )
        self.store.create(operation)
        return ExternalAgentHandle(self, operation, system, messages)

    def _commit(
        self,
        operation: OperationState,
        *,
        phase: OperationPhase,
        event_type: RuntimeEventType,
        payload: dict[str, Any],
        ui_event: dict[str, Any],
    ) -> tuple[OperationState, ExternalAgentEvent]:
        event = RuntimeEvent(
            event_id=uuid.uuid4().hex,
            owner_id=operation.owner_id,
            operation_id=operation.operation_id,
            session_id=operation.session_id,
            sequence=operation.last_event_sequence + 1,
            event_type=event_type,
            timestamp=time.time(),
            payload=payload,
        )
        state = dict(operation.state)
        state["last_external_event"] = payload.get("external_event", payload)
        next_operation = replace(
            operation,
            phase=phase,
            version=operation.version + 1,
            last_event_sequence=event.sequence,
            state=state,
        )
        self.store.commit(StateTransition(
            previous_version=operation.version,
            new_state=next_operation,
            events=(event,),
        ))
        return next_operation, ExternalAgentEvent(event, ui_event)

    def _finish(
        self,
        operation: OperationState,
        outcome: RunOutcome,
        message: str,
        event_type: RuntimeEventType,
        payload: dict[str, Any],
        ui_event: dict[str, Any],
    ) -> tuple[OperationState, RuntimeEvent, ExternalAgentResult]:
        phase = {
            RunOutcome.COMPLETED: OperationPhase.COMPLETED,
            RunOutcome.FAILED: OperationPhase.FAILED,
            RunOutcome.ABORTED: OperationPhase.ABORTED,
        }[outcome]
        event = RuntimeEvent(
            event_id=uuid.uuid4().hex,
            owner_id=operation.owner_id,
            operation_id=operation.operation_id,
            session_id=operation.session_id,
            sequence=operation.last_event_sequence + 1,
            event_type=event_type,
            timestamp=time.time(),
            payload=payload,
        )
        state = dict(operation.state)
        state["final_message"] = message if outcome is RunOutcome.COMPLETED else ""
        next_operation = replace(
            operation,
            phase=phase,
            version=operation.version + 1,
            last_event_sequence=event.sequence,
            state=state,
        )
        self.store.commit(StateTransition(
            previous_version=operation.version,
            new_state=next_operation,
            events=(event,),
        ))
        return next_operation, event, ExternalAgentResult(
            outcome=outcome,
            operation_id=operation.operation_id,
            final_message=message if outcome is RunOutcome.COMPLETED else "",
            error=message if outcome is not RunOutcome.COMPLETED else "",
        )
