"""Public Runtime facade.

WP1 intentionally exposes the shape without starting a production execution
loop.  The model/tool loop is implemented in WP2 after these contracts are
validated by tests and adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import uuid
from typing import Callable, Iterator

from ..domain.errors import OperationConflictError, RuntimeNotImplementedError
from ..domain.events import RuntimeEvent
from ..domain.operations import OperationPhase, OperationState, StateTransition
from ..domain.requests import ExecutionOptions, ResumeInput, RunRequest
from ..domain.messages import Message
from ..domain.tools import RecoveryPolicy, ToolSpec
from ..domain.results import RunOutcome, RunResult
from ..domain.tools import ToolRequest, ToolResult
from ..ports.model import ModelPort
from ..ports.store import OperationStorePort
from ..ports.tools import ToolExecutorPort
from .loop import execute_operation


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


class AgentRuntime:
    """Dependency-inverted single-Agent Runtime."""

    def __init__(
        self,
        store: OperationStorePort,
        model: ModelPort | None = None,
        tools: ToolExecutorPort | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.tools = tools

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
                    }
                    for tool in request.tools
                ],
                "execution_options": {
                    "max_turns": request.execution_options.max_turns,
                    "max_tool_calls": request.execution_options.max_tool_calls,
                    "timeout_seconds": request.execution_options.timeout_seconds,
                    "max_model_retries": request.execution_options.max_model_retries,
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
        handle = RunHandle(operation_id=operation_id)
        handle._run = lambda current_handle: execute_operation(
            request=request,
            operation=operation,
            handle=current_handle,
            store=self.store,
            model=self.model,
            tools=self.tools,
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
        approval_id = str(resume_input.payload.get("approval_id", ""))
        if not approval_id:
            approval_id = str(operation.state.get("pending_tool_call", {}).get("approval_id", ""))
        decision = resume_input.decision or str(resume_input.payload.get("decision", "approve"))
        approval = self.store.get_approval(owner_id, approval_id)
        if approval is None:
            raise OperationConflictError("approval not found or owner mismatch")
        from ..domain.approvals import ApprovalDecision
        updated = self.store.decide_approval(
            owner_id, approval_id,
            ApprovalDecision.APPROVE if decision == "approve" else ApprovalDecision.SKIP,
        )
        handle = RunHandle(operation_id=operation_id)
        state = operation.state
        request = RunRequest(
            owner_id=operation.owner_id, session_id=operation.session_id,
            agent_id=operation.agent_id,
            messages=[], system_prompt=state.get("system_prompt", ""),
            model=state.get("model", ""),
            tools=[ToolSpec(
                name=item["name"], description=item.get("description", ""),
                input_schema=item.get("input_schema", {}),
                recovery_policy=RecoveryPolicy(item.get("recovery_policy", "manual")),
                requires_approval=bool(item.get("requires_approval", False)),
            ) for item in state.get("tools", [])],
            execution_options=ExecutionOptions(**{
                key: value for key, value in state.get("execution_options", {}).items()
                if key in {"max_turns", "max_tool_calls", "timeout_seconds", "max_model_retries"}
            }), metadata=dict(state.get("metadata", {})),
            orchestration_run_id=operation.orchestration_run_id,
        )
        handle._run = lambda current_handle: execute_operation(
            request=request,
            operation=operation,
            handle=current_handle,
            store=self.store,
            model=self.model,
            tools=self.tools,
            resume_approval_id=updated.approval_id,
            resume_decision=updated.decision.value if updated.decision else decision,
        )
        return handle
