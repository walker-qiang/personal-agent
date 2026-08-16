from __future__ import annotations

from dataclasses import replace
import time

from matrix.runtime.core.runtime import AgentRuntime
from matrix.runtime.domain.approvals import ApprovalStatus
from matrix.runtime.domain.events import RuntimeEventType
from matrix.runtime.domain.messages import Message
from matrix.runtime.domain.operations import OperationPhase
from matrix.runtime.domain.requests import ExecutionOptions, ResumeInput, RunRequest
from matrix.runtime.domain.results import RunOutcome
from matrix.runtime.domain.tools import RecoveryPolicy, ToolSpec
from matrix.runtime.ports.model import ModelResponse
from matrix.runtime.testing.fake_model import FakeModel, tool_call
from matrix.runtime.testing.fake_tools import FakeToolExecutor
from matrix.runtime.testing.memory_store import MemoryOperationStore


def _request(**kwargs) -> RunRequest:
    return RunRequest(
        owner_id="user-a",
        session_id="session-1",
        agent_id="assistant",
        messages=[Message(role="user", content="hello")],
        tools=[ToolSpec(name="lookup")],
        **kwargs,
    )


def test_runtime_completes_without_tools_and_emits_ordered_events() -> None:
    runtime = AgentRuntime(
        store := MemoryOperationStore(),
        model=FakeModel([ModelResponse(content="你好")]),
        tools=FakeToolExecutor(),
    )
    handle = runtime.start(_request())

    events = list(handle.events())
    result = handle.result()

    assert result.outcome is RunOutcome.COMPLETED
    assert result.final_message == "你好"
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].event_type is RuntimeEventType.RUN_START
    assert events[-1].event_type is RuntimeEventType.RUN_END
    assert store.load("user-a", handle.operation_id).is_terminal


def test_runtime_executes_tool_then_returns_follow_up_answer() -> None:
    model = FakeModel([
        ModelResponse(tool_calls=(tool_call("call-1", "lookup", {"q": "x"}),), finish_reason="tool_calls"),
        ModelResponse(content="结果是 x"),
    ])
    tools = FakeToolExecutor({"lookup": lambda args: {"value": args["q"]}})
    runtime = AgentRuntime(MemoryOperationStore(), model=model, tools=tools)

    events = list(runtime.start(_request()).events())

    assert events[-1].event_type is RuntimeEventType.RUN_END
    assert any(event.event_type is RuntimeEventType.TOOL_START for event in events)
    assert any(event.event_type is RuntimeEventType.TOOL_END for event in events)
    assert len(tools.requests) == 1


def test_runtime_enforces_max_tool_calls() -> None:
    model = FakeModel([
        ModelResponse(
            tool_calls=(tool_call("call-1", "lookup"), tool_call("call-2", "lookup")),
            finish_reason="tool_calls",
        ),
    ])
    tools = FakeToolExecutor({"lookup": lambda args: {"ok": True}})
    runtime = AgentRuntime(MemoryOperationStore(), model=model, tools=tools)

    result = runtime.start(_request(
        execution_options=ExecutionOptions(max_tool_calls=1),
    )).result()

    assert result.outcome is RunOutcome.FAILED
    assert "maximum tool calls" in result.error
    assert tools.requests == []


def test_runtime_can_be_cancelled_before_execution() -> None:
    runtime = AgentRuntime(
        MemoryOperationStore(),
        model=FakeModel([ModelResponse(content="never")]),
        tools=FakeToolExecutor(),
    )
    handle = runtime.start(_request())
    handle.cancel("user stopped")

    result = handle.result()

    assert result.outcome is RunOutcome.ABORTED
    assert result.error == "operation cancelled"


def test_runtime_retries_transient_model_failure() -> None:
    class FlakyModel(FakeModel):
        def __init__(self) -> None:
            super().__init__([ModelResponse(content="ok")])
            self.failures = 1

        def complete(self, request):
            if self.failures:
                self.failures -= 1
                raise TimeoutError("temporary")
            return super().complete(request)

    model = FlakyModel()
    runtime = AgentRuntime(
        MemoryOperationStore(), model=model, tools=FakeToolExecutor()
    )
    result = runtime.start(_request(
        execution_options=ExecutionOptions(max_model_retries=1),
    )).result()

    assert result.outcome is RunOutcome.COMPLETED
    assert result.final_message == "ok"


def test_runtime_settles_unexpected_tool_exception_as_tool_error() -> None:
    class RaisingTools:
        def execute(self, request):
            raise RuntimeError("downstream unavailable")

    model = FakeModel([
        ModelResponse(
            tool_calls=(tool_call("call-error", "lookup"),),
            finish_reason="tool_calls",
        ),
        ModelResponse(content="已识别工具异常"),
    ])
    store = MemoryOperationStore()
    result = AgentRuntime(store, model=model, tools=RaisingTools()).start(
        _request()
    ).result()

    assert result.outcome is RunOutcome.COMPLETED
    assert result.final_message == "已识别工具异常"
    assert result.tool_results[0].is_error is True
    assert "downstream unavailable" in result.tool_results[0].error
    assert store.effects[(result.operation_id, "call-error")]["status"] == "failed"


def _suspend_for_approval(store: MemoryOperationStore):
    runtime = AgentRuntime(
        store,
        model=FakeModel([
            ModelResponse(
                tool_calls=(tool_call("call-approval", "lookup", {"q": "x"}),),
                finish_reason="tool_calls",
            ),
        ]),
        tools=FakeToolExecutor({"lookup": lambda args: {"value": args["q"]}}),
    )
    handle = runtime.start(RunRequest(
        owner_id="user-a",
        session_id="approval-session",
        agent_id="assistant",
        messages=[Message(role="user", content="run lookup")],
        tools=[ToolSpec(
            name="lookup",
            requires_approval=True,
            recovery_policy=RecoveryPolicy.MANUAL,
        )],
    ))
    result = handle.result()
    operation = store.load("user-a", handle.operation_id)
    return handle.operation_id, operation.state["pending_tool_call"]["approval_id"], result


def test_approved_resume_uses_effect_journal() -> None:
    store = MemoryOperationStore()
    operation_id, approval_id, suspended = _suspend_for_approval(store)
    tools = FakeToolExecutor({"lookup": lambda args: {"value": args["q"]}})
    runtime = AgentRuntime(
        store,
        model=FakeModel([ModelResponse(content="approved")]),
        tools=tools,
    )

    handle = runtime.resume(
        "user-a",
        operation_id,
        ResumeInput(
            kind="approval",
            decision="approve",
            payload={"approval_id": approval_id},
        ),
    )
    events = list(handle.events())
    result = handle.result()

    assert suspended.outcome is RunOutcome.SUSPENDED
    assert result.outcome is RunOutcome.COMPLETED
    assert len(tools.requests) == 1
    assert store.effects[(operation_id, "call-approval")]["status"] == "settled"
    assert RuntimeEventType.RUN_RESUMED in [event.event_type for event in events]
    assert RuntimeEventType.TOOL_START in [event.event_type for event in events]


def test_skipped_resume_does_not_create_effect() -> None:
    store = MemoryOperationStore()
    operation_id, approval_id, _ = _suspend_for_approval(store)
    tools = FakeToolExecutor({"lookup": lambda args: {"value": args["q"]}})
    runtime = AgentRuntime(
        store,
        model=FakeModel([ModelResponse(content="skipped")]),
        tools=tools,
    )

    result = runtime.resume(
        "user-a",
        operation_id,
        ResumeInput(
            kind="approval",
            decision="skip",
            payload={"approval_id": approval_id},
        ),
    ).result()

    assert result.outcome is RunOutcome.COMPLETED
    assert tools.requests == []
    assert store.effects == {}


def test_expired_approval_aborts_without_executing_tool() -> None:
    store = MemoryOperationStore()
    operation_id, approval_id, _ = _suspend_for_approval(store)
    store.approvals[approval_id] = replace(
        store.approvals[approval_id], expires_at=time.time() - 1,
    )
    tools = FakeToolExecutor({"lookup": lambda args: {"value": args["q"]}})
    runtime = AgentRuntime(
        store,
        model=FakeModel([ModelResponse(content="must not run")]),
        tools=tools,
    )

    result = runtime.resume(
        "user-a",
        operation_id,
        ResumeInput(
            kind="approval",
            decision="approve",
            payload={"approval_id": approval_id},
        ),
    ).result()

    assert result.outcome is RunOutcome.ABORTED
    assert result.error == "approval expired"
    assert tools.requests == []
    assert store.approvals[approval_id].status is ApprovalStatus.EXPIRED
    assert store.load("user-a", operation_id).phase is OperationPhase.ABORTED
