from __future__ import annotations

from matrix.runtime.core.runtime import AgentRuntime
from matrix.runtime.domain.events import RuntimeEventType
from matrix.runtime.domain.messages import Message
from matrix.runtime.domain.requests import ExecutionOptions, RunRequest
from matrix.runtime.domain.results import RunOutcome
from matrix.runtime.domain.tools import ToolSpec
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
