from __future__ import annotations

from matrix.runtime.adapters.model import MatrixModelAdapter
from matrix.runtime.adapters.tools import MatrixToolAdapter, tool_specs
from matrix.runtime.domain.messages import Message
from matrix.runtime.domain.tools import ToolRequest
from matrix.runtime.ports.model import ModelRequest
from matrix.runtime.testing.memory_store import MemoryOperationStore
from matrix.runtime.testing.fake_model import tool_call
from matrix.runtime.domain.tools import ToolSpec
from matrix.llm.protocol import FunctionCallResult, ToolCall
from matrix.tools import ToolDefinition, ToolRegistry


class StubLLM:
    def __init__(self):
        self.last_messages = []

    def function_call(self, system, messages, tools, tool_choice="auto", temperature=None):
        assert system == "system"
        self.last_messages = messages
        assert messages[0]["role"] == "user"
        if tools:
            assert tools[0]["name"] == "lookup"
        return FunctionCallResult(
            content="",
            tool_calls=[ToolCall(id="call-1", name="lookup", arguments={"q": "x"})],
            finish_reason="tool_calls",
        )

    def stream_complete(self, system, messages, temperature=None):
        yield "unused"


def test_matrix_model_adapter_translates_function_calls() -> None:
    client = StubLLM()
    adapter = MatrixModelAdapter(client)
    response = adapter.complete(ModelRequest(
        system_prompt="system",
        messages=[Message(role="user", content="question")],
        tools=[ToolSpec(name="lookup")],
    ))

    assert response.tool_calls[0].call_id == "call-1"
    assert response.tool_calls[0].arguments == {"q": "x"}


def test_matrix_model_adapter_preserves_multimodal_content() -> None:
    client = StubLLM()
    MatrixModelAdapter(client).complete(ModelRequest(
        system_prompt="system",
        messages=[Message(role="user", content=[
            {"type": "text", "text": "描述图片"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])],
        tools=[],
    ))

    assert client.last_messages[0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_matrix_tool_adapter_preserves_registry_errors_and_results() -> None:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="lookup",
        description="lookup",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        handler=lambda **kwargs: {"value": kwargs["q"]},
    ))
    adapter = MatrixToolAdapter(registry, session_id="session-1")

    result = adapter.execute(ToolRequest(
        operation_id="op-1", call_id="call-1", name="lookup", arguments={"q": "x"},
    ))
    error = adapter.execute(ToolRequest(
        operation_id="op-1", call_id="call-2", name="lookup", arguments={},
    ))

    assert result.result == {"value": "x"}
    assert not result.is_error
    assert error.is_error
    assert "缺少必需参数" in error.error
    assert tool_specs(registry)[0].name == "lookup"
