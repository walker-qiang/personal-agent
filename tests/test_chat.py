"""Tests for ChatService orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from matrix.chat import ChatService, preview_json
from matrix.config import AgentConfig
from matrix.tools import ToolRegistry
from matrix.tools.finance import register_all


class FakeLLM:
    """Fake LLM client that returns predefined responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[tuple[str, list[dict]]] = []

    def complete(self, system: str, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append(("complete", messages))
        if not self.responses:
            raise AssertionError("no fake LLM responses left")
        return self.responses.pop(0)

    def stream_complete(self, system: str, messages: list[dict[str, str]], **kwargs):
        """Fake streaming: yield the next response character by character."""
        self.calls.append(("stream", messages))
        text = self.responses.pop(0) if self.responses else ""
        for ch in text:
            yield ch

    def function_call(self, system, messages, tools, tool_choice="auto", **kwargs):
        """Fake function calling: returns a FunctionCallResult with no tool calls."""
        from matrix.llm import FunctionCallResult
        self.calls.append(("function_call", messages))
        text = self.responses.pop(0) if self.responses else ""
        return FunctionCallResult(content=text, tool_calls=[])


@pytest.fixture
def chat_service(tmp_cache_path: Path) -> ChatService:
    config = AgentConfig(
        root_path=tmp_cache_path.parent,
        cache_path=tmp_cache_path,
        trace_path=tmp_cache_path.parent / "trace.jsonl",
        store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
        checkpoint_path=str(tmp_cache_path.parent / "var" / "agent" / "checkpoints.db"),
        skills_dir=tmp_cache_path.parent / "skills" / "investment",
        skills_base_dir=tmp_cache_path.parent / "skills",
        host="127.0.0.1",
        port=0,
        deepseek_api_key="test-key",
    )
    registry = ToolRegistry()
    register_all(registry, tmp_cache_path)
    # Also register web/agnes tools for Commander agent
    from matrix.tools.web import register_all as register_web
    from matrix.tools.agnes import register_all as register_agnes
    register_web(registry)
    register_agnes(registry)
    return ChatService(config, registry)


class TestPreviewJson:
    def test_returns_full_when_short(self):
        result = preview_json({"a": 1}, limit=100)
        assert '"a"' in result
        assert "1" in result

    def test_truncates_when_long(self):
        result = preview_json({"data": "x" * 2000}, limit=10)
        assert result.endswith("...(truncated)")


class TestChatService:
    """Tests for stream_chat (LangGraph-based orchestration)."""

    def test_empty_message_returns_error(self, chat_service):
        events = list(chat_service.stream_chat(""))
        types = [e["type"] for e in events]
        assert "error" in types
        assert "done" in types

    def test_no_llm_returns_error(self, tmp_cache_path):
        from matrix.tools import ToolRegistry
        from matrix.tools.finance import register_all
        config = AgentConfig(
            root_path=tmp_cache_path.parent,
            cache_path=tmp_cache_path,
            trace_path=tmp_cache_path.parent / "trace.jsonl",
            store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
            checkpoint_path=str(tmp_cache_path.parent / "var" / "agent" / "checkpoints.db"),
            skills_dir=tmp_cache_path.parent / "skills" / "investment",
            skills_base_dir=tmp_cache_path.parent / "skills",
            host="127.0.0.1",
            port=0,
        )
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        service = ChatService(config, registry)
        events = list(service.stream_chat("test"))
        types = [e["type"] for e in events]
        assert "error" in types

    def test_returns_done_event(self, chat_service):
        # Native function calling: returns content directly (no tool calls)
        chat_service._default_llm = FakeLLM([
            "当前持仓健康。",
        ])
        chat_service._pipeline_llm = FakeLLM(["[]"])
        chat_service.skills = []
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        types = [e["type"] for e in events]
        assert "done" in types
        assert "token" in types

    def test_returns_token_events(self, chat_service):
        chat_service._default_llm = FakeLLM([
            "当前持仓健康。",
        ])
        chat_service._pipeline_llm = FakeLLM(["[]"])
        chat_service.skills = []
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) >= 1, f"events={[(e['type'], e.get('content','')[:60]) for e in events]}"

    def test_session_memory_persists(self, chat_service):
        chat_service._default_llm = FakeLLM([
            "持仓健康。",
            "仍然健康。",
        ])
        chat_service._pipeline_llm = FakeLLM(["[]", "[]"])
        chat_service.skills = []
        sid = "mem-test"
        list(chat_service.stream_chat("当前持仓怎么样？", sid))
        events = list(chat_service.stream_chat("还有变化吗？", sid))
        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) >= 1, f"events={[(e['type'], e.get('content','')[:60]) for e in events]}"

    def test_reset_clears_session(self, chat_service):
        chat_service._default_llm = FakeLLM([
            "ok.",
        ])
        chat_service._pipeline_llm = FakeLLM(["[]"])
        chat_service.skills = []
        sid = "reset-test"
        list(chat_service.stream_chat("test", sid))
        chat_service.reset(sid)
        assert len(chat_service._get_history(sid)) == 0

    def test_skill_flow_in_graph(self, chat_service):
        chat_service._default_llm = FakeLLM([
            "技能执行完成，共2个持仓。",
        ])
        chat_service._pipeline_llm = FakeLLM(["[]"])
        from matrix.skills import SkillDefinition
        chat_service.skills = [
            SkillDefinition(
                name="test-skill",
                title="测试技能",
                description="跑测试技能",
                workflow=[
                    {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
                ],
            ),
        ]
        events = list(chat_service.stream_chat("跑测试技能"))
        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) >= 1

    def test_needs_summary_streaming_path(self, chat_service):
        """Test streaming summarization when function_call returns tool_calls then answer."""
        from matrix.llm import FunctionCallResult, ToolCall

        class StreamingLLM(FakeLLM):
            def __init__(self, responses):
                super().__init__(responses)
                self._fc_count = 0

            def function_call(self, system, messages, tools, tool_choice="auto", **kwargs):
                self.calls.append(("function_call", messages))
                self._fc_count += 1
                if self._fc_count == 1:
                    # First call: return tool calls
                    return FunctionCallResult(
                        content="",
                        tool_calls=[
                            ToolCall(id="call_1", name="web_search", arguments={"query": "test"}),
                        ],
                        finish_reason="tool_calls",
                    )
                # Subsequent calls: return the answer
                text = self.responses.pop(0) if self.responses else ""
                return FunctionCallResult(content=text, tool_calls=[])

        chat_service._default_llm = StreamingLLM([
            "当前持仓健康，共2个持仓。",
        ])
        # Pipeline LLM used by commander_plan_node; returns single commander step
        chat_service._pipeline_llm = FakeLLM([
            '[{"agent_id": "commander", "task": "查询当前持仓", "step": 1}]',
        ])
        chat_service.skills = []
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        tokens = [e for e in events if e["type"] == "token"]
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) >= 1, "Should have tool_call events"
        assert len(tokens) >= 1, "Should have streaming token events"