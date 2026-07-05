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

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((system, messages))
        if not self.responses:
            raise AssertionError("no fake LLM responses left")
        return self.responses.pop(0)


@pytest.fixture
def chat_service(tmp_cache_path: Path) -> ChatService:
    config = AgentConfig(
        root_path=tmp_cache_path.parent,
        cache_path=tmp_cache_path,
        trace_path=tmp_cache_path.parent / "trace.jsonl",
        store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
        host="127.0.0.1",
        port=0,
        deepseek_api_key="test-key",
    )
    registry = ToolRegistry()
    register_all(registry, tmp_cache_path)
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
        chat_service.llm = FakeLLM([
            '{"intent": "react", "skill_name": ""}',
            "Thought: 查询持仓\nAction: finance.holdings_summary\nAction Input: {}",
            "Thought: 已有数据\nFinal Answer: 当前持仓健康。",
        ])
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        types = [e["type"] for e in events]
        assert "done" in types
        assert "tool_call" in types
        assert "tool_result" in types

    def test_returns_tool_calls(self, chat_service):
        chat_service.llm = FakeLLM([
            '{"intent": "react", "skill_name": ""}',
            "Thought: 查询持仓\nAction: finance.holdings_summary\nAction Input: {}",
            "Thought: 已有数据\nFinal Answer: 当前持仓健康。",
        ])
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) >= 1
        assert tool_calls[0]["name"] == "finance.holdings_summary"

    def test_returns_token_events(self, chat_service):
        chat_service.llm = FakeLLM([
            '{"intent": "react", "skill_name": ""}',
            "Thought: 查询持仓\nAction: finance.holdings_summary\nAction Input: {}",
            "Thought: 已有数据\nFinal Answer: 当前持仓健康。",
        ])
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) >= 1

    def test_session_memory_persists(self, chat_service):
        chat_service.llm = FakeLLM([
            '{"intent": "react", "skill_name": ""}',
            "Thought: 查询\nAction: finance.holdings_summary\nAction Input: {}",
            "Thought: done\nFinal Answer: 持仓健康。",
            '{"intent": "react", "skill_name": ""}',
            "Thought: 回忆\nAction: finance.holdings_summary\nAction Input: {}",
            "Thought: done\nFinal Answer: 仍然健康。",
        ])
        sid = "mem-test"
        list(chat_service.stream_chat("当前持仓怎么样？", sid))
        events = list(chat_service.stream_chat("还有变化吗？", sid))
        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) >= 1

    def test_reset_clears_session(self, chat_service):
        chat_service.llm = FakeLLM([
            '{"intent": "react", "skill_name": ""}',
            "Thought: q\nAction: finance.holdings_summary\nAction Input: {}",
            "Thought: done\nFinal Answer: ok.",
        ])
        sid = "reset-test"
        list(chat_service.stream_chat("test", sid))
        chat_service.reset(sid)
        assert len(chat_service._get_history(sid)) == 0

    def test_skill_flow_in_graph(self, chat_service):
        chat_service.llm = FakeLLM([
            '{"intent": "skill", "skill_name": "test-skill"}',
        ])
        from matrix.skills import SkillDefinition
        chat_service.skills = [
            SkillDefinition(
                name="test-skill",
                title="测试技能",
                trigger_keywords=["测试"],
                workflow=[
                    {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
                ],
            ),
        ]
        events = list(chat_service.stream_chat("跑测试技能"))
        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) >= 1