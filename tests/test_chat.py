"""Tests for ChatService orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from matrix.chat import (
    ChatService,
    extract_json_object,
    parse_tool_calls,
    preview_json,
    result_count,
    timestamp,
)
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
        host="127.0.0.1",
        port=0,
        deepseek_api_key="test-key",
    )
    registry = ToolRegistry()
    register_all(registry, tmp_cache_path)
    return ChatService(config, registry)


class TestExtractJsonObject:
    def test_extracts_fenced_json(self):
        payload = extract_json_object(
            """```json\n{"tool_calls":[{"name":"test","arguments":{}}]}\n```"""
        )
        assert payload["tool_calls"][0]["name"] == "test"

    def test_extracts_bare_json(self):
        payload = extract_json_object('{"tool_calls":[]}')
        assert payload["tool_calls"] == []

    def test_extracts_json_with_prefix_text(self):
        payload = extract_json_object('Some text before {"tool_calls":[]} and after')
        assert payload["tool_calls"] == []

    def test_raises_on_non_object(self):
        with pytest.raises(ValueError, match="JSON object"):
            extract_json_object("[1, 2, 3]")


class TestParseToolCalls:
    def test_parses_valid_calls(self):
        allowed = {"finance.holdings_summary", "finance.bucket_allocation"}
        calls = parse_tool_calls(
            '{"tool_calls":[{"name":"finance.holdings_summary","arguments":{}}]}',
            allowed,
        )
        assert len(calls) == 1
        assert calls[0].name == "finance.holdings_summary"

    def test_rejects_unknown_tool(self):
        with pytest.raises(ValueError, match="unknown tool"):
            parse_tool_calls(
                '{"tool_calls":[{"name":"finance.unknown","arguments":{}}]}',
                {"finance.holdings_summary"},
            )

    def test_returns_empty_for_empty_calls(self):
        calls = parse_tool_calls('{"tool_calls":[]}', {"finance.holdings_summary"})
        assert calls == []

    def test_limits_to_four_calls(self):
        calls = parse_tool_calls(
            json.dumps({
                "tool_calls": [
                    {"name": "finance.holdings_summary", "arguments": {}},
                    {"name": "finance.holdings_summary", "arguments": {}},
                    {"name": "finance.holdings_summary", "arguments": {}},
                    {"name": "finance.holdings_summary", "arguments": {}},
                    {"name": "finance.holdings_summary", "arguments": {}},
                ]
            }),
            {"finance.holdings_summary"},
        )
        assert len(calls) == 4


class TestPreviewJson:
    def test_returns_full_when_short(self):
        result = preview_json({"a": 1}, limit=100)
        assert '"a"' in result
        assert '1' in result

    def test_truncates_when_long(self):
        result = preview_json({"data": "x" * 2000}, limit=10)
        assert result.endswith("...(truncated)")


class TestResultCount:
    def test_returns_count_field(self):
        assert result_count({"count": 5}) == 5

    def test_returns_holding_count(self):
        assert result_count({"holding_count": 3}) == 3

    def test_returns_bucket_count(self):
        assert result_count({"bucket_count": 2}) == 2

    def test_returns_list_length(self):
        assert result_count({"buckets": [1, 2, 3]}) == 3

    def test_returns_zero_when_no_counts(self):
        assert result_count({"other": "value"}) == 0


class TestTimestamp:
    def test_returns_iso_format(self):
        ts = timestamp()
        assert "T" in ts
        assert "Z" in ts


class TestChatService:
    def test_runs_planned_tool_and_returns_answer(self, chat_service):
        chat_service.llm = FakeLLM([
            '{"tool_calls":[{"name":"finance.bucket_allocation","arguments":{}}]}',
            '{"tool_calls":[]}',
            "\u73b0\u91d1 42.9%\uff0c\u6210\u957f 57.1%\u3002",  # 现金 42.9%，成长 57.1%。
        ])

        events = list(chat_service.stream_chat("\u5f53\u524d\u914d\u7f6e\u600e\u4e48\u6837\uff1f"))

        types = [e["type"] for e in events]
        assert types == ["tool_call", "tool_result", "token", "done"]
        assert events[0]["name"] == "finance.bucket_allocation"
        assert "cash" in events[1]["preview"]
        assert "\u73b0\u91d1" in events[2]["content"]
        assert events[-1]["session_id"]

    def test_chat_answers_after_non_json_planner_followup(self, chat_service):
        """When planner returns non-JSON after tool results, it should still answer."""
        chat_service.llm = FakeLLM([
            '{"tool_calls":[{"name":"finance.holdings_summary","arguments":{}}]}',
            "\u5df2\u6709\u6570\u636e\uff0c\u53ef\u4ee5\u56de\u7b54\u3002",  # non-JSON
            "smoke ok",
        ])

        events = list(chat_service.stream_chat("\u786e\u8ba4 cache \u53ef\u8bfb"))

        types = [e["type"] for e in events]
        assert types == ["tool_call", "tool_result", "token", "done"]
        assert events[0]["name"] == "finance.holdings_summary"
        assert events[2]["content"] == "smoke ok"

    def test_reports_error_when_no_llm_key(self, tmp_cache_path):
        config = AgentConfig(
            root_path=tmp_cache_path.parent,
            cache_path=tmp_cache_path,
            trace_path=tmp_cache_path.parent / "trace.jsonl",
            host="127.0.0.1",
            port=0,
        )
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        service = ChatService(config, registry)

        events = list(service.stream_chat("\u4f60\u597d"))
        types = [e["type"] for e in events]
        assert types == ["error", "done"]
        assert "LLM unavailable" in events[0]["message"]

    def test_reports_error_when_empty_message(self, chat_service):
        events = list(chat_service.stream_chat("  "))
        types = [e["type"] for e in events]
        assert types == ["error", "done"]
        assert events[0]["message"] == "message is required"

    def test_session_memory_persists_across_calls(self, chat_service):
        chat_service.llm = FakeLLM([
            '{"tool_calls":[]}',
            "Hello there",
            '{"tool_calls":[]}',
            "How can I help?",
        ])

        sid = None
        for event in chat_service.stream_chat("hi"):
            if event["type"] == "done":
                sid = event["session_id"]
        assert sid is not None
        assert sid in chat_service.memory
        assert len(chat_service.memory[sid]) == 2

        for event in chat_service.stream_chat("help", session_id=sid):
            if event["type"] == "done":
                pass
        assert len(chat_service.memory[sid]) == 4

    def test_reset_clears_session(self, chat_service):
        chat_service.memory["test-session"] = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        chat_service.reset("test-session")
        assert "test-session" not in chat_service.memory

    def test_reset_nonexistent_session_is_noop(self, chat_service):
        chat_service.reset("nonexistent")
        assert "nonexistent" not in chat_service.memory

    def test_tool_call_deduplication(self, chat_service):
        """Duplicate tool calls across planner rounds should be skipped."""
        chat_service.llm = FakeLLM([
            json.dumps({
                "tool_calls": [
                    {"name": "finance.holdings_summary", "arguments": {}},
                ]
            }),
            json.dumps({
                "tool_calls": [
                    {"name": "finance.holdings_summary", "arguments": {}},
                ]
            }),
            '{"tool_calls":[]}',
            "ok",
        ])

        events = list(chat_service.stream_chat("test"))
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) == 1  # cross-round duplicate removed