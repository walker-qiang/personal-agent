"""Tests for ChatService orchestration."""

from __future__ import annotations

import time
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
        self.provider = "test"
        self.model = "test-model"

    def complete(self, system: str, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append(("complete", messages))
        if not self.responses:
            raise AssertionError("no fake LLM responses left")
        return self.responses.pop(0)

    def complete_json(self, system: str, messages: list[dict[str, str]], schema=None, **kwargs):
        """Fake JSON completion: parse the next response as JSON."""
        import json
        self.calls.append(("complete_json", messages))
        if not self.responses:
            raise AssertionError("no fake LLM responses left")
        text = self.responses.pop(0)
        try:
            return json.loads(text) if isinstance(text, str) else text
        except (json.JSONDecodeError, TypeError):
            return []

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
        skills_base_dir=tmp_cache_path.parent / "skills",
        host="127.0.0.1",
        port=0,
        deepseek_api_key="test-key",
        agnes_api_key="test-key",
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

    def test_memory_extraction_formats_json_prompt_and_persists(self, chat_service):
        chat_service._pipeline_llm = FakeLLM([
            '{"memories": [{"key": "language", "value": "中文", "type": "preference"}]}',
        ])

        chat_service._extract_memories(
            "我希望以后都用中文回答。",
            "好的，后续我会使用中文。",
            "memory-user",
        )

        assert chat_service.store.get_profile("memory-user") == {"language": "中文"}

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
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        types = [e["type"] for e in events]
        assert "done" in types
        assert "token" in types

    def test_returns_token_events(self, chat_service):
        chat_service._default_llm = FakeLLM([
            "当前持仓健康。",
        ])
        chat_service._pipeline_llm = FakeLLM(["[]"])
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        tokens = [e for e in events if e["type"] == "token"]
        assert len(tokens) >= 1, f"events={[(e['type'], e.get('content','')[:60]) for e in events]}"

    def test_client_disconnect_closes_stream_without_yielding_done(self, chat_service):
        llm = FakeLLM(["unused"])
        llm.provider = "codex"
        chat_service._default_llm = llm
        stream = chat_service.stream_chat("close after first event", session_id="disconnect-test")

        assert next(stream)["type"] == "classify"
        stream.close()

    def test_codex_direct_runtime_persists_completed_operation(self, chat_service):
        llm = FakeLLM(["Codex 运行时适配完成。"])
        llm.provider = "codex"
        chat_service._default_llm = llm

        events = list(chat_service.stream_chat(
            "验证 Codex Runtime", session_id="codex-runtime-test", user_id="codex-user",
        ))

        assert any(event["type"] == "token" for event in events)
        operations = chat_service._runtime_store.list_operations(
            "codex-user", session_id="codex-runtime-test",
        )
        assert len(operations) == 1
        assert operations[0].phase.value == "completed"

    def test_image_attachment_uses_runtime_multimodal_message(self, chat_service):
        upload_dir = chat_service.config.root_path.parent / "var" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / "runtime-image.png").write_bytes(b"fake-png")
        chat_service._default_llm = FakeLLM(["图片已收到。"])
        chat_service._pipeline_llm = FakeLLM(["[]"])

        events = list(chat_service.stream_chat(
            "请描述图片", session_id="runtime-image", user_id="image-user",
            file_id="runtime-image",
        ))

        assert any(event["type"] == "token" for event in events)
        user_entries = chat_service._runtime_store.list_session_entries(
            "image-user", "runtime-image", entry_type="user", limit=1,
        )
        assert isinstance(user_entries[0]["payload"]["content"], list)
        assert any(
            call[0] == "function_call"
            and isinstance(call[1][-1]["content"], list)
            for call in chat_service._default_llm.calls
        )
        assert chat_service._runtime_store.list_operations(
            "image-user", session_id="runtime-image",
        )[0].phase.value == "completed"

    def test_runtime_history_keeps_legacy_prefix_and_runtime_suffix(self, chat_service):
        sid = "mixed-history"
        user_id = "history-user"
        chat_service.store.save_message(sid, "user", "旧问题", user_id=user_id)
        chat_service.store.save_message(sid, "assistant", "旧回答", user_id=user_id)
        chat_service._runtime_store.append_session_entry(
            user_id, sid, "user", {"content": "新问题"},
        )
        chat_service._runtime_store.append_session_entry(
            user_id, sid, "assistant", {"content": "新回答"},
        )

        history = chat_service._get_history(sid, user_id)

        assert [(item["role"], item["content"]) for item in history] == [
            ("user", "旧问题"), ("assistant", "旧回答"),
            ("user", "新问题"), ("assistant", "新回答"),
        ]

    def test_session_memory_persists(self, chat_service):
        chat_service._default_llm = FakeLLM([
            "持仓健康。",
            "仍然健康。",
        ])
        chat_service._pipeline_llm = FakeLLM(["[]", "[]"])
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
        sid = "reset-test"
        list(chat_service.stream_chat("test", sid))
        chat_service.reset(sid)
        assert len(chat_service._get_history(sid)) == 0
        assert chat_service._runtime_store.list_session_entries(
            "default", sid, limit=20,
        ) == []

    def test_skill_flow_in_graph(self, chat_service):
        chat_service._default_llm = FakeLLM([
            "技能执行完成，共2个持仓。",
        ])
        chat_service._pipeline_llm = FakeLLM(["[]"])
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
        events = list(chat_service.stream_chat("当前持仓怎么样？"))
        tokens = [e for e in events if e["type"] == "token"]
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_calls) >= 1, "Should have tool_call events"
        assert len(tokens) >= 1, "Should have streaming token events"

    def test_branch_summary_is_idempotent_and_recovered(self, chat_service):
        session_id = "branch-summary-recovery"
        user_id = "branch-user"
        from_message_id = chat_service.store.save_message(
            session_id, "user", "分叉起点", user_id=user_id,
        )
        abandoned_leaf_id = chat_service.store.save_message(
            session_id, "assistant", "旧分支内容", user_id=user_id,
        )
        chat_service._pipeline_llm = FakeLLM([
            {"summary": "旧分支讨论了一个事实。", "key_points": ["事实"], "unresolved": ""},
        ])

        entry_id, _ = chat_service._runtime_store.ensure_branch_summary_entry(
            user_id, session_id, from_message_id, abandoned_leaf_id, 1,
        )
        chat_service._recover_branch_summaries()
        for _ in range(100):
            entry = chat_service._runtime_store.get_session_entry(user_id, entry_id)
            if entry and entry["payload"].get("status") == "completed":
                break
            time.sleep(0.01)

        assert entry is not None
        assert entry["payload"]["status"] == "completed"
        assert len(chat_service._pipeline_llm.calls) == 1
        chat_service._recover_branch_summaries()
        assert len(chat_service._runtime_store.list_session_entries(
            user_id, session_id, entry_type="branch_summary",
        )) == 1

    def test_branch_summary_retries_and_persists_failure(self, chat_service):
        class AlwaysFailLLM(FakeLLM):
            def complete_json(self, system, messages, schema=None, **kwargs):
                self.calls.append(("complete_json", messages))
                raise RuntimeError("temporary model failure")

        session_id = "branch-summary-failure"
        user_id = "branch-user"
        from_message_id = chat_service.store.save_message(
            session_id, "user", "分叉起点", user_id=user_id,
        )
        abandoned_leaf_id = chat_service.store.save_message(
            session_id, "assistant", "旧分支内容", user_id=user_id,
        )
        entry_id, _ = chat_service._runtime_store.ensure_branch_summary_entry(
            user_id, session_id, from_message_id, abandoned_leaf_id, 1,
        )
        chat_service._pipeline_llm = AlwaysFailLLM([])

        chat_service._generate_branch_summary(
            entry_id, session_id, from_message_id, abandoned_leaf_id,
            user_id, [{"role": "assistant", "content": "旧分支内容"}],
        )

        entry = chat_service._runtime_store.get_session_entry(user_id, entry_id)
        assert entry is not None
        assert entry["payload"]["status"] == "failed"
        assert entry["payload"]["attempts"] == 2
        assert "temporary model failure" in entry["payload"]["error"]
        assert len(chat_service._pipeline_llm.calls) == 2
