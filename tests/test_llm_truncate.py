"""Tests for message truncation under a tight token budget.

The failures this file guards against are structural, not cosmetic: a
starved history drops tool results, which makes the model repeat the same
call and makes strict tool APIs reject the request outright.
"""

from __future__ import annotations

from matrix.llm.truncate import estimate_tokens, truncate_messages


def _assistant_call(call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "finance.asset_lookup", "arguments": "{}"},
        }],
    }


def _tool_output(call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": '{"error": "no data"}'}


def _history() -> list[dict]:
    return [
        {"role": "user", "content": "五粮液现在能买吗"},
        _assistant_call("call-1"),
        _tool_output("call-1"),
        _assistant_call("call-2"),
        _tool_output("call-2"),
    ]


class TestBudgetIsNotStarvedBySystemPrompt:
    def test_keeps_tool_results_when_system_prompt_exhausts_budget(self):
        """A huge system prompt must not collapse the history to one message."""
        system = "你是一个投资分析师。" * 3000  # far beyond the 4000 token budget

        result = truncate_messages(
            _history(), system_prompt=system, max_tokens=4000, reserve_tokens=500,
        )

        assert any(m.get("role") == "tool" for m in result), result
        assert len(result) >= 2

    def test_keeps_latest_tool_exchange_pairing(self):
        system = "x" * 20000
        result = truncate_messages(
            _history(), system_prompt=system, max_tokens=4000, reserve_tokens=500,
        )

        call_ids = {
            m["tool_call_id"] for m in result if m.get("role") == "tool"
        }
        assistant_ids = {
            tc["id"]
            for m in result
            if m.get("role") == "assistant"
            for tc in m.get("tool_calls", [])
        }
        assert call_ids, result
        assert call_ids <= assistant_ids or not assistant_ids

    def test_still_truncates_when_history_is_long(self):
        long_history = [
            {"role": "user", "content": f"第{i}轮问题" + "内容" * 200}
            for i in range(40)
        ]
        result = truncate_messages(
            long_history, system_prompt="简短", max_tokens=4000, reserve_tokens=500,
        )

        assert len(result) < len(long_history)


class TestStructuralValidity:
    def test_drops_orphan_tool_messages(self):
        messages = [
            {"role": "user", "content": "查询"},
            {"role": "tool", "tool_call_id": "orphan", "content": "{}"},
        ]
        result = truncate_messages(messages, max_tokens=4000, reserve_tokens=500)

        assert all(m.get("role") != "tool" for m in result)

    def test_never_returns_empty_for_non_empty_input(self):
        result = truncate_messages([_tool_output("call-1")])

        assert result

    def test_preserves_most_recent_message(self):
        history = _history() + [{"role": "user", "content": "换个说法再查一次"}]
        result = truncate_messages(history, max_tokens=4000, reserve_tokens=500)

        assert result[-1]["content"] == "换个说法再查一次"


class TestTokenEstimation:
    def test_chinese_costs_more_than_ascii(self):
        assert estimate_tokens("中文") > estimate_tokens("ab")

    def test_empty_text_is_zero(self):
        assert estimate_tokens("") == 0
