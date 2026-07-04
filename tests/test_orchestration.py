"""Tests for LangGraph orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

from matrix.orchestration import build_graph
from matrix.orchestration.nodes import (
    classify_node,
    execute_node,
    plan_node,
    react_node,
    skill_node,
    summarize_node,
)
from matrix.orchestration.state import AgentState
from matrix.skills import SkillDefinition
from matrix.tools import ToolRegistry
from matrix.tools.finance import register_all


class FakeLLM:
    """Fake LLM client that returns predefined responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[tuple[str, list]] = []

    def complete(self, system: str, messages: list) -> str:
        self.calls.append((system, messages))
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


def make_config(llm, tools):
    """Helper to wrap llm and tools in LangGraph config format."""
    return {"configurable": {"llm": llm, "tools": tools}}


@pytest.fixture
def registry(tmp_cache_path: Path) -> ToolRegistry:
    r = ToolRegistry()
    register_all(r, tmp_cache_path)
    return r


@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        messages=[],
        user_message="测试",
        session_id="test-session",
        intent="",
        skill_name="",
        tool_results=[],
        tool_call_count=0,
        current_plan=[],
        react_iteration=0,
        findings=[],
        final_answer="",
        error="",
    )


class TestClassifyNode:
    def test_classifies_as_react(self, base_state):
        llm = FakeLLM(['{"intent": "react", "skill_name": ""}'])
        result = classify_node(base_state, config=make_config(llm, registry))
        assert result["intent"] == "react"

    def test_classifies_as_plan_execute(self, base_state):
        llm = FakeLLM(['{"intent": "plan_execute", "skill_name": ""}'])
        result = classify_node(base_state, config=make_config(llm, registry))
        assert result["intent"] == "plan_execute"

    def test_falls_back_to_react_on_error(self, base_state):
        llm = FakeLLM(["not valid json"])
        result = classify_node(base_state, config=make_config(llm, registry))
        assert result["intent"] == "react"


class TestReactNode:
    def test_calls_tool_and_returns_results(self, base_state, registry):
        llm = FakeLLM([
            "Thought: 需要查询持仓\n"
            "Action: finance.holdings_summary\n"
            "Action Input: {}",
        ])
        result = react_node(base_state, config=make_config(llm, registry))
        assert result["tool_call_count"] == 1
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["name"] == "finance.holdings_summary"
        assert result["react_iteration"] == 1

    def test_returns_final_answer(self, base_state, registry):
        llm = FakeLLM([
            "Thought: 已有足够数据\n"
            "Final Answer: 当前持仓健康。",
        ])
        result = react_node(base_state, config=make_config(llm, registry))
        assert "当前持仓健康" in result["final_answer"]

    def test_handles_unknown_tool(self, base_state, registry):
        llm = FakeLLM([
            "Thought: 尝试未知工具\n"
            "Action: finance.unknown\n"
            "Action Input: {}",
        ])
        result = react_node(base_state, config=make_config(llm, registry))
        assert "不可用" in result.get("final_answer", "")

    def test_respects_max_iterations(self, base_state, registry):
        state = {**base_state, "react_iteration": 5}
        llm = FakeLLM([])
        result = react_node(state, config=make_config(llm, registry))
        assert "最大分析步数" in result.get("final_answer", "")


class TestPlanNode:
    def test_generates_plan(self, base_state, registry):
        llm = FakeLLM([
            '[{"step": 1, "tool": "finance.holdings_summary", "arguments": {}, "purpose": "获取持仓"}]',
        ])
        result = plan_node(base_state, config=make_config(llm, registry))
        assert len(result["current_plan"]) == 1
        assert result["current_plan"][0]["tool"] == "finance.holdings_summary"

    def test_fallback_plan_on_error(self, base_state, registry):
        llm = FakeLLM(["not json at all"])
        result = plan_node(base_state, config=make_config(llm, registry))
        assert len(result["current_plan"]) >= 1


class TestExecuteNode:
    def test_executes_first_step(self, base_state, registry):
        state = {
            **base_state,
            "current_plan": [
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
            ],
        }
        llm = FakeLLM([])
        result = execute_node(state, config=make_config(llm, registry))
        assert result["tool_call_count"] == 1
        assert len(result["tool_results"]) == 1

    def test_executes_second_step(self, base_state, registry):
        state = {
            **base_state,
            "current_plan": [
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
                {"step": 2, "tool": "finance.bucket_allocation", "arguments": {}},
            ],
            "tool_call_count": 1,
            "tool_results": [{"name": "finance.holdings_summary", "result": {"test": True}}],
        }
        llm = FakeLLM([])
        result = execute_node(state, config=make_config(llm, registry))
        assert result["tool_call_count"] == 2
        assert len(result["tool_results"]) == 2

    def test_returns_final_answer_when_plan_done(self, base_state, registry):
        state = {
            **base_state,
            "current_plan": [
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
            ],
            "tool_call_count": 1,
        }
        llm = FakeLLM([])
        result = execute_node(state, config=make_config(llm, registry))
        assert "计划执行完毕" in result.get("final_answer", "")


class TestSummarizeNode:
    def test_uses_existing_final_answer(self, base_state):
        state = {**base_state, "final_answer": "已有答案"}
        llm = FakeLLM([])
        result = summarize_node(state, config=make_config(llm, registry))
        assert result["final_answer"] == "已有答案"

    def test_generates_answer_when_no_final_answer(self, base_state):
        state = {
            **base_state,
            "tool_results": [
                {"name": "finance.holdings_summary", "result": {"holding_count": 2}},
            ],
        }
        llm = FakeLLM(["回答: 共有2个持仓。"])
        result = summarize_node(state, config=make_config(llm, registry))
        assert "2个持仓" in result["final_answer"]

    def test_handles_no_tool_results(self, base_state):
        llm = FakeLLM([])
        result = summarize_node(base_state, config=make_config(llm, registry))
        assert "未获取到任何数据" in result["final_answer"]


class TestBuildGraph:
    def test_builds_compilable_graph(self):
        graph = build_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_react_flow(self, registry):
        """End-to-end test: reactive flow with FakeLLM."""
        graph = build_graph()
        compiled = graph.compile()

        llm = FakeLLM([
            # classify
            '{"intent": "react", "skill_name": ""}',
            # react
            "Thought: 查询持仓\n"
            "Action: finance.holdings_summary\n"
            "Action Input: {}",
            # react again
            "Thought: 已有数据\n"
            "Final Answer: 当前持仓健康，共2个资产。",
        ])

        state: AgentState = {
            "messages": [],
            "user_message": "当前持仓怎么样？",
            "session_id": "test",
            "intent": "",
            "skill_name": "",
            "tool_results": [],
            "tool_call_count": 0,
            "current_plan": [],
            "react_iteration": 0,
            "findings": [],
            "final_answer": "",
            "error": "",
        }

        events = list(compiled.stream(
            state,
            stream_mode="values",
            config={"configurable": {"llm": llm, "tools": registry}},
        ))
        assert len(events) > 0
        final = events[-1]
        assert isinstance(final, dict)
        assert "当前持仓健康" in final.get("final_answer", "")

    def test_skill_flow(self, registry):
        """End-to-end test: skill execution flow."""
        graph = build_graph()
        compiled = graph.compile()

        skill = SkillDefinition(
            name="test-skill",
            title="测试技能",
            description="测试用技能",
            trigger_keywords=["测试"],
            workflow=[
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
            ],
        )

        llm = FakeLLM([
            # classify returns skill intent (keyword match bypasses LLM)
            '{"intent": "skill", "skill_name": "test-skill"}',
        ])

        state: AgentState = {
            "messages": [],
            "user_message": "跑一下测试技能",
            "session_id": "test",
            "intent": "",
            "skill_name": "",
            "tool_results": [],
            "tool_call_count": 0,
            "current_plan": [],
            "react_iteration": 0,
            "findings": [],
            "final_answer": "",
            "error": "",
        }

        events = list(compiled.stream(
            state,
            stream_mode="values",
            config={"configurable": {
                "llm": llm,
                "tools": registry,
                "skills": [skill],
            }},
        ))
        final = events[-1]
        assert final.get("tool_call_count", 0) >= 1
        # summarize_node should generate answer from tool results
        assert len(final.get("final_answer", "")) > 0


class TestSkillNode:
    def test_executes_skill(self, registry):
        skill = SkillDefinition(
            name="test-skill",
            title="测试技能",
            workflow=[
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
            ],
        )
        llm = FakeLLM([])
        state: AgentState = {
            "messages": [],
            "user_message": "test",
            "session_id": "test",
            "intent": "skill",
            "skill_name": "test-skill",
            "tool_results": [],
            "tool_call_count": 0,
            "current_plan": [],
            "react_iteration": 0,
            "findings": [],
            "final_answer": "",
            "error": "",
        }
        result = skill_node(state, config={
            "configurable": {"llm": llm, "tools": registry, "skills": [skill]},
        })
        assert result["tool_call_count"] == 1
        assert len(result["tool_results"]) == 1
        # skill_node should NOT set final_answer — let summarize_node do it
        assert result.get("final_answer", "") == ""

    def test_unknown_skill(self, registry):
        llm = FakeLLM([])
        state: AgentState = {
            "messages": [],
            "user_message": "test",
            "session_id": "test",
            "intent": "skill",
            "skill_name": "nonexistent",
            "tool_results": [],
            "tool_call_count": 0,
            "current_plan": [],
            "react_iteration": 0,
            "findings": [],
            "final_answer": "",
            "error": "",
        }
        result = skill_node(state, config={
            "configurable": {"llm": llm, "tools": registry, "skills": []},
        })
        assert "未找到" in result["final_answer"]
        assert result.get("error") is not None

    def test_skill_with_error(self, registry):
        skill = SkillDefinition(
            name="test-skill",
            title="测试技能",
            workflow=[
                {"step": 1, "tool": "finance.unknown_tool", "arguments": {}},
            ],
        )
        llm = FakeLLM([])
        state: AgentState = {
            "messages": [],
            "user_message": "test",
            "session_id": "test",
            "intent": "skill",
            "skill_name": "test-skill",
            "tool_results": [],
            "tool_call_count": 0,
            "current_plan": [],
            "react_iteration": 0,
            "findings": [],
            "final_answer": "",
            "error": "",
        }
        result = skill_node(state, config={
            "configurable": {"llm": llm, "tools": registry, "skills": [skill]},
        })
        assert "错误" in result["final_answer"]
        assert result.get("error") is not None


class TestExtractJson:
    """Tests for _extract_json helper (handles both objects and arrays)."""

    def test_extracts_object(self):
        from matrix.orchestration.nodes import _extract_json
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_extracts_array(self):
        from matrix.orchestration.nodes import _extract_json
        result = _extract_json('[{"step": 1, "tool": "test"}]')
        assert isinstance(result, list)
        assert result[0]["step"] == 1

    def test_extracts_fenced_array(self):
        from matrix.orchestration.nodes import _extract_json
        result = _extract_json('```json\n[{"a": 1}]\n```')
        assert isinstance(result, list)
        assert result[0]["a"] == 1

    def test_extracts_json_with_prefix(self):
        from matrix.orchestration.nodes import _extract_json
        result = _extract_json('prefix text {"key": 1}')
        assert result == {"key": 1}