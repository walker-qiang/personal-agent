"""Tests for LangGraph orchestration graph and nodes."""

from __future__ import annotations

import json

import pytest

from matrix.orchestration.graph import build_graph
from matrix.orchestration.nodes import (
    classify_node,
    execute_node,
    plan_node,
    react_node,
    reflection_node,
    skill_node,
    summarize_node,
)
from matrix.orchestration.state import AgentState
from matrix.tools import ToolRegistry, ToolDefinition
from matrix.llm import FunctionCallResult, LLMError, ToolCall


class FakeLLM:
    """Fake LLM client that returns predefined responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[tuple[str, list]] = []
        self._stream_idx = 0

    def complete(self, system: str, messages: list) -> str:
        self.calls.append(("complete", messages))
        if not self.responses:
            return "{}"
        return self.responses.pop(0)

    def stream_complete(self, system: str, messages: list):
        self.calls.append(("stream", messages))
        text = self.responses.pop(0) if self.responses else ""
        for ch in text:
            yield ch

    def function_call(self, system, messages, tools, tool_choice="auto"):
        self.calls.append(("function_call", messages))
        text = self.responses.pop(0) if self.responses else ""
        return FunctionCallResult(content=text, tool_calls=[])


# ---- Fixtures ----

registry = ToolRegistry()

registry.register(
    ToolDefinition(
        name="finance.holdings_summary",
        description="Get holdings summary",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda **kw: {"holding_count": 2, "total_value": 100000},
    )
)
registry.register(
    ToolDefinition(
        name="finance.bucket_allocation",
        description="Get bucket allocation",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda **kw: {"buckets": [{"name": "stock", "target": 40, "current": 45}]},
    )
)


@pytest.fixture
def base_state():
    """Return a factory that creates AgentState with optional overrides."""
    def _make(**overrides) -> AgentState:
        return {
            "messages": [],
            "user_message": "当前持仓情况如何？",
            "session_id": "test",
            "intent": "",
            "skill_name": "",
            "tool_results": [],
            "tool_call_count": 0,
            "current_plan": [],
            "react_iteration": 0,
            "findings": [],
            "final_answer": "",
            "needs_summary": False,
            "error": "",
            **overrides,
        }
    return _make


def make_config(llm, tools, skills=None, trace=None, skills_dir=None):
    return {
        "configurable": {
            "llm": llm,
            "tools": tools,
            "skills": skills or [],
            "trace": trace,
            "skills_dir": skills_dir,
        },
    }


# ---- Classify ----

class TestClassifyNode:
    def test_classifies_react(self, base_state):
        llm = FakeLLM(['{"intent": "react"}'])
        result = classify_node(base_state(), config=make_config(llm, registry))
        assert result["intent"] == "react"

    def test_classifies_plan_execute(self, base_state):
        llm = FakeLLM(['{"intent": "plan_execute"}'])
        result = classify_node(base_state(), config=make_config(llm, registry))
        assert result["intent"] == "plan_execute"

    def test_classify_skill_keyword_match(self, base_state):
        """Keyword matching should trigger without LLM call."""
        from matrix.skills import SkillDefinition
        skill = SkillDefinition(
            name="test-skill",
            title="测试",
            description="测试描述",
            workflow=[],
        )
        llm = FakeLLM([])
        state = base_state(user_message="跑测试技能")
        result = classify_node(state, config=make_config(llm, registry, [skill]))
        assert result["intent"] == "skill"

    def test_classify_fallback_on_error(self, base_state):
        """No responses → LLM will fail, classify_node should catch and return react."""
        llm = FakeLLM([])
        result = classify_node(base_state(), config=make_config(llm, registry))
        assert result["intent"] == "react"


# ---- React ----

class TestReactNode:
    def test_react_returns_final_answer(self, base_state):
        """Native function calling: LLM returns content directly."""
        llm = FakeLLM(["当前持仓健康。"])

        result = react_node(base_state(), config=make_config(llm, registry))
        assert "final_answer" in result or "needs_summary" in result

    def test_react_fallback_regex(self, base_state):
        """Regex fallback: LLM returns Final Answer via text."""
        llm = FakeLLM(["Final Answer: 当前持仓健康。"])

        result = react_node(base_state(), config=make_config(llm, registry))
        assert "final_answer" in result or "needs_summary" in result

    def test_react_max_iterations(self, base_state):
        llm = FakeLLM(["当前持仓健康。"])
        state = base_state(react_iteration=5)
        result = react_node(state, config=make_config(llm, registry))
        assert "final_answer" in result

    def test_react_fallback_tool_call(self, base_state):
        """Regex fallback: LLM calls a tool. Force fallback by raising on function_call."""
        class FallbackFakeLLM(FakeLLM):
            def function_call(self, system, messages, tools, tool_choice="auto"):
                raise LLMError("simulated function_call failure")

        llm = FallbackFakeLLM([
            "Thought: 查询持仓\nAction: finance.holdings_summary\nAction Input: {}",
        ])
        result = react_node(base_state(), config=make_config(llm, registry))
        assert result["tool_call_count"] >= 1


# ---- Plan ----

class TestPlanNode:
    def test_plan_generates_steps(self, base_state):
        llm = FakeLLM([
            json.dumps([
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}, "purpose": "获取持仓"},
            ]),
        ])
        result = plan_node(base_state(), config=make_config(llm, registry))
        assert len(result["current_plan"]) >= 1

    def test_plan_fallback_on_error(self, base_state):
        llm = FakeLLM(["not json at all..."])
        result = plan_node(base_state(), config=make_config(llm, registry))
        assert len(result["current_plan"]) >= 1  # fallback plan


# ---- Execute ----

class TestExecuteNode:
    def test_execute_one_step(self, base_state):
        state = base_state(
            current_plan=[
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}, "purpose": "获取持仓"},
            ],
            tool_call_count=0,
        )
        result = execute_node(state, config=make_config(FakeLLM([]), registry))
        assert result["tool_call_count"] == 1
        assert len(result["tool_results"]) == 1


# ---- Summarize ----

class TestSummarizeNode:
    def test_uses_existing_final_answer(self, base_state):
        state = base_state(final_answer="已有答案")
        result = summarize_node(state, config=make_config(FakeLLM([]), registry))
        assert result["final_answer"] == "已有答案"

    def test_sets_needs_summary_when_tool_results(self, base_state):
        state = base_state(
            tool_results=[
                {"name": "finance.holdings_summary", "result": {"holding_count": 2}},
            ],
        )
        result = summarize_node(state, config=make_config(FakeLLM([]), registry))
        assert result.get("needs_summary") is True
        assert "final_answer" not in result

    def test_handles_no_tool_results(self, base_state):
        result = summarize_node(base_state(), config=make_config(FakeLLM([]), registry))
        assert "未获取到任何数据" in result["final_answer"]


# ---- Skill ----

class TestSkillNode:
    def test_executes_skill_workflow(self, base_state):
        from matrix.skills import SkillDefinition
        skill = SkillDefinition(
            name="test-skill",
            title="测试技能",
            description="测试",
            workflow=[
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
            ],
        )
        llm = FakeLLM([])
        state = base_state(intent="skill", skill_name="test-skill")
        result = skill_node(state, config=make_config(llm, registry, [skill]))
        assert result["tool_call_count"] == 1
        assert len(result["tool_results"]) == 1


# ---- Reflection ----

class TestReflectionNode:
    def test_reflection_passes(self, base_state):
        llm = FakeLLM(['{"ok": true}'])
        state = base_state(
            final_answer="当前持仓健康，共2个持仓。",
            user_message="当前持仓怎么样？",
        )
        result = reflection_node(state, config=make_config(llm, registry))
        assert "final_answer" not in result or result.get("final_answer") == "当前持仓健康，共2个持仓。"

    def test_reflection_short_answer_skipped(self, base_state):
        """Short answers (< 15 chars) skip reflection."""
        llm = FakeLLM([])
        state = base_state(final_answer="OK。", user_message="test")
        result = reflection_node(state, config=make_config(llm, registry))
        assert result == {}

    def test_reflection_finds_issues(self, base_state):
        llm = FakeLLM([
            '{"ok": false, "issues": ["回答不完整", "缺少数据支撑"]}',
        ])
        state = base_state(
            final_answer="当前持仓看起来不错，应该没问题。",
            user_message="当前持仓的配置偏离度是多少？",
        )
        result = reflection_node(state, config=make_config(llm, registry))
        assert "final_answer" in result
        assert "自检发现问题" in result["final_answer"]


# ---- Graph integration ----

class TestGraphIntegration:
    def test_graph_compiles(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_react_path(self, base_state):
        """Test react path through the compiled graph."""
        llm = FakeLLM([
            '{"intent": "react"}',
            "当前持仓健康。",
        ])
        graph = build_graph()
        compiled = graph.compile()
        events = list(
            compiled.stream(
                base_state(),
                stream_mode="values",
                config=make_config(llm, registry),
                thread_id="test-graph",
            )
        )
        final = events[-1]
        assert final.get("needs_summary") is True or final.get("final_answer") is not None

    def test_graph_skill_path(self, base_state):
        """Test skill path through the compiled graph."""
        from matrix.skills import SkillDefinition
        skill = SkillDefinition(
            name="test-skill",
            title="测试",
            description="测试",
            workflow=[
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
            ],
        )
        llm = FakeLLM([
            '{"intent": "skill", "skill_name": "test-skill"}',
        ])
        graph = build_graph()
        compiled = graph.compile()
        state = base_state(intent="skill", skill_name="test-skill")
        events = list(
            compiled.stream(
                state,
                stream_mode="values",
                config=make_config(llm, registry, [skill]),
                thread_id="test-graph-skill",
            )
        )
        final = events[-1]
        assert final.get("tool_call_count", 0) >= 1
        assert final.get("needs_summary") is True

    def test_graph_needs_summary_path(self, base_state):
        """Test needs_summary → streaming summarization: skill path with tool_results."""
        from matrix.skills import SkillDefinition
        skill = SkillDefinition(
            name="test-skill",
            title="测试",
            description="测试",
            workflow=[
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
            ],
        )
        llm = FakeLLM([])  # skill path doesn't call LLM in classify (intent pre-set)
        graph = build_graph()
        compiled = graph.compile()
        state = base_state(intent="skill", skill_name="test-skill")
        events = list(
            compiled.stream(
                state,
                stream_mode="values",
                config=make_config(llm, registry, [skill]),
                thread_id="test-needs-summary",
            )
        )
        final = events[-1]
        # skill_node executes workflow → tool_results, summarize_node sets needs_summary=True
        assert final.get("tool_call_count", 0) >= 1
        assert final.get("needs_summary") is True


class TestFunctionCallPath:
    """Tests for the native function calling path in react_node."""

    def test_function_call_returns_tool_calls(self, base_state):
        """Test react_node using native function calling that returns tool_calls."""
        from matrix.llm import FunctionCallResult, ToolCall

        class ToolCallLLM(FakeLLM):
            def function_call(self, system, messages, tools, tool_choice="auto"):
                self.calls.append(("function_call", messages))
                return FunctionCallResult(
                    content="",
                    tool_calls=[
                        ToolCall(name="finance.holdings_summary", arguments={}),
                    ],
                    finish_reason="tool_calls",
                )

        llm = ToolCallLLM([])
        result = react_node(base_state(), config=make_config(llm, registry))
        assert result["tool_call_count"] >= 1
        assert len(result["tool_results"]) >= 1

    def test_function_call_returns_direct_answer(self, base_state):
        """Test react_node when function_call returns content directly (no tools)."""
        from matrix.llm import FunctionCallResult

        class DirectAnswerLLM(FakeLLM):
            def function_call(self, system, messages, tools, tool_choice="auto"):
                self.calls.append(("function_call", messages))
                return FunctionCallResult(
                    content="当前持仓健康，共2个持仓。",
                    tool_calls=[],
                    finish_reason="stop",
                )

        llm = DirectAnswerLLM([])
        result = react_node(base_state(), config=make_config(llm, registry))
        assert "final_answer" in result
        assert "持仓" in result["final_answer"]

    def test_execute_tool_calls_dedup(self, base_state):
        """Test that _execute_tool_calls skips duplicate tool calls."""
        from matrix.llm import FunctionCallResult, ToolCall

        class DupToolCallLLM(FakeLLM):
            def function_call(self, system, messages, tools, tool_choice="auto"):
                self.calls.append(("function_call", messages))
                return FunctionCallResult(
                    content="",
                    tool_calls=[
                        ToolCall(name="finance.holdings_summary", arguments={}),
                        ToolCall(name="finance.holdings_summary", arguments={}),
                    ],
                    finish_reason="tool_calls",
                )

        llm = DupToolCallLLM([])
        result = react_node(base_state(), config=make_config(llm, registry))
        # Two identical calls, only one should execute
        assert result["tool_call_count"] == 1

    def test_execute_tool_calls_unknown_tool(self, base_state):
        """Test that _execute_tool_calls skips unknown tools."""
        from matrix.llm import FunctionCallResult, ToolCall

        class UnknownToolLLM(FakeLLM):
            def function_call(self, system, messages, tools, tool_choice="auto"):
                self.calls.append(("function_call", messages))
                return FunctionCallResult(
                    content="",
                    tool_calls=[
                        ToolCall(name="nonexistent.tool", arguments={}),
                        ToolCall(name="finance.holdings_summary", arguments={}),
                    ],
                    finish_reason="tool_calls",
                )

        llm = UnknownToolLLM([])
        result = react_node(base_state(), config=make_config(llm, registry))
        # Only holdings_summary should execute
        assert result["tool_call_count"] == 1