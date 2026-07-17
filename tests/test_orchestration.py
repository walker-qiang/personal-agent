"""Tests for multi-agent LangGraph orchestration.

Commander + Domain Agents architecture:
  commander_plan → delegate → aggregate → reflection
"""

from __future__ import annotations

import json

import pytest

from matrix.orchestration.graph import build_graph
from matrix.orchestration.nodes import (
    aggregate_node,
    commander_plan_node,
    delegate_node,
    reflection_node,
)
from matrix.orchestration.state import AgentState
from matrix.tools import ToolRegistry, ToolDefinition
from matrix.llm import FunctionCallResult, LLMError, ToolCall
from matrix.agent import AgentRegistry
from matrix.agent.commander import COMMANDER
from matrix.agent.domain_agents import INVESTMENT_ANALYST


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

def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="finance.holdings_summary",
            description="Get holdings summary",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: {"holding_count": 2, "total_value": 100000},
        )
    )
    reg.register(
        ToolDefinition(
            name="finance.bucket_allocation",
            description="Get bucket allocation",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: {"buckets": [{"name": "stock", "target": 40, "current": 45}]},
        )
    )
    reg.register(
        ToolDefinition(
            name="web_search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=lambda **kw: {"results": [{"title": "test", "url": "http://test.com"}]},
        )
    )
    reg.register(
        ToolDefinition(
            name="web_fetch",
            description="Fetch a web page",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            handler=lambda **kw: {"text": "test content"},
        )
    )
    return reg


def _build_agent_registry() -> AgentRegistry:
    reg = AgentRegistry(skills_base_dir="skills")
    reg.register_all([COMMANDER, INVESTMENT_ANALYST])
    return reg


@pytest.fixture
def base_state():
    def _make(**overrides) -> AgentState:
        return {
            "messages": [],
            "user_message": "当前持仓情况如何？",
            "session_id": "test",
            "delegation_plan": [],
            "current_step": 0,
            "agent_results": [],
            "tool_results": [],
            "tool_call_count": 0,
            "react_iteration": 0,
            "final_answer": "",
            "needs_summary": False,
            "error": "",
            **overrides,
        }
    return _make


@pytest.fixture
def full_tools():
    return _build_registry()


@pytest.fixture
def agent_registry():
    return _build_agent_registry()


def make_config(llm, full_tools, agent_registry, trace=None):
    return {
        "configurable": {
            "llm": llm,
            "pipeline_llm": llm,  # use same llm for pipeline in tests
            "full_tools": full_tools,
            "agent_registry": agent_registry,
            "trace": trace,
        },
    }


# ---- Commander Plan ----

class TestCommanderPlanNode:
    def test_generates_plan(self, base_state, full_tools, agent_registry):
        plan_json = json.dumps([
            {"step": 1, "agent_id": "investment-analyst", "task": "分析持仓", "skill_name": "", "purpose": "获取持仓数据"},
        ])
        llm = FakeLLM([plan_json])
        result = commander_plan_node(base_state(), config=make_config(llm, full_tools, agent_registry))
        assert len(result["delegation_plan"]) >= 1
        assert result["delegation_plan"][0]["agent_id"] == "investment-analyst"

    def test_empty_plan_creates_commander_self_plan(self, base_state, full_tools, agent_registry):
        llm = FakeLLM(["[]"])
        result = commander_plan_node(base_state(), config=make_config(llm, full_tools, agent_registry))
        assert len(result["delegation_plan"]) == 1
        assert result["delegation_plan"][0]["agent_id"] == "commander"

    def test_plan_fallback_on_error(self, base_state, full_tools, agent_registry):
        llm = FakeLLM(["not json at all..."])
        result = commander_plan_node(base_state(), config=make_config(llm, full_tools, agent_registry))
        assert len(result["delegation_plan"]) == 1
        assert result["delegation_plan"][0]["agent_id"] == "commander"


# ---- Delegate ----

class TestDelegateNode:
    def test_delegate_executes_agent(self, base_state, full_tools, agent_registry):
        """Domain agent runs ReAct with function calling."""
        from matrix.llm import FunctionCallResult, ToolCall

        class ToolLLM(FakeLLM):
            def __init__(self, responses):
                super().__init__(responses)
                self._fc_count = 0

            def function_call(self, system, messages, tools, tool_choice="auto"):
                self.calls.append(("function_call", messages))
                self._fc_count += 1
                if self._fc_count == 1:
                    return FunctionCallResult(
                        content="",
                        tool_calls=[ToolCall(id="call_1", name="finance.holdings_summary", arguments={})],
                    )
                return FunctionCallResult(content="当前持仓健康，共2个持仓。", tool_calls=[])

        llm = ToolLLM([])
        state = base_state(
            delegation_plan=[
                {"step": 1, "agent_id": "investment-analyst", "task": "分析当前持仓", "skill_name": "", "purpose": "获取持仓"},
            ],
            current_step=0,
        )
        result = delegate_node(state, config=make_config(llm, full_tools, agent_registry))
        assert len(result["agent_results"]) == 1
        assert "持仓" in result["agent_results"][0]["result"]

    def test_delegate_agent_not_found(self, base_state, full_tools, agent_registry):
        llm = FakeLLM([])
        state = base_state(
            delegation_plan=[
                {"step": 1, "agent_id": "nonexistent-agent", "task": "测试", "skill_name": "", "purpose": "测试"},
            ],
            current_step=0,
        )
        result = delegate_node(state, config=make_config(llm, full_tools, agent_registry))
        assert "error" in result["agent_results"][0]

    def test_delegate_with_tool_call(self, base_state, full_tools, agent_registry):
        """Domain agent calls a tool via function calling."""
        from matrix.llm import FunctionCallResult, ToolCall

        class ToolCallLLM(FakeLLM):
            def function_call(self, system, messages, tools, tool_choice="auto"):
                self.calls.append(("function_call", messages))
                if not self.responses:
                    return FunctionCallResult(content="", tool_calls=[])
                text = self.responses.pop(0)
                if text.startswith("TOOL:"):
                    return FunctionCallResult(
                        content="",
                        tool_calls=[ToolCall(name="finance.holdings_summary", arguments={})],
                        finish_reason="tool_calls",
                    )
                return FunctionCallResult(content=text, tool_calls=[])

        llm = ToolCallLLM(["TOOL:", "当前持仓健康。"])
        state = base_state(
            delegation_plan=[
                {"step": 1, "agent_id": "investment-analyst", "task": "分析持仓", "skill_name": "", "purpose": "获取"},
            ],
            current_step=0,
        )
        result = delegate_node(state, config=make_config(llm, full_tools, agent_registry))
        assert len(result["agent_results"]) == 1


# ---- Aggregate ----

class TestAggregateNode:
    def test_aggregate_no_results(self, base_state, full_tools, agent_registry):
        """Empty agent_results should trigger needs_summary."""
        llm = FakeLLM([])
        state = base_state(agent_results=[])
        result = aggregate_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result.get("needs_summary") is True

    def test_aggregate_with_results(self, base_state, full_tools, agent_registry):
        llm = FakeLLM(["当前持仓健康，共2个持仓，总价值100,000元。"])
        state = base_state(
            agent_results=[
                {
                    "agent_id": "investment-analyst",
                    "task": "分析持仓",
                    "result": "持仓健康，共2个持仓。",
                    "error": "",
                },
            ],
        )
        result = aggregate_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result.get("needs_summary") is False
        assert "持仓" in result.get("final_answer", "")

    def test_aggregate_all_errors(self, base_state, full_tools, agent_registry):
        llm = FakeLLM([])
        state = base_state(
            agent_results=[
                {"agent_id": "agent1", "task": "测试", "result": "", "error": "执行失败"},
                {"agent_id": "agent2", "task": "测试", "result": "", "error": "工具不可用"},
            ],
        )
        result = aggregate_node(state, config=make_config(llm, full_tools, agent_registry))
        assert "所有领域专家执行失败" in result.get("final_answer", "")


# ---- Reflection ----

class TestReflectionNode:
    def test_reflection_passes(self, base_state, full_tools, agent_registry):
        llm = FakeLLM(['{"ok": true}'])
        state = base_state(
            final_answer="当前持仓健康，共2个持仓。",
            user_message="当前持仓怎么样？",
        )
        result = reflection_node(state, config=make_config(llm, full_tools, agent_registry))
        assert "final_answer" not in result

    def test_reflection_short_answer_skipped(self, base_state, full_tools, agent_registry):
        llm = FakeLLM([])
        state = base_state(final_answer="OK。", user_message="test")
        result = reflection_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result == {}

    def test_reflection_finds_issues(self, base_state, full_tools, agent_registry):
        llm = FakeLLM([
            '{"ok": false, "issues": ["回答不完整", "缺少数据支撑"]}',
            "修正后的回答：当前持仓配置偏离度为5%，超出目标范围。",
        ])
        state = base_state(
            final_answer="当前持仓看起来不错，应该没问题。",
            user_message="当前持仓的配置偏离度是多少？",
        )
        result = reflection_node(state, config=make_config(llm, full_tools, agent_registry))
        assert "final_answer" in result
        assert "修正" in result["final_answer"] or "偏离" in result["final_answer"]


# ---- Graph Integration ----

class TestGraphIntegration:
    def test_graph_compiles(self, full_tools, agent_registry):
        graph = build_graph()
        assert graph is not None

    def test_graph_simple_path(self, base_state, full_tools, agent_registry):
        """Test simple question path through the compiled graph.
        Empty plan → Commander self-plan → delegate → aggregate → reflection.
        """
        llm = FakeLLM([
            "[]",  # commander_plan: empty → Commander handles it
            "你好！有什么可以帮助你的？",  # delegate: Commander ReAct
            '{"ok": true}',  # reflection
        ])
        graph = build_graph()
        compiled = graph.compile()
        events = list(
            compiled.stream(
                base_state(),
                stream_mode="values",
                config=make_config(llm, full_tools, agent_registry),
                thread_id="test-graph-simple",
            )
        )
        final = events[-1]
        assert len(final.get("agent_results", [])) == 1
        assert final["agent_results"][0]["agent_id"] == "commander"
        assert "帮助" in final.get("final_answer", "")

    def test_graph_delegate_path(self, base_state, full_tools, agent_registry):
        """Test delegate path through the compiled graph."""
        llm = FakeLLM([
            json.dumps([  # commander_plan
                {"step": 1, "agent_id": "investment-analyst", "task": "分析持仓", "skill_name": "", "purpose": "获取"},
            ]),
            "当前持仓健康。",  # delegate (domain agent)
            "汇总：当前持仓健康。",  # aggregate
            '{"ok": true}',  # reflection
        ])
        graph = build_graph()
        compiled = graph.compile()
        events = list(
            compiled.stream(
                base_state(),
                stream_mode="values",
                config=make_config(llm, full_tools, agent_registry),
                thread_id="test-graph-delegate",
            )
        )
        final = events[-1]
        assert len(final.get("agent_results", [])) >= 1
        assert final.get("final_answer") is not None