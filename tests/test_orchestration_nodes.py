"""Additional tests for orchestration node internals.

Covers: hallucination detection, tool deduplication, tool error handling,
tool gate, and skill matching.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from matrix.orchestration.nodes import (
    _is_hallucination,
    _force_tool_call,
    _run_tool_calls,
    _build_tools_for_llm,
)
from matrix.tools import ToolRegistry, ToolDefinition, FinanceToolError
from matrix.llm import FunctionCallResult, ToolCall, LLMError
from matrix.agent import AgentRegistry
from matrix.agent.commander import COMMANDER
from matrix.agent.domain_agents import INVESTMENT_ANALYST


# ---- Hallucination Detection ----

class TestHallucinationDetection:
    def test_detects_generated_pattern_chinese(self):
        assert _is_hallucination("已为您生成了一张图片")
        assert _is_hallucination("已经生成了视频")
        assert _is_hallucination("已创建了文档")
        assert _is_hallucination("已完成制作")

    def test_detects_result_pattern(self):
        assert _is_hallucination("生成结果如下：")
        assert _is_hallucination("具体效果如下：")

    def test_detects_english_patterns(self):
        assert _is_hallucination("Here is the generated image:")
        assert _is_hallucination("I have created a video")

    def test_normal_response_not_hallucination(self):
        assert not _is_hallucination("当前持仓健康，共2个持仓。")
        assert not _is_hallucination("根据分析，你的配置偏离度为5%。")
        assert not _is_hallucination("Let me check your holdings first.")
        assert not _is_hallucination("")

    def test_edge_cases(self):
        # Partial match in non-hallucination context
        assert _is_hallucination("已生成一个测试文件")
        # Not matched
        assert not _is_hallucination("生成是机器学习的重要环节")


# ---- Force Tool Call ----

class TestForceToolCall:
    def test_force_tool_call_success(self):
        """LLM responds with a tool call when forced."""

        class ForcedLLM:
            def function_call(self, system, messages, tools, tool_choice="auto"):
                # Verify tool_choice is "required"
                assert tool_choice == "required"
                return FunctionCallResult(
                    content="",
                    tool_calls=[ToolCall(name="finance.holdings_summary", arguments={})],
                    finish_reason="tool_calls",
                )

        llm = ForcedLLM()
        result = _force_tool_call(
            llm,
            system_prompt="You are a domain expert.",
            task="分析持仓",
            tools=[{"name": "finance.holdings_summary", "description": "Get holdings"}],
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "finance.holdings_summary"

    def test_force_tool_call_error_returns_empty(self):
        """LLM API error returns empty FunctionCallResult."""

        class ErrorLLM:
            def function_call(self, system, messages, tools, tool_choice="auto"):
                raise LLMError("API error")

        llm = ErrorLLM()
        result = _force_tool_call(
            llm,
            system_prompt="You are a domain expert.",
            task="分析持仓",
            tools=[{"name": "finance.holdings_summary", "description": "Get holdings"}],
        )
        assert result.content == ""
        assert len(result.tool_calls) == 0

    def test_force_tool_call_hallucination_returns_content(self):
        """_force_tool_call passes through LLM response (hallucination check is at caller level)."""

        class HallucinatingLLM:
            def function_call(self, system, messages, tools, tool_choice="auto"):
                return FunctionCallResult(
                    content="已为您生成了图片",
                    tool_calls=[],
                    finish_reason="stop",
                )

        llm = HallucinatingLLM()
        result = _force_tool_call(
            llm,
            system_prompt="You are a domain expert.",
            task="生成一张图片",
            tools=[{"name": "agnes.generate_image", "description": "Generate image"}],
        )
        # _force_tool_call passes through LLM response; hallucination check
        # is done by the caller (delegate_node) which checks for content
        assert result.content == "已为您生成了图片"
        assert len(result.tool_calls) == 0


# ---- Tool Call Execution ----

class TestRunToolCalls:
    def test_executes_new_tool_calls(self):
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(
                name="test.echo",
                description="Echo tool",
                input_schema={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                handler=lambda msg="": {"echo": msg},
            )
        )
        tool_calls = [ToolCall(name="test.echo", arguments={"msg": "hello"})]
        results = []
        count = _run_tool_calls(tool_calls, results, reg, {})
        assert count == 1
        assert len(results) == 1
        assert results[0]["name"] == "test.echo"
        assert results[0]["result"]["echo"] == "hello"

    def test_deduplicates_same_tool_and_args(self):
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(
                name="test.echo",
                description="Echo",
                input_schema={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                handler=lambda msg="": {"echo": msg},
            )
        )
        # Pre-populate results with existing call
        existing = [{"name": "test.echo", "arguments": {"msg": "hello"}, "result": {"echo": "hello"}}]
        tool_calls = [ToolCall(name="test.echo", arguments={"msg": "hello"})]
        count = _run_tool_calls(tool_calls, existing, reg, {})
        assert count == 0  # Deduplicated
        assert len(existing) == 1  # No new result added

    def test_unknown_tool_skipped(self):
        reg = ToolRegistry()
        tool_calls = [ToolCall(name="nonexistent.tool", arguments={})]
        results = []
        count = _run_tool_calls(tool_calls, results, reg, {})
        assert count == 0
        assert len(results) == 0

    def test_tool_error_captured(self):
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(
                name="test.fail",
                description="Always fails",
                input_schema={"type": "object", "properties": {}, "required": []},
                handler=lambda **kw: (_ for _ in ()).throw(FinanceToolError("deliberate test failure")),
            )
        )
        tool_calls = [ToolCall(name="test.fail", arguments={})]
        results = []
        count = _run_tool_calls(tool_calls, results, reg, {})
        assert count == 1
        assert len(results) == 1
        assert "error" in results[0]
        assert "deliberate test failure" in results[0]["error"]

    def test_tool_arg_order_independent_dedup(self):
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(
                name="test.args",
                description="Args test",
                input_schema={"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]},
                handler=lambda a="", b="": {"a": a, "b": b},
            )
        )
        # First call
        results = []
        count1 = _run_tool_calls([ToolCall(name="test.args", arguments={"a": "x", "b": "y"})], results, reg, {})
        assert count1 == 1
        # Second call with same args different order
        count2 = _run_tool_calls([ToolCall(name="test.args", arguments={"b": "y", "a": "x"})], results, reg, {})
        assert count2 == 0  # Deduplicated despite different key order


# ---- Skill Matching ----

class TestSkillMatching:
    def test_negation_detection_prefix(self):
        """'没有异动' should NOT match '异动' in anomaly-diagnosis."""
        from matrix.skills.loader import SkillDefinition

        skill = SkillDefinition(
            name="anomaly-diagnosis",
            title="异动诊断",
            description="检测持仓异动并提供归因分析",
        )
        assert not skill.matches("今天没有异动")
        assert not skill.matches("不是异动")

    def test_skill_matches_positive(self):
        from matrix.skills.loader import SkillDefinition

        skill = SkillDefinition(
            name="anomaly-diagnosis",
            title="异动诊断",
            description="检测持仓异动并提供归因分析",
        )
        assert skill.matches("帮我诊断异动")
        assert skill.matches("持仓有异动吗")

    def test_negation_in_different_clause(self):
        """'没有异动，但帮我做组合复盘' — '没有' negates '异动', not '组合复盘'."""
        from matrix.skills.loader import SkillDefinition

        anomaly = SkillDefinition(
            name="anomaly-diagnosis",
            title="异动诊断",
            description="检测持仓异动并提供归因分析",
        )
        portfolio = SkillDefinition(
            name="portfolio-review",
            title="组合复盘",
            description="对投资组合进行定期复盘分析",
        )
        # "没有异动" should NOT match anomaly-diagnosis
        assert not anomaly.matches("没有异动，但帮我做组合复盘")
        # "组合复盘" should still match portfolio-review
        assert portfolio.matches("没有异动，但帮我做组合复盘")

    def test_bigram_fuzzy_match(self):
        from matrix.skills.loader import SkillDefinition

        skill = SkillDefinition(
            name="allocation-check",
            title="配置偏离检查",
            description="检查各bucket配置偏离度",
        )
        assert skill.matches("检查配置")
        assert skill.matches("偏离度")

    def test_short_query_match(self):
        from matrix.skills.loader import SkillDefinition

        skill = SkillDefinition(
            name="portfolio-review",
            title="组合复盘",
            description="对投资组合进行定期复盘分析",
        )
        assert skill.matches("复盘")


# ---- Build Tools for LLM ----

class TestBuildToolsForLLM:
    def test_builds_tool_list(self):
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(
                name="test.echo",
                description="Echo",
                input_schema={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
                handler=lambda msg="": {"echo": msg},
            )
        )
        reg.register(
            ToolDefinition(
                name="test.upper",
                description="Upper",
                input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                handler=lambda text="": {"upper": text.upper()},
            )
        )
        tools = _build_tools_for_llm(reg)
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        assert "test.echo" in names
        assert "test.upper" in names


# ---- Agent Registry ----

class TestAgentRegistry:
    def test_register_agents(self):
        reg = AgentRegistry(skills_base_dir="skills")
        reg.register_all([COMMANDER, INVESTMENT_ANALYST])
        assert reg.get("commander") is not None
        assert reg.get("investment-analyst") is not None

    def test_agents_for_commander(self):
        reg = AgentRegistry(skills_base_dir="skills")
        reg.register_all([COMMANDER, INVESTMENT_ANALYST])
        agents = reg.agents_for_commander()
        assert len(agents) >= 1
        assert isinstance(agents, list)

    def test_build_tool_registry_for_agent(self):
        from matrix.tools import ToolRegistry as TR

        tools = TR()
        tools.register(
            ToolDefinition(
                name="finance.holdings_summary",
                description="Get holdings",
                input_schema={"type": "object", "properties": {}, "required": []},
                handler=lambda **kw: {"count": 2},
            )
        )
        tools.register(
            ToolDefinition(
                name="web_search",
                description="Search",
                input_schema={"type": "object", "properties": {}, "required": []},
                handler=lambda **kw: {"results": []},
            )
        )
        reg = AgentRegistry(skills_base_dir="skills")
        reg.register_all([INVESTMENT_ANALYST])
        agent_tools = reg.build_tool_registry("investment-analyst", tools)
        assert len(agent_tools.tool_names()) > 0