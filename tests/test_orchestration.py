"""Tests for multi-agent LangGraph orchestration.

Commander + Domain Agents architecture:
  commander_plan → delegate → aggregate → reflection
"""

from __future__ import annotations

import json
import queue

import pytest

from matrix.orchestration.graph import (
    build_graph,
    _route_dag_first,
    _route_after_replan,
    _plan_index,
)
from matrix.orchestration.nodes import (
    _get_ready_steps,
    aggregate_node,
    commander_plan_node,
    delegate_node,
    reflection_node,
    replan_node,
    _run_domain_agent_react,
)
from matrix.orchestration.nodes._helpers import (
    CircuitBreaker,
    _focus_tools_for_task,
    _prune_tools,
    _requires_browser,
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

    def complete(self, system: str, messages: list, **kwargs) -> str:
        self.calls.append(("complete", messages))
        if not self.responses:
            return "{}"
        return self.responses.pop(0)

    def complete_json(self, system: str, messages: list, schema=None, **kwargs):
        import json
        self.calls.append(("complete_json", messages))
        if not self.responses:
            return {}
        text = self.responses.pop(0)
        try:
            return json.loads(text) if isinstance(text, str) else text
        except (json.JSONDecodeError, TypeError):
            return {}

    def stream_complete(self, system: str, messages: list, **kwargs):
        self.calls.append(("stream", messages))
        text = self.responses.pop(0) if self.responses else ""
        for ch in text:
            yield ch

    def function_call(self, system, messages, tools, tool_choice="auto", **kwargs):
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
            capabilities=["market_data", "portfolio_analysis"],
        )
    )
    reg.register(
        ToolDefinition(
            name="finance.bucket_allocation",
            description="Get bucket allocation",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=lambda **kw: {"buckets": [{"name": "stock", "target": 40, "current": 45}]},
            capabilities=["portfolio_analysis"],
        )
    )
    reg.register(
        ToolDefinition(
            name="web_search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            handler=lambda **kw: {"results": [{"title": "test", "url": "http://test.com"}]},
            capabilities=["web_search"],
        )
    )
    reg.register(
        ToolDefinition(
            name="web_fetch",
            description="Fetch a web page",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            handler=lambda **kw: {"text": "test content"},
            capabilities=["web_search"],
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
        defaults = {"user_message": "当前持仓情况如何？", "session_id": "test"}
        return AgentState(**(defaults | overrides))
    return _make


@pytest.fixture
def full_tools():
    return _build_registry()


@pytest.fixture
def agent_registry():
    return _build_agent_registry()


def make_config(llm, full_tools, agent_registry, trace=None, circuit_breaker=None):
    return {
        "configurable": {
            "llm": llm,
            "pipeline_llm": llm,  # use same llm for pipeline in tests
            "full_tools": full_tools,
            "agent_registry": agent_registry,
            "trace": trace,
            "circuit_breaker": circuit_breaker,
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

            def function_call(self, system, messages, tools, tool_choice="auto", **kwargs):
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
            def function_call(self, system, messages, tools, tool_choice="auto", **kwargs):
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
        # Graceful degradation: should produce a user-friendly fallback message,
        # NOT expose internal details like agent names or error codes
        answer = result.get("final_answer", "")
        assert len(answer) > 10
        assert "所有领域专家执行失败" not in answer  # no internal details leaked
        assert "建议" in answer or "稍后" in answer or "重试" in answer

    def test_aggregate_all_errors_with_llm_fallback(self, base_state, full_tools, agent_registry):
        """LLM generates a friendly fallback message when all agents fail."""
        llm = FakeLLM([
            "很抱歉，我暂时无法获取您的持仓数据。建议您检查网络连接后重新提问，或尝试询问其他问题。",
        ])
        state = base_state(
            agent_results=[
                {"agent_id": "agent1", "task": "获取持仓", "result": "", "error": "数据源不可用"},
            ],
        )
        result = aggregate_node(state, config=make_config(llm, full_tools, agent_registry))
        answer = result.get("final_answer", "")
        assert len(answer) > 10
        assert "抱歉" in answer or "暂时无法" in answer
        assert "agent" not in answer.lower()  # no internal terms leaked
        assert "数据源" not in answer  # no internal error details leaked

    def test_aggregate_all_errors_fallback_llm_fails(self, base_state, full_tools, agent_registry):
        """When fallback LLM also fails, hard-coded message is used."""
        # Empty responses → LLM fallback fails → hard-coded message
        llm = FakeLLM([])
        state = base_state(
            agent_results=[
                {"agent_id": "agent1", "task": "获取数据", "result": "", "error": "执行失败"},
                {"agent_id": "agent2", "task": "分析数据", "result": "", "error": "工具不可用"},
            ],
        )
        result = aggregate_node(state, config=make_config(llm, full_tools, agent_registry))
        answer = result.get("final_answer", "")
        assert len(answer) > 10
        assert "抱歉" in answer
        assert "建议" in answer

    def test_aggregate_all_errors_skips_reflection(self, base_state, full_tools, agent_registry):
        """Fallback messages should skip reflection to avoid unnecessary retries."""
        llm = FakeLLM([])
        state = base_state(
            agent_results=[
                {"agent_id": "agent1", "task": "测试", "result": "", "error": "失败"},
            ],
        )
        result = aggregate_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result.get("skip_reflection") is True

    def test_aggregate_partial_errors_uses_normal_path(self, base_state, full_tools, agent_registry):
        """When some agents succeed, normal aggregation should be used (not fallback)."""
        llm = FakeLLM(["当前持仓健康，共2个持仓，总价值100,000元。"])
        state = base_state(
            agent_results=[
                {"agent_id": "agent1", "task": "获取持仓", "result": "持仓数据正常", "error": "", "tool_results": []},
                {"agent_id": "agent2", "task": "分析风险", "result": "", "error": "风险分析失败", "tool_results": []},
            ],
        )
        result = aggregate_node(state, config=make_config(llm, full_tools, agent_registry))
        answer = result.get("final_answer", "")
        # Should use normal aggregation, not fallback
        assert "抱歉" not in answer or "暂时" not in answer
        assert "持仓" in answer


# ---- Circuit Breaker Integration ----

class TestCircuitBreakerIntegration:
    def test_prune_tools_hides_blocked_tools(self):
        """_prune_tools should hide tools that are blocked by circuit breaker."""
        cb = CircuitBreaker()
        # Trip web_search
        for _ in range(3):
            cb.record_failure("web_search")
        assert cb.is_blocked("web_search")

        tools = [
            {"function": {"name": "web_search", "description": "Search the web"}},
            {"function": {"name": "web_fetch", "description": "Fetch a page"}},
            {"function": {"name": "news_search", "description": "Search news"}},
        ]
        pruned = _prune_tools(tools, [{"role": "user", "content": "test"}], circuit_breaker=cb)
        pruned_names = [t["function"]["name"] for t in pruned]
        assert "web_search" not in pruned_names
        assert "web_fetch" in pruned_names
        assert "news_search" in pruned_names

    def test_prune_tools_allows_when_breaker_cleared(self):
        """_prune_tools should show tools when breaker is reset."""
        cb = CircuitBreaker()
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_success("web_search")  # reset

        tools = [
            {"function": {"name": "web_search", "description": "Search the web"}},
        ]
        pruned = _prune_tools(tools, [{"role": "user", "content": "test"}], circuit_breaker=cb)
        pruned_names = [t["function"]["name"] for t in pruned]
        assert "web_search" in pruned_names

    def test_prune_tools_without_breaker_passes_all(self):
        """Without circuit_breaker, all tools should pass through."""
        tools = [
            {"function": {"name": "web_search", "description": "Search the web"}},
            {"function": {"name": "web_fetch", "description": "Fetch a page"}},
        ]
        pruned = _prune_tools(tools, [{"role": "user", "content": "test"}])
        pruned_names = [t["function"]["name"] for t in pruned]
        assert "web_search" in pruned_names
        assert "web_fetch" in pruned_names

    def test_breaker_integrated_in_aggregate(self, base_state, full_tools, agent_registry):
        """Circuit breaker is accepted in config but doesn't affect aggregate logic."""
        cb = CircuitBreaker()
        llm = FakeLLM(["当前持仓健康，共2个持仓。"])
        state = base_state(
            agent_results=[
                {"agent_id": "agent1", "task": "获取持仓", "result": "持仓数据", "error": "", "tool_results": []},
            ],
        )
        result = aggregate_node(
            state,
            config=make_config(llm, full_tools, agent_registry, circuit_breaker=cb),
        )
        assert result.get("needs_summary") is False
        assert "持仓" in result.get("final_answer", "")


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
            reflexion_max=0,  # disable Reflexion retry, test revision path
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


# ── P3: Tool Capability Declaration ──────────────────────────────────────────

class TestToolCapabilities:
    """Tests for P3: Tool capability declaration and aggregation."""

    def test_tool_definition_with_capabilities(self):
        """ToolDefinition should store capabilities."""
        tool = ToolDefinition(
            name="test.tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: {},
            capabilities=["market_data", "web_search"],
        )
        assert tool.capabilities == ["market_data", "web_search"]

    def test_tool_definition_default_capabilities(self):
        """ToolDefinition should default to empty capabilities list."""
        tool = ToolDefinition(
            name="test.tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: {},
        )
        assert tool.capabilities == []

    def test_tool_definition_to_dict_excludes_capabilities(self):
        """to_dict() should not include capabilities (LLM planners don't need them)."""
        tool = ToolDefinition(
            name="test.tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda **kw: {},
            capabilities=["market_data"],
        )
        d = tool.to_dict()
        assert "capabilities" not in d
        assert d["name"] == "test.tool"

    def test_registry_get_capabilities_summary(self):
        """get_capabilities_summary should group tools by capability tag."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="finance.holdings", description="Get holdings",
            input_schema={}, handler=lambda **kw: {},
            capabilities=["market_data", "portfolio_analysis"],
        ))
        reg.register(ToolDefinition(
            name="web_search", description="Search web",
            input_schema={}, handler=lambda **kw: {},
            capabilities=["web_search"],
        ))
        reg.register(ToolDefinition(
            name="finance.quotes", description="Get quotes",
            input_schema={}, handler=lambda **kw: {},
            capabilities=["market_data"],
        ))

        summary = reg.get_capabilities_summary()
        assert "market_data" in summary
        assert set(summary["market_data"]) == {"finance.holdings", "finance.quotes"}
        assert "web_search" in summary
        assert summary["web_search"] == ["web_search"]
        assert "portfolio_analysis" in summary
        assert summary["portfolio_analysis"] == ["finance.holdings"]

    def test_registry_get_tool_capabilities(self):
        """get_tool_capabilities should return per-tool capability mapping."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="tool_a", description="A",
            input_schema={}, handler=lambda **kw: {},
            capabilities=["cap1", "cap2"],
        ))
        reg.register(ToolDefinition(
            name="tool_b", description="B",
            input_schema={}, handler=lambda **kw: {},
        ))

        caps = reg.get_tool_capabilities()
        assert caps["tool_a"] == ["cap1", "cap2"]
        assert caps["tool_b"] == []

    def test_capabilities_summary_empty(self):
        """Empty registry should return empty capability summary."""
        reg = ToolRegistry()
        assert reg.get_capabilities_summary() == {}
        assert reg.get_tool_capabilities() == {}

    def test_agents_for_commander_with_capabilities(self, full_tools):
        """agents_for_commander should include capabilities when full_tools is provided."""
        agent_reg = _build_agent_registry()
        agents = agent_reg.agents_for_commander(full_tools)
        # INVESTMENT_ANALYST has access to all tools → should have capabilities
        inv_agent = next((a for a in agents if a["id"] == "investment-analyst"), None)
        assert inv_agent is not None
        # The test tool registry has capabilities on tools, so agent should show them
        assert "capabilities" in inv_agent
        caps = inv_agent["capabilities"]
        assert "market_data" in caps
        assert "portfolio_analysis" in caps
        assert "web_search" in caps

    def test_agents_for_commander_without_tools(self, agent_registry):
        """agents_for_commander without full_tools should not include capabilities."""
        agents = agent_registry.agents_for_commander()
        inv_agent = next((a for a in agents if a["id"] == "investment-analyst"), None)
        assert inv_agent is not None
        assert "capabilities" not in inv_agent


# ── P2: Execution Progress Monitoring ────────────────────────────────────────

def _make_config_with_events(llm, full_tools, agent_registry):
    """Make config with event_queue for capturing progress events."""
    event_queue = queue.Queue()
    config = {
        "configurable": {
            "llm": llm,
            "pipeline_llm": llm,
            "full_tools": full_tools,
            "agent_registry": agent_registry,
            "event_queue": event_queue,
        },
    }
    return config, event_queue


def _drain_progress_events(event_queue: queue.Queue) -> list[dict]:
    """Drain all events from queue and return progress events."""
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    return [e[1] for e in events if e[0] == "progress"]


def test_browser_task_without_mcp_is_explicitly_blocked(full_tools):
    result = _run_domain_agent_react(
        agent_def=INVESTMENT_ANALYST,
        task="用浏览器打开一个 SPA 页面并提取内容",
        tools=full_tools,
        skill_results=[],
        cfg={"llm": object()},
        agent_id="investment-analyst",
    )

    assert _requires_browser("用浏览器打开一个 SPA 页面并提取内容") is True
    assert result["environment_blocked"] is True
    assert "浏览器 MCP" in result["answer"]
    assert result["tool_results"] == []


def test_explicit_tasks_focus_the_action_space():
    tools = [
        {"type": "function", "function": {"name": "weather"}},
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "finance.recent_snapshots"}},
        {"type": "function", "function": {"name": "code.run_python"}},
        {"type": "function", "function": {"name": "mcp_browser_navigate"}},
        {"type": "function", "function": {"name": "mcp_browser_extract"}},
    ]

    assert [
        t["function"]["name"]
        for t in _focus_tools_for_task("今天北京天气怎么样", tools)
    ] == ["weather"]
    assert [
        t["function"]["name"]
        for t in _focus_tools_for_task("最近5条快照记录", tools)
    ] == ["finance.recent_snapshots"]
    assert [
        t["function"]["name"]
        for t in _focus_tools_for_task("打开 SPA 页面并提取内容", tools)
    ] == ["mcp_browser_navigate", "mcp_browser_extract"]


class TestProgressEvents:
    """Tests for P2: Execution progress monitoring events."""

    def test_commander_emits_plan_created_for_multi_step(self, base_state, full_tools, agent_registry):
        """Multi-step plan should emit plan_created progress event."""
        plan_json = json.dumps([
            {"step": 1, "agent_id": "investment-analyst", "task": "获取数据",
             "depends_on": [], "skill_name": "", "purpose": "获取"},
            {"step": 2, "agent_id": "investment-analyst", "task": "分析数据",
             "depends_on": [1], "skill_name": "", "purpose": "分析"},
        ])
        llm = FakeLLM([plan_json])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        result = commander_plan_node(base_state(), config=config)
        assert len(result["delegation_plan"]) == 2

        progress = _drain_progress_events(eq)
        plan_created = [e for e in progress if e.get("type") == "plan_created"]
        assert len(plan_created) == 1
        assert plan_created[0]["total_steps"] == 2
        assert plan_created[0]["plan_type"] in ("agent", "subtask")
        assert len(plan_created[0]["steps"]) == 2

    def test_commander_no_progress_for_single_step(self, base_state, full_tools, agent_registry):
        """Single-step plan should NOT emit plan_created progress event."""
        plan_json = json.dumps([
            {"step": 1, "agent_id": "investment-analyst", "task": "分析",
             "skill_name": "", "purpose": "分析"},
        ])
        llm = FakeLLM([plan_json])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        result = commander_plan_node(base_state(), config=config)
        assert len(result["delegation_plan"]) == 1

        progress = _drain_progress_events(eq)
        plan_created = [e for e in progress if e.get("type") == "plan_created"]
        assert len(plan_created) == 0

    def test_commander_no_progress_for_empty_plan(self, base_state, full_tools, agent_registry):
        """Empty plan (Commander fallback) should NOT emit plan_created."""
        llm = FakeLLM(["[]"])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        result = commander_plan_node(base_state(), config=config)
        assert len(result["delegation_plan"]) == 1  # fallback to commander

        progress = _drain_progress_events(eq)
        plan_created = [e for e in progress if e.get("type") == "plan_created"]
        assert len(plan_created) == 0

    def test_delegate_emits_step_start_and_done(self, base_state, full_tools, agent_registry):
        """Multi-step plan delegate should emit step_start and step_done."""
        llm = FakeLLM(["分析完成：持仓健康。"])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        state = base_state(
            delegation_plan=[
                {"step": 1, "agent_id": "investment-analyst", "task": "获取数据",
                 "depends_on": [], "skill_name": "", "purpose": "获取"},
                {"step": 2, "agent_id": "investment-analyst", "task": "分析数据",
                 "depends_on": [1], "skill_name": "", "purpose": "分析"},
            ],
            current_step=0,
        )
        result = delegate_node(state, config=config)
        assert len(result["agent_results"]) == 1

        progress = _drain_progress_events(eq)
        step_start = [e for e in progress if e.get("type") == "step_start"]
        step_done = [e for e in progress if e.get("type") == "step_done"]

        assert len(step_start) >= 1
        assert step_start[0]["step"] == 1
        assert step_start[0]["total"] == 2
        assert len(step_done) >= 1
        assert step_done[0]["step"] == 1

    def test_delegate_no_progress_for_single_step(self, base_state, full_tools, agent_registry):
        """Single-step plan should NOT emit step_start/step_done events."""
        llm = FakeLLM(["分析完成。"])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        state = base_state(
            delegation_plan=[
                {"step": 1, "agent_id": "investment-analyst", "task": "分析",
                 "skill_name": "", "purpose": "分析"},
            ],
            current_step=0,
        )
        result = delegate_node(state, config=config)
        assert len(result["agent_results"]) == 1

        progress = _drain_progress_events(eq)
        step_events = [e for e in progress if e.get("type") in ("step_start", "step_done", "step_error")]
        assert len(step_events) == 0

    def test_delegate_emits_step_error_for_failed_agent(self, base_state, full_tools, agent_registry):
        """Failed agent (not found) should emit step_error event."""
        llm = FakeLLM([])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        state = base_state(
            delegation_plan=[
                {"step": 1, "agent_id": "nonexistent-agent", "task": "测试",
                 "depends_on": [], "skill_name": "", "purpose": "测试"},
                {"step": 2, "agent_id": "investment-analyst", "task": "分析",
                 "depends_on": [1], "skill_name": "", "purpose": "分析"},
            ],
            current_step=0,
        )
        result = delegate_node(state, config=config)
        assert "error" in result["agent_results"][0]

        progress = _drain_progress_events(eq)
        step_error = [e for e in progress if e.get("type") == "step_error"]
        assert len(step_error) >= 1
        assert "error" in step_error[0]

    def test_delegate_step_done_includes_result_preview(self, base_state, full_tools, agent_registry):
        """step_done event should include result_preview."""
        llm = FakeLLM(["分析完成：持仓配置合理，建议继续持有。"])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        state = base_state(
            delegation_plan=[
                {"step": 1, "agent_id": "investment-analyst", "task": "获取数据",
                 "depends_on": [], "skill_name": "", "purpose": "获取"},
                {"step": 2, "agent_id": "investment-analyst", "task": "分析",
                 "depends_on": [1], "skill_name": "", "purpose": "分析"},
            ],
            current_step=0,
        )
        delegate_node(state, config=config)

        progress = _drain_progress_events(eq)
        step_done = [e for e in progress if e.get("type") == "step_done"]
        assert len(step_done) >= 1
        assert "result_preview" in step_done[0]

    def test_replan_emits_progress_event(self, base_state, full_tools, agent_registry):
        """Replan should emit progress event when revision is needed."""
        revised_plan = [
            {"step": 1, "depends_on": [], "task": "task1", "output_key": "step_1"},
            {"step": 2, "depends_on": [1], "task": "revised task2", "output_key": "step_2"},
        ]
        llm = FakeLLM([
            json.dumps({"needs_revision": True, "reason": "数据缺失需调整", "revised_plan": revised_plan}),
        ])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        state = base_state(
            delegation_plan=[
                {"step": 1, "depends_on": [], "task": "task1", "output_key": "step_1"},
                {"step": 2, "depends_on": [1], "task": "original task2", "output_key": "step_2"},
            ],
            completed_steps=[1],
            agent_results=[{"task": "task1", "result": "empty", "error": ""}],
        )
        result = replan_node(state, config=config)
        assert result["needs_replan"] is True

        progress = _drain_progress_events(eq)
        replan_events = [e for e in progress if e.get("type") == "replan"]
        assert len(replan_events) == 1
        assert replan_events[0]["reason"] == "数据缺失需调整"
        assert replan_events[0]["attempt"] == 1

    def test_replan_no_progress_when_no_revision(self, base_state, full_tools, agent_registry):
        """Replan should NOT emit progress event when no revision needed."""
        llm = FakeLLM([
            '{"needs_revision": false, "reason": "", "revised_plan": []}',
        ])
        config, eq = _make_config_with_events(llm, full_tools, agent_registry)

        state = base_state(
            delegation_plan=[{"step": 1, "depends_on": [], "task": "task1"}],
            completed_steps=[1],
            agent_results=[{"task": "task1", "result": "success", "error": ""}],
        )
        result = replan_node(state, config=config)
        assert result["needs_replan"] is False

        progress = _drain_progress_events(eq)
        replan_events = [e for e in progress if e.get("type") == "replan"]
        assert len(replan_events) == 0

    def test_no_progress_events_without_event_queue(self, base_state, full_tools, agent_registry):
        """Progress events should be silently dropped when no event_queue in config."""
        plan_json = json.dumps([
            {"step": 1, "agent_id": "investment-analyst", "task": "获取数据",
             "depends_on": [], "skill_name": "", "purpose": "获取"},
            {"step": 2, "agent_id": "investment-analyst", "task": "分析数据",
             "depends_on": [1], "skill_name": "", "purpose": "分析"},
        ])
        llm = FakeLLM([plan_json])
        # No event_queue in config — should not crash
        config = make_config(llm, full_tools, agent_registry)
        result = commander_plan_node(base_state(), config=config)
        assert len(result["delegation_plan"]) == 2


# ── Plan-and-Execute: DAG Resolution ─────────────────────────────────────────

class TestGetReadySteps:
    """Unit tests for _get_ready_steps — DAG dependency resolution."""

    def test_no_deps_all_ready(self):
        """All steps have no dependencies → all should be ready."""
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [], "task": "task2"},
            {"step": 3, "depends_on": [], "task": "task3"},
        ]
        ready = _get_ready_steps(plan, [])
        assert len(ready) == 3
        assert {s["step"] for s in ready} == {1, 2, 3}

    def test_chain_dependency_step_by_step(self):
        """Chain: 1→2→3. Only step 1 ready initially."""
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [1], "task": "task2"},
            {"step": 3, "depends_on": [2], "task": "task3"},
        ]
        ready = _get_ready_steps(plan, [])
        assert len(ready) == 1
        assert ready[0]["step"] == 1

        ready = _get_ready_steps(plan, [1])
        assert len(ready) == 1
        assert ready[0]["step"] == 2

        ready = _get_ready_steps(plan, [1, 2])
        assert len(ready) == 1
        assert ready[0]["step"] == 3

    def test_parallel_branches_then_merge(self):
        """1,2 parallel → 3 depends on both. 1,2 both ready initially."""
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [], "task": "task2"},
            {"step": 3, "depends_on": [1, 2], "task": "task3"},
        ]
        # Initially: 1,2 ready (parallel)
        ready = _get_ready_steps(plan, [])
        assert len(ready) == 2
        assert {s["step"] for s in ready} == {1, 2}

        # After step 1 done: 2 still ready, 3 still blocked
        ready = _get_ready_steps(plan, [1])
        assert len(ready) == 1
        assert ready[0]["step"] == 2

        # After both 1,2 done: 3 ready
        ready = _get_ready_steps(plan, [1, 2])
        assert len(ready) == 1
        assert ready[0]["step"] == 3

    def test_all_completed_none_ready(self):
        """All steps completed → no ready steps."""
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [1], "task": "task2"},
        ]
        ready = _get_ready_steps(plan, [1, 2])
        assert len(ready) == 0

    def test_missing_depends_on_field_defaults_to_empty(self):
        """Steps without depends_on field should be treated as no dependencies."""
        plan = [
            {"step": 1, "task": "task1"},  # No depends_on key
            {"step": 2, "depends_on": [1], "task": "task2"},
        ]
        ready = _get_ready_steps(plan, [])
        assert len(ready) == 1
        assert ready[0]["step"] == 1

    def test_diamond_dependency(self):
        """Diamond: 1→2, 1→3, 2,3→4."""
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [1], "task": "task2"},
            {"step": 3, "depends_on": [1], "task": "task3"},
            {"step": 4, "depends_on": [2, 3], "task": "task4"},
        ]
        ready = _get_ready_steps(plan, [])
        assert len(ready) == 1
        assert ready[0]["step"] == 1

        ready = _get_ready_steps(plan, [1])
        assert len(ready) == 2
        assert {s["step"] for s in ready} == {2, 3}

        ready = _get_ready_steps(plan, [1, 2])
        assert len(ready) == 1
        assert ready[0]["step"] == 3

        ready = _get_ready_steps(plan, [1, 2, 3])
        assert len(ready) == 1
        assert ready[0]["step"] == 4

    def test_plan_index_conversion(self):
        """_plan_index converts 1-based step to 0-based index."""
        plan = [
            {"step": 1, "task": "task1"},
            {"step": 2, "task": "task2"},
            {"step": 5, "task": "task5"},  # non-sequential step numbers
        ]
        assert _plan_index(plan, 1) == 0
        assert _plan_index(plan, 2) == 1
        assert _plan_index(plan, 5) == 2
        assert _plan_index(plan, 99) == 0  # not found → default 0


# ── Plan-and-Execute: Replan Node ────────────────────────────────────────────

class TestReplanNode:
    """Tests for replan_node — dynamic plan revision."""

    def test_replan_skips_when_no_plan(self, base_state, full_tools, agent_registry):
        """No plan → no replan needed."""
        llm = FakeLLM([])
        state = base_state(delegation_plan=[], completed_steps=[])
        result = replan_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result["needs_replan"] is False

    def test_replan_skips_when_no_completed_steps(self, base_state, full_tools, agent_registry):
        """Plan exists but no completed steps → skip."""
        llm = FakeLLM([])
        state = base_state(
            delegation_plan=[{"step": 1, "depends_on": [], "task": "task1"}],
            completed_steps=[],
        )
        result = replan_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result["needs_replan"] is False

    def test_replan_no_revision_needed(self, base_state, full_tools, agent_registry):
        """LLM says plan is fine → no replan."""
        llm = FakeLLM(['{"needs_revision": false, "reason": "", "revised_plan": []}'])
        state = base_state(
            delegation_plan=[
                {"step": 1, "depends_on": [], "task": "task1", "output_key": "step_1"},
            ],
            completed_steps=[1],
            agent_results=[
                {"task": "task1", "result": "success", "error": ""},
            ],
        )
        result = replan_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result["needs_replan"] is False

    def test_replan_triggers_revision(self, base_state, full_tools, agent_registry):
        """LLM detects plan needs revision → replan triggered."""
        revised_plan = [
            {"step": 1, "depends_on": [], "task": "task1", "output_key": "step_1"},
            {"step": 2, "depends_on": [1], "task": "revised task2", "output_key": "step_2"},
        ]
        llm = FakeLLM([
            json.dumps({"needs_revision": True, "reason": "步2数据缺失需调整", "revised_plan": revised_plan}),
        ])
        state = base_state(
            delegation_plan=[
                {"step": 1, "depends_on": [], "task": "task1", "output_key": "step_1"},
                {"step": 2, "depends_on": [1], "task": "original task2", "output_key": "step_2"},
            ],
            completed_steps=[1],
            agent_results=[
                {"task": "task1", "result": "empty result", "error": ""},
            ],
        )
        result = replan_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result["needs_replan"] is True
        assert result["replan_attempts"] == 1
        assert len(result["delegation_plan"]) == 2
        assert result["delegation_plan"][1]["task"] == "revised task2"

    def test_replan_max_attempts_exceeded(self, base_state, full_tools, agent_registry):
        """Max replan attempts reached → force continue."""
        llm = FakeLLM(['{"needs_revision": true, "reason": "still bad", "revised_plan": []}'])
        state = base_state(
            delegation_plan=[{"step": 1, "depends_on": [], "task": "task1"}],
            completed_steps=[1],
            agent_results=[{"task": "task1", "result": "x", "error": ""}],
            replan_attempts=2,  # Already at max
        )
        result = replan_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result["needs_replan"] is False

    def test_replan_llm_error_graceful_fallback(self, base_state, full_tools, agent_registry):
        """LLM error during replan → graceful fallback (no replan)."""
        llm = FakeLLM(["not valid json {{{{"])
        state = base_state(
            delegation_plan=[{"step": 1, "depends_on": [], "task": "task1"}],
            completed_steps=[1],
            agent_results=[{"task": "task1", "result": "x", "error": ""}],
        )
        result = replan_node(state, config=make_config(llm, full_tools, agent_registry))
        assert result["needs_replan"] is False


# ── Plan-and-Execute: DAG Routing ────────────────────────────────────────────

class TestDAGRouting:
    """Tests for DAG routing functions."""

    def test_route_dag_first_single_step(self, base_state):
        """Single-step plan → react_prepare (backward compatible)."""
        plan = [{"step": 1, "depends_on": [], "task": "task1"}]
        result = _route_dag_first(base_state(delegation_plan=plan))
        assert result == "react_prepare"

    def test_route_dag_first_empty_plan(self, base_state):
        """Empty plan → react_prepare."""
        result = _route_dag_first(base_state(delegation_plan=[]))
        assert result == "react_prepare"

    def test_route_dag_first_parallel_no_deps(self, base_state):
        """Multi-step plan with no dependencies → all fan out in parallel."""
        from langgraph.types import Send
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [], "task": "task2"},
        ]
        result = _route_dag_first(base_state(delegation_plan=plan, completed_steps=[]))
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)
        assert all(s.node == "delegate" for s in result)

    def test_route_dag_first_chain_only_first_ready(self, base_state):
        """Chain dependency → only first step fans out."""
        from langgraph.types import Send
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [1], "task": "task2"},
            {"step": 3, "depends_on": [2], "task": "task3"},
        ]
        result = _route_dag_first(base_state(delegation_plan=plan, completed_steps=[]))
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].arg["current_step"] == 0

    def test_route_dag_first_no_ready_steps(self, base_state):
        """All steps blocked → aggregate (shouldn't happen normally)."""
        plan = [
            {"step": 1, "depends_on": [999], "task": "task1"},
            {"step": 2, "depends_on": [1], "task": "task2"},
        ]
        result = _route_dag_first(base_state(delegation_plan=plan, completed_steps=[]))
        assert result == "aggregate"

    def test_route_after_replan_triggers_replan(self, base_state):
        """needs_replan=True → commander_plan."""
        state = base_state(needs_replan=True)
        result = _route_after_replan(state)
        assert result == "commander_plan"

    def test_route_after_replan_continue_next_batch(self, base_state):
        """More ready steps → next batch of delegates."""
        from langgraph.types import Send
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [1], "task": "task2"},
        ]
        result = _route_after_replan(
            base_state(delegation_plan=plan, completed_steps=[1], needs_replan=False)
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].node == "delegate"

    def test_route_after_replan_all_done(self, base_state):
        """All steps completed → aggregate."""
        plan = [
            {"step": 1, "depends_on": [], "task": "task1"},
            {"step": 2, "depends_on": [1], "task": "task2"},
        ]
        result = _route_after_replan(
            base_state(delegation_plan=plan, completed_steps=[1, 2], needs_replan=False)
        )
        assert result == "aggregate"


# ── Plan-and-Execute: Full Graph Integration ─────────────────────────────────

class TestPlanAndExecuteGraph:
    """Integration tests for the full Plan-and-Execute graph flow."""

    def test_graph_has_all_nodes(self):
        """Verify all 10 nodes are present in the compiled graph."""
        graph = build_graph()
        expected = {
            "commander_plan", "react_prepare", "react_llm", "react_tool",
            "react_evaluate", "delegate", "confirm", "aggregate",
            "reflection", "replan_node",
        }
        actual = set(graph.nodes.keys())
        assert actual == expected

    def test_dag_chain_execution(self, base_state, full_tools, agent_registry):
        """Chain: Step 1→2→3. Verify sequential execution through DAG routing."""
        llm = FakeLLM([
            # commander_plan: chain plan
            json.dumps([
                {"step": 1, "agent_id": "investment-analyst", "task": "获取市场数据",
                 "depends_on": [], "output_key": "market_data", "skill_name": "", "purpose": "获取"},
                {"step": 2, "agent_id": "investment-analyst", "task": "分析持仓",
                 "depends_on": [1], "output_key": "analysis", "skill_name": "", "purpose": "分析"},
                {"step": 3, "agent_id": "investment-analyst", "task": "汇总建议",
                 "depends_on": [2], "output_key": "summary", "skill_name": "", "purpose": "汇总"},
            ]),
            # PreFlect: approve plan (no revision needed)
            '{"needs_revision": false, "issues": [], "adjusted_plan": []}',
            # delegate Step 1
            "市场数据：上证指数3300点。",
            # replan_node: batch 1 check
            '{"needs_revision": false, "reason": "", "revised_plan": []}',
            # delegate Step 2
            "持仓分析：配置健康。",
            # replan_node: batch 2 check
            '{"needs_revision": false, "reason": "", "revised_plan": []}',
            # delegate Step 3
            "汇总：市场平稳，持仓健康。",
            # replan_node: batch 3 check → all done → aggregate
            '{"needs_revision": false, "reason": "", "revised_plan": []}',
            # aggregate
            "最终汇总：市场平稳，持仓健康，建议继续持有。",
            # reflection
            '{"ok": true}',
        ])
        graph = build_graph()
        compiled = graph.compile()
        events = list(
            compiled.stream(
                base_state(user_message="全面分析我的投资组合"),
                stream_mode="values",
                config=make_config(llm, full_tools, agent_registry),
                thread_id="test-dag-chain",
            )
        )
        final = events[-1]
        assert len(final.get("agent_results", [])) == 3
        assert len(final.get("completed_steps", [])) == 3
        assert set(final.get("completed_steps", [])) == {1, 2, 3}
        assert "持有" in final.get("final_answer", "")

    def test_dag_parallel_execution(self, base_state, full_tools, agent_registry):
        """Parallel: 1,2 independent → 3 depends on both. Verify parallel fan-out."""
        llm = FakeLLM([
            # commander_plan: parallel plan
            json.dumps([
                {"step": 1, "agent_id": "investment-analyst", "task": "获取A股数据",
                 "depends_on": [], "output_key": "a_shares", "skill_name": "", "purpose": "获取"},
                {"step": 2, "agent_id": "investment-analyst", "task": "获取港股数据",
                 "depends_on": [], "output_key": "hk_shares", "skill_name": "", "purpose": "获取"},
                {"step": 3, "agent_id": "investment-analyst", "task": "对比分析",
                 "depends_on": [1, 2], "output_key": "comparison", "skill_name": "", "purpose": "对比"},
            ]),
            # PreFlect: approve plan (no revision needed)
            '{"needs_revision": false, "issues": [], "adjusted_plan": []}',
            # delegate Step 1 and Step 2 run in parallel
            "A股数据：沪深300上涨。",
            "港股数据：恒生指数持平。",
            # replan_node: batch 1 check
            '{"needs_revision": false, "reason": "", "revised_plan": []}',
            # delegate Step 3
            "对比：A股强于港股。",
            # replan_node: batch 2 check
            '{"needs_revision": false, "reason": "", "revised_plan": []}',
            # aggregate
            "最终：A股表现优于港股，建议关注A股机会。",
            # reflection
            '{"ok": true}',
        ])
        graph = build_graph()
        compiled = graph.compile()
        events = list(
            compiled.stream(
                base_state(user_message="对比A股和港股"),
                stream_mode="values",
                config=make_config(llm, full_tools, agent_registry),
                thread_id="test-dag-parallel",
            )
        )
        final = events[-1]
        # Verify all 3 steps completed
        assert len(final.get("agent_results", [])) == 3
        assert set(final.get("completed_steps", [])) == {1, 2, 3}

    def test_replan_revision_in_chain(self, base_state, full_tools, agent_registry):
        """Step 1 returns empty → replan triggers revision → new plan executed."""
        original_plan = [
            {"step": 1, "agent_id": "investment-analyst", "task": "获取数据",
             "depends_on": [], "output_key": "data", "skill_name": "", "purpose": "获取"},
            {"step": 2, "agent_id": "investment-analyst", "task": "分析数据",
             "depends_on": [1], "output_key": "analysis", "skill_name": "", "purpose": "分析"},
        ]
        revised_plan = [
            {"step": 1, "agent_id": "investment-analyst", "task": "获取数据（已重试）",
             "depends_on": [], "output_key": "data", "skill_name": "", "purpose": "获取"},
            {"step": 2, "agent_id": "investment-analyst", "task": "改用其他方式分析",
             "depends_on": [1], "output_key": "analysis", "skill_name": "", "purpose": "分析"},
        ]
        llm = FakeLLM([
            # commander_plan: chain plan
            json.dumps(original_plan),
            # delegate Step 1
            "数据获取失败，返回空。",
            # replan_node: detects problem → revision
            json.dumps({"needs_revision": True, "reason": "数据为空", "revised_plan": revised_plan}),
            # commander_plan (replan) → same plan returned
            json.dumps(revised_plan),
            # delegate Step 1 (retry)
            "重新获取成功：上证指数3300。",
            # replan_node: batch 1 check
            '{"needs_revision": false, "reason": "", "revised_plan": []}',
            # delegate Step 2
            "分析完成：市场平稳。",
            # replan_node: all done
            '{"needs_revision": false, "reason": "", "revised_plan": []}',
            # aggregate
            "最终：市场平稳。",
            # reflection
            '{"ok": true}',
        ])
        graph = build_graph()
        compiled = graph.compile()
        events = list(
            compiled.stream(
                base_state(user_message="分析市场"),
                stream_mode="values",
                config=make_config(llm, full_tools, agent_registry),
                thread_id="test-dag-replan",
            )
        )
        final = events[-1]
        assert "市场" in final.get("final_answer", "")

    def test_single_step_backward_compatible(self, base_state, full_tools, agent_registry):
        """Single-step plan still goes through react_prepare (backward compatible)."""
        llm = FakeLLM([
            json.dumps([  # commander_plan: single step
                {"step": 1, "agent_id": "investment-analyst", "task": "分析持仓",
                 "depends_on": [], "output_key": "step_1", "skill_name": "", "purpose": "获取"},
            ]),
            # delegate → react_prepare path
            "当前持仓健康。",
            "汇总：当前持仓健康。",  # aggregate
            '{"ok": true}',  # reflection
        ])
        graph = build_graph()
        compiled = graph.compile()
        events = list(
            compiled.stream(
                base_state(user_message="分析持仓"),
                stream_mode="values",
                config=make_config(llm, full_tools, agent_registry),
                thread_id="test-dag-compat",
            )
        )
        final = events[-1]
        assert len(final.get("agent_results", [])) >= 1
        assert final.get("final_answer") is not None
