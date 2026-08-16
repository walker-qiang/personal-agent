"""Tests for the Agent-as-Tool module (hierarchical agent delegation)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from matrix.tools.agent_tool import (
    AgentToolWrapper,
    CfgFactory,
    _call_depth,
    _MAX_DEPTH,
    _has_agent_tool_dependency,
    make_agent_tool,
    register_agent_tools,
)
from matrix.tools.base import ToolDefinition
from matrix.agent.base import AgentDefinition


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def agent_def():
    """A simple domain agent for testing."""
    return AgentDefinition(
        id="test-agent",
        name="测试Agent",
        description="用于测试的领域Agent",
        domain="test",
        persona="你是一个测试Agent.",
        tools=["finance.*", "web_search"],
    )


@pytest.fixture
def commander_def():
    """Commander agent (should be excluded from agent-tool registration)."""
    return AgentDefinition(
        id="commander",
        name="Commander",
        description="协调器",
        domain="commander",
        persona="你是协调器.",
        tools=[],
    )


@pytest.fixture
def cfg_factory():
    """A mock cfg factory that returns a valid cfg dict."""
    def factory():
        return {
            "llm": MagicMock(),
            "pipeline_llm": MagicMock(),
            "agent_registry": MagicMock(),
            "full_tools": MagicMock(),
            "ref_store": None,
            "lesson_store": None,
            "history": [],
            "attachments": [],
            "working_memory": {"pinned": "", "insights": []},
            "circuit_breaker": None,
            "question": "",
        }
    return factory


# ---- AgentToolWrapper tests -------------------------------------------------

class TestAgentToolWrapper:
    def test_tool_name(self, agent_def, cfg_factory):
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
        assert wrapper.tool_name == "agent_test-agent"

    def test_tool_description(self, agent_def, cfg_factory):
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
        assert wrapper.tool_description == "用于测试的领域Agent"

    def test_input_schema(self, agent_def, cfg_factory):
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
        schema = wrapper.input_schema
        assert schema["type"] == "object"
        assert "task" in schema["properties"]
        assert schema["required"] == ["task"]

    def test_capabilities(self, agent_def, cfg_factory):
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
        assert "agent_delegation" in wrapper.capabilities

    def test_to_tool_definition(self, agent_def, cfg_factory):
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
        td = wrapper.to_tool_definition()
        assert isinstance(td, ToolDefinition)
        assert td.name == "agent_test-agent"
        assert td.description == "用于测试的领域Agent"
        assert "agent_delegation" in td.capabilities


# ── Handler tests ────────────────────────────────────────────────────────────

class TestAgentToolHandler:
    def test_empty_task_returns_error(self, agent_def, cfg_factory):
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
        result = wrapper(task="")
        assert "error" in result
        assert "task 参数为空" in result["error"]

    def test_whitespace_task_returns_error(self, agent_def, cfg_factory):
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
        result = wrapper(task="   ")
        assert "error" in result

    @patch("matrix.tools.agent_tool._call_depth")
    def test_max_depth_blocks_recursion(self, mock_depth, agent_def, cfg_factory):
        """When depth >= _MAX_DEPTH, the handler should return an error."""
        mock_depth.get.return_value = _MAX_DEPTH
        mock_depth.set.return_value = None
        mock_depth.reset.return_value = None

        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)
        result = wrapper(task="some task")

        assert "error" in result
        assert "最大 Agent 委派深度" in result["error"]

    def test_cfg_factory_returns_empty_dict(self, agent_def):
        """When cfg_factory returns empty dict, handler should return error."""
        def empty_factory():
            return {}

        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=empty_factory)
        result = wrapper(task="some task")
        assert "error" in result
        assert "agent_registry" in result["error"].lower() or "配置" in result["error"]

    def test_cfg_factory_raises_exception(self, agent_def):
        """When cfg_factory raises, handler should return error."""
        def raising_factory():
            raise RuntimeError("config init failed")

        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=raising_factory)
        result = wrapper(task="some task")
        assert "error" in result
        assert "cfg_factory" in result["error"].lower() or "配置" in result["error"]

    def test_successful_execution(self, agent_def, cfg_factory):
        """Test a successful agent-tool execution with mocked ReAct loop."""
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)

        # Mock the cfg to return proper agent_registry and full_tools
        cfg = cfg_factory()
        mock_registry = cfg["agent_registry"]
        mock_tools = cfg["full_tools"]
        mock_registry.build_tool_registry.return_value = MagicMock()

        # Reset depth to 0
        _call_depth.set(0)

        with patch(
            "matrix.orchestration.runtime_adapter.run_nested_agent_runtime",
            return_value={"answer": "任务完成结果", "tool_results": [{"data": "test"}]},
        ):
            result = wrapper(task="分析持仓")

        assert "result" in result
        assert result["result"] == "任务完成结果"
        assert result["agent"] == "test-agent"
        assert result["tool_results_count"] == 1

    def test_empty_answer_returns_error(self, agent_def, cfg_factory):
        """When the ReAct loop returns an empty answer, handler should error."""
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)

        _call_depth.set(0)

        with patch(
            "matrix.orchestration.runtime_adapter.run_nested_agent_runtime",
            return_value={"answer": "", "tool_results": []},
        ):
            result = wrapper(task="some task")

        assert "error" in result
        assert "空回答" in result["error"]

    def test_answer_truncation(self, agent_def, cfg_factory):
        """Answers exceeding _MAX_ANSWER_LENGTH should be truncated."""
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)

        _call_depth.set(0)

        long_answer = "A" * 5000
        with patch(
            "matrix.orchestration.runtime_adapter.run_nested_agent_runtime",
            return_value={"answer": long_answer, "tool_results": []},
        ):
            result = wrapper(task="some task")

        assert "result" in result
        assert len(result["result"]) < 5000
        assert result.get("truncated") is True
        assert "[已截断]" in result["result"]

    def test_react_exception_returns_error(self, agent_def, cfg_factory):
        """When _run_domain_agent_react raises, handler should return error."""
        wrapper = AgentToolWrapper(agent_def=agent_def, cfg_factory=cfg_factory)

        _call_depth.set(0)

        with patch(
            "matrix.orchestration.runtime_adapter.run_nested_agent_runtime",
            side_effect=RuntimeError("LLM service unavailable"),
        ):
            result = wrapper(task="some task")

        assert "error" in result
        assert "RuntimeError" in result["error"]


# ── make_agent_tool tests ────────────────────────────────────────────────────

class TestMakeAgentTool:
    def test_creates_valid_tool_definition(self, agent_def, cfg_factory):
        td = make_agent_tool(agent_def, cfg_factory)
        assert isinstance(td, ToolDefinition)
        assert td.name == "agent_test-agent"
        assert td.input_schema["properties"]["task"]["type"] == "string"

    def test_handler_is_callable(self, agent_def, cfg_factory):
        td = make_agent_tool(agent_def, cfg_factory)
        assert callable(td.handler)


# ── _has_agent_tool_dependency tests ─────────────────────────────────────────

class TestHasAgentToolDependency:
    def test_no_dependency(self):
        agent = AgentDefinition(
            id="a", name="A", description="d", domain="t",
            persona="p", tools=["finance.*", "web_search"],
        )
        assert _has_agent_tool_dependency(agent) is False

    def test_exact_agent_tool(self):
        agent = AgentDefinition(
            id="a", name="A", description="d", domain="t",
            persona="p", tools=["agent_other_agent"],
        )
        assert _has_agent_tool_dependency(agent) is True

    def test_wildcard_agent_tool(self):
        agent = AgentDefinition(
            id="a", name="A", description="d", domain="t",
            persona="p", tools=["agent.*"],
        )
        assert _has_agent_tool_dependency(agent) is True

    def test_empty_tools(self):
        agent = AgentDefinition(
            id="a", name="A", description="d", domain="t",
            persona="p", tools=[],
        )
        assert _has_agent_tool_dependency(agent) is False


# ── register_agent_tools tests ──────────────────────────────────────────────

class TestRegisterAgentTools:
    def test_registers_all_domain_agents(self):
        """All non-commander, non-circular agents should be registered."""
        mock_registry = MagicMock()
        mock_agent_registry = MagicMock()

        agents = [
            AgentDefinition(
                id="agent-1", name="Agent 1", description="First agent",
                domain="test", persona="p1", tools=["finance.*"],
            ),
            AgentDefinition(
                id="agent-2", name="Agent 2", description="Second agent",
                domain="test", persona="p2", tools=["web_search"],
            ),
        ]
        mock_agent_registry.list_domain_agents.return_value = agents

        cfg_factory = lambda: {}  # noqa: E731
        count = register_agent_tools(mock_registry, mock_agent_registry, cfg_factory)

        assert count == 2
        assert mock_registry.register.call_count == 2

    def test_skips_circular_agents(self):
        """Agents with agent_* in their tools should be skipped."""
        mock_registry = MagicMock()
        mock_agent_registry = MagicMock()

        agents = [
            AgentDefinition(
                id="safe-agent", name="Safe", description="Safe agent",
                domain="test", persona="p1", tools=["finance.*"],
            ),
            AgentDefinition(
                id="circular-agent", name="Circular", description="Has agent tools",
                domain="test", persona="p2", tools=["agent_safe_agent"],
            ),
        ]
        mock_agent_registry.list_domain_agents.return_value = agents

        cfg_factory = lambda: {}  # noqa: E731
        count = register_agent_tools(mock_registry, mock_agent_registry, cfg_factory)

        assert count == 1  # Only safe-agent registered
        # Verify the registered tool name
        registered_td = mock_registry.register.call_args[0][0]
        assert registered_td.name == "agent_safe-agent"

    def test_duplicate_name_skipped(self):
        """If a tool with the same name already exists, it should be skipped."""
        mock_registry = MagicMock()
        mock_registry.register.side_effect = [None, ValueError("already exists")]

        mock_agent_registry = MagicMock()
        mock_agent_registry.list_domain_agents.return_value = [
            AgentDefinition(
                id="a1", name="A1", description="d1",
                domain="t", persona="p", tools=["finance.*"],
            ),
            AgentDefinition(
                id="a2", name="A2", description="d2",
                domain="t", persona="p", tools=["web_search"],
            ),
        ]

        cfg_factory = lambda: {}  # noqa: E731
        count = register_agent_tools(mock_registry, mock_agent_registry, cfg_factory)

        assert count == 1  # Only first one succeeded

    def test_no_domain_agents(self):
        """When there are no domain agents, should return 0."""
        mock_registry = MagicMock()
        mock_agent_registry = MagicMock()
        mock_agent_registry.list_domain_agents.return_value = []

        count = register_agent_tools(
            mock_registry, mock_agent_registry, lambda: {},
        )
        assert count == 0
        mock_registry.register.assert_not_called()


# ── Recursion depth tracking tests ──────────────────────────────────────────

class TestRecursionDepth:
    def test_depth_starts_at_zero(self):
        """By default, the call depth should be 0."""
        # Reset to ensure clean state
        _call_depth.set(0)
        assert _call_depth.get() == 0

    def test_depth_increment_and_reset(self):
        """Depth should be incremented and then reset after the handler completes."""
        _call_depth.set(0)
        token = _call_depth.set(_call_depth.get() + 1)
        try:
            assert _call_depth.get() == 1
        finally:
            _call_depth.reset(token)
        assert _call_depth.get() == 0
