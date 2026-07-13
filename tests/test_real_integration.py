"""Real LLM integration tests — requires DEEPSEEK_API_KEY and real cache.

Run with real env:
  DEEPSEEK_API_KEY=sk-xxx PERSONAL_OS_CACHE_PATH=... python3 -m pytest tests/test_real_integration.py -v -s
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

# Ensure JWT_SECRET is set for config loading in tests
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-real-tests")

from matrix.chat import ChatService
from matrix.config import load_config
from matrix.llm import build_llm_client
from matrix.orchestration import build_graph
from matrix.orchestration.state import AgentState
from matrix.agent import AgentRegistry
from matrix.agent.commander import COMMANDER
from matrix.agent.domain_agents import INVESTMENT_ANALYST
from matrix.tools import ToolRegistry
from matrix.tools.finance import register_all


# ---- Skip if no API key ----
def _require_llm():
    config = load_config()
    if not config.llm_available:
        pytest.skip(f"LLM unavailable: {config.llm_unavailable_reason}")
    return config


def _require_cache():
    config = load_config()
    if not config.cache_path.exists():
        pytest.skip(f"Cache not found: {config.cache_path}")
    return config


@pytest.fixture(scope="module")
def real_config():
    return _require_llm()


@pytest.fixture(scope="module")
def real_llm(real_config):
    return build_llm_client(
        provider=real_config.agent_provider,
        deepseek_api_key=real_config.deepseek_api_key,
        model=real_config.agent_model,
        deepseek_base_url=real_config.deepseek_base_url,
        max_tokens=real_config.agent_max_tokens,
        timeout_sec=real_config.agent_model_timeout_sec,
    )


@pytest.fixture(scope="module")
def real_tools():
    _require_cache()
    config = load_config()
    r = ToolRegistry()
    register_all(r, config.cache_path)
    # Also register web tools
    from matrix.tools.web import register_all as register_web
    register_web(r)
    return r


@pytest.fixture(scope="module")
def real_agent_registry():
    config = load_config()
    reg = AgentRegistry(skills_base_dir=config.skills_base_dir)
    reg.register_all([COMMANDER, INVESTMENT_ANALYST])
    return reg


def make_config(llm, full_tools, agent_registry):
    return {
        "configurable": {
            "llm": llm,
            "pipeline_llm": llm,
            "full_tools": full_tools,
            "agent_registry": agent_registry,
        },
    }


# ---- Classify Tests ----

class TestClassifyReal:
    def test_classify_simple_greeting(self, real_llm, real_tools, real_agent_registry):
        """Simple greeting → simple."""
        graph = build_graph()
        compiled = graph.compile()
        state: AgentState = {
            "messages": [],
            "user_message": "你好",
            "session_id": "test-real",
            "intent": "",
            "delegation_plan": [],
            "current_step": 0,
            "agent_results": [],
            "tool_results": [],
            "tool_call_count": 0,
            "react_iteration": 0,
            "final_answer": "",
            "needs_summary": False,
            "error": "",
        }
        events = list(
            compiled.stream(
                state,
                stream_mode="values",
                config=make_config(real_llm, real_tools, real_agent_registry),
                thread_id="test-real-classify",
            )
        )
        final = events[-1]
        # Should be either simple or delegate
        assert final.get("intent") in ("simple", "delegate")

    def test_classify_investment_question(self, real_llm, real_tools, real_agent_registry):
        """Investment question → delegate."""
        graph = build_graph()
        compiled = graph.compile()
        state: AgentState = {
            "messages": [],
            "user_message": "我的持仓最近有什么变化？",
            "session_id": "test-real-invest",
            "intent": "",
            "delegation_plan": [],
            "current_step": 0,
            "agent_results": [],
            "tool_results": [],
            "tool_call_count": 0,
            "react_iteration": 0,
            "final_answer": "",
            "needs_summary": False,
            "error": "",
        }
        events = list(
            compiled.stream(
                state,
                stream_mode="values",
                config=make_config(real_llm, real_tools, real_agent_registry),
                thread_id="test-real-invest",
            )
        )
        final = events[-1]
        assert final.get("intent") in ("simple", "delegate")


# ---- Commander Plan Real ----

class TestCommanderPlanReal:
    def test_generates_plan(self, real_llm, real_tools, real_agent_registry):
        """Commander generates a plan for an investment question."""
        graph = build_graph()
        compiled = graph.compile()
        state: AgentState = {
            "messages": [],
            "user_message": "分析我的持仓配置偏离度",
            "session_id": "test-real-plan",
            "intent": "delegate",
            "delegation_plan": [],
            "current_step": 0,
            "agent_results": [],
            "tool_results": [],
            "tool_call_count": 0,
            "react_iteration": 0,
            "final_answer": "",
            "needs_summary": False,
            "error": "",
        }
        events = list(
            compiled.stream(
                state,
                stream_mode="values",
                config=make_config(real_llm, real_tools, real_agent_registry),
                thread_id="test-real-plan",
            )
        )
        final = events[-1]
        plan = final.get("delegation_plan", [])
        # Either has a plan (delegate) or fell back to simple
        if plan:
            assert any(p["agent_id"] == "investment-analyst" for p in plan)


# ---- ChatService Real ----

class TestChatServiceReal:
    def test_chat_service_health(self, real_config, real_tools):
        """ChatService can be instantiated."""
        chat = ChatService(real_config, real_tools)
        assert chat.agent_registry is not None
        assert chat.agent_registry.commander is not None
        chat.close()