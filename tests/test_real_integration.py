"""Real LLM integration tests — requires DEEPSEEK_API_KEY and real cache.

Run with real env:
  DEEPSEEK_API_KEY=sk-xxx PERSONAL_OS_CACHE_PATH=... python3 -m pytest tests/test_real_integration.py -v -s
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from matrix.chat import ChatService
from matrix.config import load_config
from matrix.llm import build_llm_client
from matrix.orchestration import build_graph
from matrix.orchestration.state import AgentState
from matrix.role import INVESTMENT_ANALYST
from matrix.skills import load_skills
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
    return r


@pytest.fixture(scope="module")
def real_skills():
    skills_dir = Path("skills/investment")
    return load_skills(skills_dir) if skills_dir.exists() else []


# ---- Classify Tests ----

class TestClassifyReal:
    def test_classify_simple_question(self, real_llm, real_tools):
        """Simple question → react."""
        graph = build_graph()
        compiled = graph.compile()

        state: AgentState = {
            "messages": [],
            "user_message": "我有多少持仓？",
            "session_id": "real-test",
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
                "llm": real_llm,
                "tools": real_tools,
                "role": INVESTMENT_ANALYST,
            }},
        ))
        final = events[-1]
        answer = final.get("final_answer", "")
        print(f"\n  [classify → react] answer: {answer[:200]}")
        assert len(answer) > 5

    def test_classify_complex_analysis(self, real_llm, real_tools):
        """Complex analysis → plan_execute."""
        graph = build_graph()
        compiled = graph.compile()

        state: AgentState = {
            "messages": [],
            "user_message": "分析我当前的资产配置偏离度，并给出再平衡建议",
            "session_id": "real-test",
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
                "llm": real_llm,
                "tools": real_tools,
                "role": INVESTMENT_ANALYST,
            }},
        ))
        final = events[-1]
        answer = final.get("final_answer", "")
        print(f"\n  [classify → plan_execute] answer: {answer[:300]}")
        assert len(answer) > 10


# ---- ReAct Tests ----

class TestReactReal:
    def test_single_tool_react(self, real_llm, real_tools):
        """Single tool call → react should complete."""
        graph = build_graph()
        compiled = graph.compile()

        state: AgentState = {
            "messages": [],
            "user_message": "当前持仓总金额是多少？",
            "session_id": "real-test",
            "intent": "react",
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
                "llm": real_llm,
                "tools": real_tools,
                "role": INVESTMENT_ANALYST,
            }},
        ))
        final = events[-1]
        answer = final.get("final_answer", "")
        print(f"\n  [react] answer: {answer[:300]}")
        assert len(answer) > 5

    def test_multi_tool_react(self, real_llm, real_tools):
        """Multi-tool ReAct — should call multiple tools and summarize."""
        graph = build_graph()
        compiled = graph.compile()

        state: AgentState = {
            "messages": [],
            "user_message": "查看我最近snapshot中cash分桶的资产情况",
            "session_id": "real-test",
            "intent": "react",
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
                "llm": real_llm,
                "tools": real_tools,
                "role": INVESTMENT_ANALYST,
            }},
        ))
        final = events[-1]
        answer = final.get("final_answer", "")
        print(f"\n  [react multi] answer: {answer[:300]}")
        tool_count = final.get("tool_call_count", 0)
        print(f"  tools called: {tool_count}")
        assert len(answer) > 5


# ---- Plan-Execute Tests ----

class TestPlanExecuteReal:
    def test_plan_execute_complete(self, real_llm, real_tools):
        """Plan-Execute full flow with real LLM."""
        graph = build_graph()
        compiled = graph.compile()

        state: AgentState = {
            "messages": [],
            "user_message": "分析我当前持仓的币种分布和风险等级分布",
            "session_id": "real-test",
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
                "llm": real_llm,
                "tools": real_tools,
                "role": INVESTMENT_ANALYST,
            }},
        ))
        final = events[-1]
        answer = final.get("final_answer", "")
        plan = final.get("current_plan", [])
        tool_count = final.get("tool_call_count", 0)
        print(f"\n  [plan-execute] plan steps: {len(plan)}, tools: {tool_count}")
        print(f"  answer: {answer[:300]}")
        assert len(answer) > 10
        assert tool_count >= 1


# ---- Skill Tests ----

class TestSkillReal:
    def test_anomaly_diagnosis_skill(self, real_llm, real_tools, real_skills):
        """Anomaly diagnosis skill with real data."""
        if not real_skills:
            pytest.skip("No skills loaded")

        graph = build_graph()
        compiled = graph.compile()

        state: AgentState = {
            "messages": [],
            "user_message": "诊断我的持仓异动",
            "session_id": "real-test",
            "intent": "skill",
            "skill_name": "anomaly-diagnosis",
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
                "llm": real_llm,
                "tools": real_tools,
                "role": INVESTMENT_ANALYST,
                "skills": real_skills,
            }},
        ))
        final = events[-1]
        answer = final.get("final_answer", "")
        tool_count = final.get("tool_call_count", 0)
        print(f"\n  [skill: anomaly-diagnosis] tools: {tool_count}")
        print(f"  answer: {answer[:300]}")
        assert len(answer) > 5

    def test_portfolio_review_skill(self, real_llm, real_tools, real_skills):
        """Portfolio review skill with real data."""
        if not real_skills:
            pytest.skip("No skills loaded")

        graph = build_graph()
        compiled = graph.compile()

        state: AgentState = {
            "messages": [],
            "user_message": "做一次组合复盘",
            "session_id": "real-test",
            "intent": "skill",
            "skill_name": "portfolio-review",
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
                "llm": real_llm,
                "tools": real_tools,
                "role": INVESTMENT_ANALYST,
                "skills": real_skills,
            }},
        ))
        final = events[-1]
        answer = final.get("final_answer", "")
        tool_count = final.get("tool_call_count", 0)
        print(f"\n  [skill: portfolio-review] tools: {tool_count}")
        print(f"  answer: {answer[:300]}")
        assert len(answer) > 5

    def test_allocation_check_skill(self, real_llm, real_tools, real_skills):
        """Allocation check skill with real data."""
        if not real_skills:
            pytest.skip("No skills loaded")

        graph = build_graph()
        compiled = graph.compile()

        state: AgentState = {
            "messages": [],
            "user_message": "检查配置偏离度",
            "session_id": "real-test",
            "intent": "skill",
            "skill_name": "allocation-check",
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
                "llm": real_llm,
                "tools": real_tools,
                "role": INVESTMENT_ANALYST,
                "skills": real_skills,
            }},
        ))
        final = events[-1]
        answer = final.get("final_answer", "")
        tool_count = final.get("tool_call_count", 0)
        print(f"\n  [skill: allocation-check] tools: {tool_count}")
        print(f"  answer: {answer[:300]}")
        assert len(answer) > 5


# ---- Chat Service Graph Mode ----

class TestChatServiceGraphReal:
    def test_stream_chat_graph(self, real_llm, real_tools):
        """ChatService stream_chat_graph with real LLM."""
        config = load_config()
        service = ChatService(config, real_tools, llm=real_llm)

        events = list(service.stream_chat_graph("当前持仓怎么样？"))
        types = [e["type"] for e in events]
        tokens = [e for e in events if e["type"] == "token"]
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        errors = [e for e in events if e["type"] == "error"]

        print(f"\n  [chat service graph] events: {len(events)}")
        print(f"  types: {types}")
        if tokens:
            print(f"  answer: {tokens[-1]['content'][:300]}")
        if errors:
            print(f"  errors: {[e['message'] for e in errors]}")

        assert "done" in types
        assert len(tokens) >= 1, "Should have at least one token event"
        assert len(tool_calls) >= 1, "Should have at least one tool call"

    def test_graph_with_memory(self, real_llm, real_tools):
        """Multi-turn conversation with graph mode."""
        config = load_config()
        service = ChatService(config, real_tools, llm=real_llm)
        sid = "real-memory-test"

        # Turn 1
        events1 = list(service.stream_chat_graph("当前持仓有哪些？", sid))
        tokens1 = [e for e in events1 if e["type"] == "token"]
        print(f"\n  [memory turn 1] answer: {tokens1[-1]['content'][:200] if tokens1 else 'N/A'}")

        # Turn 2 — should remember context
        events2 = list(service.stream_chat_graph("刚才提到的那个分桶占比最高？", sid))
        tokens2 = [e for e in events2 if e["type"] == "token"]
        print(f"  [memory turn 2] answer: {tokens2[-1]['content'][:200] if tokens2 else 'N/A'}")

        assert len(tokens1) >= 1
        assert len(tokens2) >= 1

        service.reset(sid)  # cleanup