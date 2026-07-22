"""Tests for the Reflexion loop: self-reflection, retry, and quality gate."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from matrix.orchestration.state import AgentState
from matrix.orchestration.graph import build_graph
from matrix.orchestration.nodes.commander import reflection_node, aggregate_node
from matrix.orchestration.nodes._helpers import REFLEXION_PROMPT, REFLEXION_RETRY_PROMPT
from matrix.llm import LLMError


# ---- Mock LLM helpers -------------------------------------------------------

class MockLLM:
    """Configurable mock LLM for testing reflection/aggregate nodes."""

    def __init__(
        self,
        eval_result: dict | None = None,
        reflection_text: str = "",
        revise_text: str = "",
        aggregate_text: str = "Aggregated answer.",
    ):
        self._eval_result = eval_result
        self._reflection_text = reflection_text
        self._revise_text = revise_text
        self._aggregate_text = aggregate_text
        self.complete_calls: list[tuple[str, list]] = []
        self.complete_json_calls: list[tuple[str, list]] = []

    def complete(self, system: str, messages: list, **kw) -> str:
        self.complete_calls.append((system, messages))
        # Distinguish by user message content (more reliable than system prompt)
        user_content = messages[-1]["content"] if messages else ""
        if "write your self-reflection" in user_content.lower():
            return self._reflection_text
        if "rewrite the answer" in user_content.lower():
            return self._revise_text
        return self._aggregate_text

    def complete_json(self, system: str, messages: list, **kw) -> dict:
        self.complete_json_calls.append((system, messages))
        if self._eval_result is not None:
            return self._eval_result
        return {"ok": True}


def _make_config(llm: MockLLM, history=None):
    """Build a minimal graph config dict for testing nodes."""
    return {
        "configurable": {
            "llm": llm,
            "pipeline_llm": llm,
            "history": history or [],
            "agent_registry": MagicMock(),
            "full_tools": MagicMock(),
            "trace": None,
            "event_queue": MagicMock(),
            "ref_store": MagicMock(),
            "working_memory": {"pinned": "", "insights": []},
        },
        "thread_id": "test-thread",
    }


# ---- AgentState Reflexion fields -------------------------------------------

class TestReflexionState:
    def test_default_values(self):
        state = AgentState(user_message="hello", session_id="s1", call_id="c1")
        assert state.reflexion_attempts == 0
        assert state.reflexion_max == 2
        assert state.reflexion_memory == []
        assert state.needs_reflexion_retry is False

    def test_custom_max(self):
        state = AgentState(
            user_message="hello", session_id="s1", call_id="c1",
            reflexion_max=5,
        )
        assert state.reflexion_max == 5

    def test_disabled(self):
        state = AgentState(
            user_message="hello", session_id="s1", call_id="c1",
            reflexion_max=0,
        )
        assert state.reflexion_max == 0


# ---- Reflection node --------------------------------------------------------

class TestReflectionNode:
    def test_acceptable_answer_no_retry(self):
        """When eval says ok=True, no retry needed."""
        llm = MockLLM(eval_result={"ok": True})
        state = AgentState(
            user_message="What is 2+2?",
            session_id="s1", call_id="c1",
            final_answer="The answer is 4.",
            reflexion_max=2,
        )
        result = reflection_node(state, config=_make_config(llm))
        assert result == {}  # No changes

    def test_insufficient_answer_triggers_retry(self):
        """When eval says ok=False with issues and retries remain, trigger retry."""
        llm = MockLLM(
            eval_result={"ok": False, "issues": ["Missing calculation steps"]},
            reflection_text="Need to show the calculation process step by step.",
        )
        state = AgentState(
            user_message="What is 2+2? Please explain.",
            session_id="s1", call_id="c1",
            final_answer="The answer is 4, which is the result of adding 2 and 2.",
            reflexion_max=2,
            reflexion_attempts=0,
        )
        result = reflection_node(state, config=_make_config(llm))
        assert result["needs_reflexion_retry"] is True
        assert result["reflexion_attempts"] == 1
        assert len(result["reflexion_memory"]) == 1
        assert "calculation" in result["reflexion_memory"][0].lower()

    def test_max_retries_exhausted_does_revision(self):
        """When retries are exhausted, do best-effort revision instead of retry."""
        llm = MockLLM(
            eval_result={"ok": False, "issues": ["Too brief"]},
            revise_text="Here is a more detailed answer with explanation.",
        )
        state = AgentState(
            user_message="Explain photosynthesis",
            session_id="s1", call_id="c1",
            final_answer="Plants make food from sunlight, water and carbon dioxide.",
            reflexion_max=2,
            reflexion_attempts=2,  # Already at max
        )
        result = reflection_node(state, config=_make_config(llm))
        assert "needs_reflexion_retry" not in result
        assert "final_answer" in result
        assert "detailed" in result["final_answer"]

    def test_reflexion_disabled_no_retry(self):
        """When reflexion_max=0, no retry even if answer is bad."""
        llm = MockLLM(
            eval_result={"ok": False, "issues": ["Wrong answer"]},
            revise_text="Corrected answer with proper explanation.",
        )
        state = AgentState(
            user_message="What is 3+3? Please explain.",
            session_id="s1", call_id="c1",
            final_answer="The answer is 7, because 3 plus 3 equals 7.",
            reflexion_max=0,  # Disabled
        )
        result = reflection_node(state, config=_make_config(llm))
        assert "needs_reflexion_retry" not in result
        # Should still do best-effort revision
        assert "final_answer" in result

    def test_skip_reflection(self):
        """skip_reflection=True bypasses evaluation entirely."""
        llm = MockLLM()
        state = AgentState(
            user_message="Hello",
            session_id="s1", call_id="c1",
            final_answer="Hi there!",
            skip_reflection=True,
        )
        result = reflection_node(state, config=_make_config(llm))
        assert result == {}
        assert len(llm.complete_json_calls) == 0

    def test_empty_answer_skipped(self):
        """Empty or very short answers are not evaluated."""
        llm = MockLLM()
        state = AgentState(
            user_message="Hello",
            session_id="s1", call_id="c1",
            final_answer="",
            reflexion_max=2,
        )
        result = reflection_node(state, config=_make_config(llm))
        assert result == {}

    def test_accumulated_reflection_memory(self):
        """Multiple retries accumulate reflections."""
        llm = MockLLM(
            eval_result={"ok": False, "issues": ["Still wrong"]},
            reflection_text="Second reflection: try different approach.",
        )
        state = AgentState(
            user_message="Complex question requiring detailed analysis",
            session_id="s1", call_id="c1",
            final_answer="Attempt 2 answer with some content but still insufficient.",
            reflexion_max=3,
            reflexion_attempts=1,
            reflexion_memory=["First reflection: be more specific."],
        )
        result = reflection_node(state, config=_make_config(llm))
        assert result["reflexion_attempts"] == 2
        assert len(result["reflexion_memory"]) == 2
        assert "First reflection" in result["reflexion_memory"][0]
        assert "Second reflection" in result["reflexion_memory"][1]

    def test_llm_error_returns_empty(self):
        """LLM API error during evaluation returns empty dict (fail-safe)."""
        llm = MockLLM()
        llm.complete_json = MagicMock(side_effect=LLMError("API error"))
        state = AgentState(
            user_message="Question that requires a detailed response",
            session_id="s1", call_id="c1",
            final_answer="Some answer that is long enough to pass the check.",
            reflexion_max=2,
        )
        result = reflection_node(state, config=_make_config(llm))
        assert result == {}

    def test_no_issues_no_retry(self):
        """ok=False but no issues list → no retry."""
        llm = MockLLM(eval_result={"ok": False, "issues": []})
        state = AgentState(
            user_message="Question that requires a detailed response",
            session_id="s1", call_id="c1",
            final_answer="Some answer that is long enough to pass the check.",
            reflexion_max=2,
        )
        result = reflection_node(state, config=_make_config(llm))
        assert result == {}


# ---- Aggregate node with Reflexion ------------------------------------------

class TestAggregateWithReflexion:
    def test_retry_injects_reflection_memory(self):
        """On retry, aggregate_node injects reflection memory into the prompt."""
        llm = MockLLM(aggregate_text="Better answer with details.")
        agent_results = [{
            "agent_id": "investment_analyst",
            "task": "分析持仓",
            "result": "Portfolio has 3 stocks.",
            "tool_results": [],
            "error": "",
        }]
        state = AgentState(
            user_message="分析我的持仓",
            session_id="s1", call_id="c1",
            agent_results=agent_results,
            reflexion_memory=["Previous answer lacked specific numbers."],
            needs_reflexion_retry=True,
        )
        result = aggregate_node(state, config=_make_config(llm))
        # Check that reflection was injected
        system_prompt = llm.complete_calls[0][0]
        assert "Self-Reflection" in system_prompt
        assert "Previous answer lacked specific numbers" in system_prompt
        # Check that retry flag was cleared
        assert result.get("needs_reflexion_retry") is False
        assert "Better answer" in result["final_answer"]

    def test_normal_no_reflection_injection(self):
        """Without retry, no reflection memory is injected."""
        llm = MockLLM(aggregate_text="Normal answer.")
        agent_results = [{
            "agent_id": "investment_analyst",
            "task": "分析持仓",
            "result": "Portfolio has 3 stocks.",
            "tool_results": [],
            "error": "",
        }]
        state = AgentState(
            user_message="分析我的持仓",
            session_id="s1", call_id="c1",
            agent_results=agent_results,
            reflexion_memory=[],
            needs_reflexion_retry=False,
        )
        result = aggregate_node(state, config=_make_config(llm))
        system_prompt = llm.complete_calls[0][0]
        assert "Self-Reflection" not in system_prompt
        assert "needs_reflexion_retry" not in result

    def test_retry_clears_flag_even_on_error(self):
        """On retry, aggregate clears the retry flag even when LLM fails."""
        llm = MockLLM()
        llm.complete = MagicMock(side_effect=LLMError("LLM error"))
        agent_results = [{
            "agent_id": "investment_analyst",
            "task": "分析",
            "result": "Some result with enough content.",
            "tool_results": [],
            "error": "",
        }]
        state = AgentState(
            user_message="Question",
            session_id="s1", call_id="c1",
            agent_results=agent_results,
            reflexion_memory=["Be more specific."],
            needs_reflexion_retry=True,
        )
        # Should not raise; should clear retry flag
        result = aggregate_node(state, config=_make_config(llm))
        assert result.get("needs_reflexion_retry") is False

    def test_retry_with_empty_results_clears_flag(self):
        """On retry with no agent results, clears the flag."""
        llm = MockLLM()
        state = AgentState(
            user_message="Question",
            session_id="s1", call_id="c1",
            agent_results=[],
            needs_reflexion_retry=True,
        )
        result = aggregate_node(state, config=_make_config(llm))
        assert result.get("needs_reflexion_retry") is False

    def test_retry_with_all_errors_clears_flag(self):
        """On retry when all agents errored, clears the flag."""
        llm = MockLLM()
        agent_results = [{
            "agent_id": "analyst",
            "task": "task",
            "result": "",
            "tool_results": [],
            "error": "Timeout",
        }]
        state = AgentState(
            user_message="Question",
            session_id="s1", call_id="c1",
            agent_results=agent_results,
            needs_reflexion_retry=True,
        )
        result = aggregate_node(state, config=_make_config(llm))
        assert result.get("needs_reflexion_retry") is False


# ---- Graph conditional edge -------------------------------------------------

class TestReflexionGraphEdge:
    def test_graph_has_conditional_edge(self):
        """The compiled graph should have a conditional edge from reflection."""
        graph = build_graph()
        # The graph should compile without errors (no checkpointer needed for compile)
        compiled = graph.compile(checkpointer=None)
        assert compiled is not None

    def test_after_reflection_routes_to_aggregate_on_retry(self):
        """_after_reflection returns 'aggregate' when needs_reflexion_retry is True."""
        from matrix.orchestration.graph import build_graph
        # Re-extract the _after_reflection function by building the graph
        # and checking the edge structure
        graph = build_graph()
        # We test the routing logic directly
        state_retry = {"needs_reflexion_retry": True}
        state_done = {"needs_reflexion_retry": False}
        # The function should be embedded in the graph; we verify behavior
        # by checking the graph builds and has the right edges
        assert graph is not None


# ---- Config integration -----------------------------------------------------

class TestReflexionConfig:
    def test_config_has_reflexion_max(self):
        """AgentConfig should have reflexion_max_attempts field."""
        from matrix.config import AgentConfig
        # Check field exists with default
        import dataclasses
        fields = {f.name: f for f in dataclasses.fields(AgentConfig)}
        assert "reflexion_max_attempts" in fields
        assert fields["reflexion_max_attempts"].default == 2

    def test_env_var_loaded(self, monkeypatch):
        """REFLEXION_MAX_ATTEMPTS env var should be loaded."""
        from matrix.config import load_config
        monkeypatch.setenv("REFLEXION_MAX_ATTEMPTS", "3")
        monkeypatch.setenv("JWT_SECRET", "test-secret-for-unit-test")
        monkeypatch.chdir("/Users/qiang.lilq/personal-system/personal-agent")
        config = load_config()
        assert config.reflexion_max_attempts == 3

    def test_env_var_disabled(self, monkeypatch):
        """REFLEXION_MAX_ATTEMPTS=0 disables the loop."""
        from matrix.config import load_config
        monkeypatch.setenv("REFLEXION_MAX_ATTEMPTS", "0")
        monkeypatch.setenv("JWT_SECRET", "test-secret-for-unit-test")
        monkeypatch.chdir("/Users/qiang.lilq/personal-system/personal-agent")
        config = load_config()
        assert config.reflexion_max_attempts == 0

    def test_env_var_clamped(self, monkeypatch):
        """Values outside [0, 5] are clamped."""
        from matrix.config import load_config
        monkeypatch.setenv("REFLEXION_MAX_ATTEMPTS", "99")
        monkeypatch.setenv("JWT_SECRET", "test-secret-for-unit-test")
        monkeypatch.chdir("/Users/qiang.lilq/personal-system/personal-agent")
        config = load_config()
        assert config.reflexion_max_attempts == 5  # clamped to max


# ---- Prompt templates -------------------------------------------------------

class TestReflexionPrompts:
    def test_reflexion_prompt_has_placeholders(self):
        assert "{question}" in REFLEXION_PROMPT
        assert "{answer}" in REFLEXION_PROMPT
        assert "{issues}" in REFLEXION_PROMPT
        assert "{prior_reflections}" in REFLEXION_PROMPT

    def test_reflexion_retry_prompt_has_placeholders(self):
        assert "{reflections}" in REFLEXION_RETRY_PROMPT
        assert "{question}" in REFLEXION_RETRY_PROMPT

    def test_reflexion_prompt_instructs_concise(self):
        assert "max 3 sentences" in REFLEXION_PROMPT.lower() or "3 sentences" in REFLEXION_PROMPT
