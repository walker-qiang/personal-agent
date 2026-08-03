"""Tests for Tree of Thoughts (ToT) / LATS module."""

from __future__ import annotations

import json

import pytest

from matrix.orchestration.tot import (
    ToTNode,
    ToTResult,
    UCB1Selector,
    ToTEvaluator,
    TreeSearchEngine,
)


# ---- UCB1Selector ----

class TestUCB1Selector:
    def test_single_candidate(self):
        selector = UCB1Selector()
        node = ToTNode(node_id="a", value=0.5, visits=1)
        result = selector.select([node], parent_visits=1)
        assert result is node

    def test_empty_candidates(self):
        selector = UCB1Selector()
        assert selector.select([], parent_visits=1) is None

    def test_prefers_unvisited(self):
        """Unvisited nodes should get infinite UCB (priority exploration)."""
        selector = UCB1Selector()
        visited = ToTNode(node_id="a", value=1.0, visits=10)
        unvisited = ToTNode(node_id="b", value=0.0, visits=0)
        result = selector.select([visited, unvisited], parent_visits=10)
        assert result is unvisited

    def test_balances_exploration_exploitation(self):
        """When all visited, UCB should balance value and exploration."""
        selector = UCB1Selector(exploration_c=1.414)
        high_value = ToTNode(node_id="a", value=0.8, visits=5)
        low_value = ToTNode(node_id="b", value=0.2, visits=5)
        result = selector.select([high_value, low_value], parent_visits=10)
        assert result is high_value

    def test_skips_pruned(self):
        """Pruned nodes should be skipped."""
        selector = UCB1Selector()
        pruned = ToTNode(node_id="a", value=0.9, visits=1, status="pruned")
        active = ToTNode(node_id="b", value=0.3, visits=1)
        result = selector.select([pruned, active], parent_visits=2)
        assert result is active

    def test_equal_values_equal_visits(self):
        """When all equal, should return first (or any)."""
        selector = UCB1Selector()
        a = ToTNode(node_id="a", value=0.5, visits=1)
        b = ToTNode(node_id="b", value=0.5, visits=1)
        result = selector.select([a, b], parent_visits=2)
        assert result in (a, b)


# ---- ToTEvaluator ----

class FakeLLM:
    """Fake LLM for testing."""

    def __init__(self, responses: list | None = None):
        self.responses = responses or []
        self.calls = []

    def complete(self, system, messages, **kwargs):
        self.calls.append(("complete", messages))
        return self.responses.pop(0) if self.responses else "{}"

    def complete_json(self, system, messages, schema=None, **kwargs):
        self.calls.append(("complete_json", messages))
        if not self.responses:
            return {"score": 0.5, "reasoning": "default"}
        resp = self.responses.pop(0)
        if isinstance(resp, str):
            return json.loads(resp)
        return resp


class TestToTEvaluator:
    def test_evaluate_action_with_llm(self):
        llm = FakeLLM([{"score": 0.8, "reasoning": "good action"}])
        evaluator = ToTEvaluator(llm=llm)
        score, reasoning = evaluator.evaluate_action(
            "分析持仓", "finance.holdings_summary", {}
        )
        assert score == 0.8
        assert "good" in reasoning

    def test_evaluate_action_no_llm(self):
        """Without LLM, should use heuristic scoring."""
        evaluator = ToTEvaluator(llm=None)
        score, reasoning = evaluator.evaluate_action(
            "查询持仓", "finance.holdings_summary", {}
        )
        assert 0.0 <= score <= 1.0
        assert reasoning  # Non-empty

    def test_evaluate_action_llm_error(self):
        """When LLM raises, should fall back to heuristic."""
        class ErrorLLM:
            def complete_json(self, *args, **kwargs):
                raise RuntimeError("LLM error")

        evaluator = ToTEvaluator(llm=ErrorLLM())
        score, reasoning = evaluator.evaluate_action("查询", "finance.holdings", {})
        assert 0.0 <= score <= 1.0

    def test_evaluate_plan_with_llm(self):
        llm = FakeLLM([{
            "score": 0.7,
            "reasoning": "complete plan",
            "missing_steps": ["need to check risk"],
        }])
        evaluator = ToTEvaluator(llm=llm)
        score, reasoning, missing = evaluator.evaluate_plan(
            "分析投资组合", [{"step": 1, "agent_id": "investment-analyst", "task": "分析"}]
        )
        assert score == 0.7
        assert "complete" in reasoning
        assert len(missing) == 1

    def test_evaluate_plan_no_llm(self):
        evaluator = ToTEvaluator(llm=None)
        score, reasoning, missing = evaluator.evaluate_plan(
            "分析投资", [{"step": 1, "agent_id": "a", "task": "t", "purpose": "p"}]
        )
        assert 0.0 <= score <= 1.0

    def test_heuristic_score_keyword_match(self):
        """Heuristic should boost score when action keywords appear in task."""
        evaluator = ToTEvaluator(llm=None)
        score, _ = evaluator._heuristic_score(
            "查询finance工具的结果", "finance.holdings_summary", {}
        )
        assert score > 0.5  # "finance" keyword should match

    def test_heuristic_plan_score_empty(self):
        evaluator = ToTEvaluator(llm=None)
        score, _, missing = evaluator._heuristic_plan_score("task", [])
        assert score < 0.2
        assert len(missing) > 0


# ---- TreeSearchEngine ----

class TestTreeSearchEngine:
    def test_select_best_plan_single_candidate(self):
        """Single candidate should be evaluated but not use ToT."""
        llm = FakeLLM()
        engine = TreeSearchEngine(llm=llm)
        result = engine.select_best_plan("任务", [[{"step": 1, "agent_id": "a"}]])

        assert result.used_tot is False
        assert len(result.best_path) == 1
        assert result.total_evaluations == 1

    def test_select_best_plan_multiple_candidates(self):
        """Multiple candidates should trigger ToT evaluation."""
        llm = FakeLLM([
            {"score": 0.3, "reasoning": "poor plan"},  # First plan (will be pruned)
            {"score": 0.8, "reasoning": "good plan"},  # Second plan
        ])
        engine = TreeSearchEngine(llm=llm, min_value=0.2)
        plan1 = [{"step": 1, "agent_id": "a", "task": "t1"}]
        plan2 = [{"step": 1, "agent_id": "b", "task": "t2"}]

        result = engine.select_best_plan("复杂分析任务", [plan1, plan2])

        assert result.used_tot is True
        assert result.total_nodes == 2
        assert result.total_evaluations == 2
        assert result.best_value == 0.8

    def test_select_best_plan_all_pruned(self):
        """When all plans are pruned, should fall back to first."""
        llm = FakeLLM([
            {"score": 0.1, "reasoning": "bad"},  # Both will be pruned
            {"score": 0.1, "reasoning": "bad"},
            {"score": 0.2, "reasoning": "fallback"},  # Fallback evaluation
        ])
        engine = TreeSearchEngine(llm=llm, min_value=0.5)
        plan1 = [{"step": 1, "agent_id": "a", "task": "t1"}]
        plan2 = [{"step": 1, "agent_id": "b", "task": "t2"}]

        result = engine.select_best_plan("任务", [plan1, plan2])

        assert result.used_tot is False
        assert result.best_path  # Should still have a fallback path

    def test_select_best_plan_empty(self):
        """Empty candidate list should return empty result."""
        engine = TreeSearchEngine(llm=FakeLLM())
        result = engine.select_best_plan("任务", [])
        assert result.total_nodes == 0
        assert not result.best_path

    def test_evaluate_action_candidates(self):
        """Should return sorted list of (candidate, score, reasoning)."""
        llm = FakeLLM([
            {"score": 0.3, "reasoning": "low"},
            {"score": 0.9, "reasoning": "high"},
        ])
        engine = TreeSearchEngine(llm=llm, min_value=0.2)
        candidates = [
            {"action": "tool_a", "arguments": {}},
            {"action": "tool_b", "arguments": {}},
        ]

        result = engine.evaluate_action_candidates("任务", candidates)

        assert len(result) == 2
        assert result[0][1] > result[1][1]  # Sorted by score descending

    def test_evaluate_action_candidates_prune_low(self):
        """Candidates below min_value should be pruned."""
        llm = FakeLLM([
            {"score": 0.1, "reasoning": "too low"},
            {"score": 0.8, "reasoning": "good"},
        ])
        engine = TreeSearchEngine(llm=llm, min_value=0.3)
        candidates = [
            {"action": "tool_a", "arguments": {}},
            {"action": "tool_b", "arguments": {}},
        ]

        result = engine.evaluate_action_candidates("任务", candidates)

        assert len(result) == 1  # Only the high-score candidate
        assert result[0][0]["action"] == "tool_b"

    def test_should_use_tot_no_llm(self):
        """Without LLM, ToT should never be used."""
        engine = TreeSearchEngine(llm=None)
        assert engine.should_use_tot("复杂分析任务", 3) is False

    def test_should_use_tot_simple_greeting(self):
        """Simple greetings should not trigger ToT."""
        llm = FakeLLM()
        engine = TreeSearchEngine(llm=llm)
        assert engine.should_use_tot("你好", 1) is False

    def test_should_use_tot_single_step(self):
        """Single-step plans should not trigger ToT."""
        llm = FakeLLM()
        engine = TreeSearchEngine(llm=llm)
        assert engine.should_use_tot("复杂分析", 1) is False

    def test_should_use_tot_complex_task(self):
        """Complex tasks should trigger ToT."""
        llm = FakeLLM()
        engine = TreeSearchEngine(llm=llm)
        assert engine.should_use_tot("分析并比较两个投资组合的风险", 3) is True

    def test_should_use_tot_comparison_keywords(self):
        """Comparison keywords should trigger ToT."""
        llm = FakeLLM()
        engine = TreeSearchEngine(llm=llm)
        assert engine.should_use_tot("比较A股和港股的估值", 2) is True

    def test_should_use_tot_long_plan(self):
        """Long plans (> 2 steps) should trigger ToT."""
        llm = FakeLLM()
        engine = TreeSearchEngine(llm=llm)
        assert engine.should_use_tot("执行任务", 3) is True

    def test_should_use_tot_multi_subquestion(self):
        """Tasks with multiple sub-questions should trigger ToT."""
        llm = FakeLLM()
        engine = TreeSearchEngine(llm=llm)
        assert engine.should_use_tot("查询持仓并分析风险然后给出建议", 2) is True


# ---- ToTNode and ToTResult data structures ----

class TestToTDataStructures:
    def test_tot_node_defaults(self):
        node = ToTNode(node_id="test")
        assert node.action == ""
        assert node.value == 0.0
        assert node.visits == 0
        assert node.depth == 0
        assert node.status == "pending"
        assert node.children_ids == []

    def test_tot_result_defaults(self):
        result = ToTResult()
        assert result.best_path == []
        assert result.total_nodes == 0
        assert result.used_tot is False
        assert result.best_value == 0.0
