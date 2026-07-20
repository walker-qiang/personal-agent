"""Unit tests for DeterministicEvaluator."""

from __future__ import annotations

import pytest

from matrix.evaluation.case import EvalCase, ExpectedBehavior
from matrix.evaluation.evaluators.deterministic import DeterministicEvaluator


@pytest.fixture
def evaluator() -> DeterministicEvaluator:
    return DeterministicEvaluator()


class TestOutcomeCheck:
    def test_answer_no_error(self, evaluator):
        case = EvalCase(case_id="t1", user_input="hello", expected=ExpectedBehavior(outcome="answer"))
        events = [{"type": "token", "content": "hi"}]
        passed, _, details = evaluator.evaluate(case, events, "hi")
        assert passed, f"details={details}"
        assert details["checks"]["outcome"] is True

    def test_answer_with_error(self, evaluator):
        case = EvalCase(case_id="t1", user_input="hello", expected=ExpectedBehavior(outcome="answer"))
        events = [{"type": "error", "message": "fail"}]
        passed, _, details = evaluator.evaluate(case, events, "")
        assert not passed, f"details={details}"
        assert details["checks"]["outcome"] is False

    def test_abstain_expects_error(self, evaluator):
        case = EvalCase(case_id="t1", user_input="hack", expected=ExpectedBehavior(outcome="abstain"))
        events = [{"type": "error", "message": "refused"}]
        passed, _, details = evaluator.evaluate(case, events, "")
        assert passed, f"details={details}"

    def test_abstain_without_error_fails(self, evaluator):
        case = EvalCase(case_id="t1", user_input="hack", expected=ExpectedBehavior(outcome="abstain"))
        events = [{"type": "token", "content": "ok"}]
        passed, _, details = evaluator.evaluate(case, events, "ok")
        assert not passed, f"details={details}"


class TestMustInclude:
    def test_all_keywords_found(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(must_include=["持仓", "健康"]),
        )
        passed, _, details = evaluator.evaluate(case, [], "当前持仓健康。")
        assert passed, f"details={details}"

    def test_missing_keyword(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(must_include=["持仓", "不存在"]),
        )
        passed, _, details = evaluator.evaluate(case, [], "当前持仓健康。")
        assert not passed, f"details={details}"
        assert "不存在" in str(details)

    def test_empty_must_include(self, evaluator):
        case = EvalCase(case_id="t1", user_input="q")
        passed, _, _ = evaluator.evaluate(case, [], "anything")
        assert passed

    def test_case_insensitive(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(must_include=["HELLO"]),
        )
        passed, _, _ = evaluator.evaluate(case, [], "hello world")
        assert passed


class TestMustNotInclude:
    def test_forbidden_found(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(must_not_include=["error"]),
        )
        passed, _, details = evaluator.evaluate(case, [], "an error occurred")
        assert not passed, f"details={details}"

    def test_forbidden_not_found(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(must_not_include=["error"]),
        )
        passed, _, _ = evaluator.evaluate(case, [], "all good")
        assert passed

    def test_empty_must_not_include(self, evaluator):
        case = EvalCase(case_id="t1", user_input="q")
        passed, _, _ = evaluator.evaluate(case, [], "error here")
        assert passed


class TestRequiredTools:
    def test_tool_called(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(required_tools=["web_search"]),
        )
        events = [{"type": "tool_call", "name": "web_search", "arguments": {"query": "x"}}]
        passed, _, _ = evaluator.evaluate(case, events, "")
        assert passed

    def test_tool_not_called(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(required_tools=["web_search"]),
        )
        events = [{"type": "tool_call", "name": "get_holdings", "arguments": {}}]
        passed, _, details = evaluator.evaluate(case, events, "")
        assert not passed, f"details={details}"
        assert "web_search" in str(details["required_tools"]["missing"])

    def test_empty_required_tools(self, evaluator):
        case = EvalCase(case_id="t1", user_input="q")
        passed, _, _ = evaluator.evaluate(case, [], "")
        assert passed


class TestForbiddenTools:
    def test_forbidden_called(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(forbidden_tools=["delete_data"]),
        )
        events = [{"type": "tool_call", "name": "delete_data", "arguments": {}}]
        passed, _, details = evaluator.evaluate(case, events, "")
        assert not passed, f"details={details}"

    def test_forbidden_not_called(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(forbidden_tools=["delete_data"]),
        )
        events = [{"type": "tool_call", "name": "web_search", "arguments": {}}]
        passed, _, _ = evaluator.evaluate(case, events, "")
        assert passed

    def test_empty_forbidden_tools(self, evaluator):
        case = EvalCase(case_id="t1", user_input="q")
        passed, _, _ = evaluator.evaluate(case, [], "")
        assert passed


class TestExpectedAgent:
    def test_agent_delegated(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(expected_agent="investment_analyst"),
        )
        events = [
            {
                "type": "classify",
                "delegation_plan": [
                    {"agent_id": "investment_analyst", "task": "analyze"},
                ],
            },
        ]
        passed, _, _ = evaluator.evaluate(case, events, "")
        assert passed

    def test_agent_not_delegated(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(expected_agent="investment_analyst"),
        )
        events = [
            {
                "type": "classify",
                "delegation_plan": [
                    {"agent_id": "commander", "task": "chat"},
                ],
            },
        ]
        passed, _, details = evaluator.evaluate(case, events, "")
        assert not passed, f"details={details}"

    def test_no_classify_event(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(expected_agent="investment_analyst"),
        )
        events = [{"type": "token", "content": "hi"}]
        passed, _, _ = evaluator.evaluate(case, events, "hi")
        assert not passed


class TestScore:
    def test_all_passed_score_1(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(must_include=["hello"]),
        )
        _, score, _ = evaluator.evaluate(case, [], "hello world")
        assert score == 1.0

    def test_partial_score(self, evaluator):
        case = EvalCase(
            case_id="t1", user_input="q",
            expected=ExpectedBehavior(
                must_include=["hello", "missing"],
                must_not_include=["error"],
            ),
        )
        _, score, _ = evaluator.evaluate(
            case, [], "hello world",
        )
        # 5 checks: outcome=True, must_include=False, must_not_include=True,
        #           required_tools=True (empty), forbidden_tools=True (empty)
        # 4/5 checks pass
        assert score == pytest.approx(4 / 5)