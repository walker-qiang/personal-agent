"""DeterministicEvaluator — zero-cost rule-based checks.

Evaluates agent output by comparing against expected behavior using
exact string matching and set operations. No LLM calls needed.
"""

from __future__ import annotations

from typing import Any

from .base import Evaluator
from ..case import EvalCase


class DeterministicEvaluator(Evaluator):
    """Checks: outcome, must_include, must_not_include, required_tools,
    forbidden_tools, expected_agent — all via deterministic rules."""

    name = "deterministic"

    def evaluate(
        self,
        case: EvalCase,
        events: list[dict[str, Any]],
        answer: str,
    ) -> tuple[bool, float, dict[str, Any]]:
        checks: dict[str, bool] = {}
        details: dict[str, Any] = {}

        # 1. Outcome check
        checks["outcome"] = self._check_outcome(case, events, details)

        # 2. Must-include keywords
        checks["must_include"] = self._check_must_include(case, answer, details)

        # 3. Must-not-include keywords
        checks["must_not_include"] = self._check_must_not_include(case, answer, details)

        # 4. Required tools called
        checks["required_tools"] = self._check_required_tools(case, events, details)

        # 5. Forbidden tools not called
        checks["forbidden_tools"] = self._check_forbidden_tools(case, events, details)

        # 6. Expected agent delegation
        if case.expected.expected_agent:
            checks["expected_agent"] = self._check_expected_agent(case, events, details)

        passed = all(checks.values())
        score = sum(1 for v in checks.values() if v) / max(len(checks), 1)
        details["checks"] = checks

        return (passed, score, details)

    # ---- individual checks ----

    def _check_outcome(self, case: EvalCase, events: list[dict], details: dict) -> bool:
        expected = case.expected.outcome
        has_error = any(e.get("type") == "error" for e in events)
        has_confirm = any(e.get("type") == "confirm_required" for e in events)

        if expected == "answer":
            ok = not has_error and not has_confirm
        elif expected == "abstain":
            ok = has_error  # agent should refuse
        elif expected == "tool_error":
            ok = has_error
        elif expected == "clarify":
            ok = True  # clarification is hard to detect deterministically
        else:
            ok = True

        details["outcome"] = {"expected": expected, "has_error": has_error, "has_confirm": has_confirm}
        return ok

    def _check_must_include(self, case: EvalCase, answer: str, details: dict) -> bool:
        if not case.expected.must_include:
            return True
        answer_lower = answer.lower()
        missing = [kw for kw in case.expected.must_include if kw.lower() not in answer_lower]
        details["must_include"] = {"missing": missing}
        return len(missing) == 0

    def _check_must_not_include(self, case: EvalCase, answer: str, details: dict) -> bool:
        if not case.expected.must_not_include:
            return True
        answer_lower = answer.lower()
        found = [kw for kw in case.expected.must_not_include if kw.lower() in answer_lower]
        details["must_not_include"] = {"found": found}
        return len(found) == 0

    def _check_required_tools(self, case: EvalCase, events: list[dict], details: dict) -> bool:
        if not case.expected.required_tools:
            return True
        called_tools = {e["name"] for e in events if e.get("type") == "tool_call" and "name" in e}
        required = set(case.expected.required_tools)
        missing = required - called_tools
        details["required_tools"] = {"required": list(required), "called": list(called_tools), "missing": list(missing)}
        return len(missing) == 0

    def _check_forbidden_tools(self, case: EvalCase, events: list[dict], details: dict) -> bool:
        if not case.expected.forbidden_tools:
            return True
        called_tools = {e["name"] for e in events if e.get("type") == "tool_call" and "name" in e}
        forbidden = set(case.expected.forbidden_tools)
        called_forbidden = forbidden & called_tools
        details["forbidden_tools"] = {"forbidden": list(forbidden), "called_forbidden": list(called_forbidden)}
        return len(called_forbidden) == 0

    def _check_expected_agent(self, case: EvalCase, events: list[dict], details: dict) -> bool:
        classify_events = [e for e in events if e.get("type") == "classify"]
        if not classify_events:
            return False
        plan = classify_events[0].get("delegation_plan", [])
        agent_ids = [item.get("agent_id", "") for item in plan]
        ok = case.expected.expected_agent in agent_ids
        details["expected_agent"] = {"expected": case.expected.expected_agent, "found": agent_ids}
        return ok