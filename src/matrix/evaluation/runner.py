"""EvalRunner — runs EvalCases through ChatService and collects results.

Design:
- Creates a fresh session_id per case to avoid cross-contamination.
- Collects all events from stream_chat(), extracts answer from token events.
- Runs the evaluator chain (ordered list of Evaluators).
- Returns EvalResult per case with full details for debugging.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .case import EvalCase
from .evaluators.base import EvalResult, Evaluator


class EvalRunner:
    """Runs evaluation cases against a ChatService.

    Usage:
        runner = EvalRunner(chat_service, [DeterministicEvaluator()])
        results = runner.run(cases)
    """

    def __init__(self, chat_service: Any, evaluators: list[Evaluator] | None = None):
        self._chat = chat_service
        self._evaluators = evaluators or []

    def run(self, cases: list[EvalCase]) -> list[EvalResult]:
        """Run all cases sequentially and return results."""
        results: list[EvalResult] = []
        for case in cases:
            result = self.run_one(case)
            results.append(result)
        return results

    def run_one(self, case: EvalCase) -> EvalResult:
        """Run a single case end-to-end.

        Steps:
        1. Call stream_chat() with the case's user_input
        2. Collect all events (token, tool_call, tool_result, error, etc.)
        3. Extract the final answer from token events
        4. Run each evaluator in the chain
        5. Return EvalResult
        """
        started = time.perf_counter()
        session_id = uuid.uuid4().hex[:12]

        events: list[dict[str, Any]] = []
        answer_parts: list[str] = []
        token_count = 0

        try:
            for event in self._chat.stream_chat(
                case.user_input,
                session_id=session_id,
                user_id=case.user_id,
            ):
                events.append(event)
                if event.get("type") == "token":
                    content = event.get("content", "")
                    answer_parts.append(content)
                    token_count += 1
        except Exception as exc:
            events.append({"type": "error", "message": str(exc)})

        answer = "".join(answer_parts)
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        # Run evaluator chain
        evaluator_results: dict[str, bool] = {}
        scores: dict[str, float] = {}
        all_details: dict[str, Any] = {}
        all_passed = True

        for evaluator in self._evaluators:
            passed, score, details = evaluator.evaluate(case, events, answer)
            evaluator_results[evaluator.name] = passed
            scores[evaluator.name] = score
            all_details[evaluator.name] = details
            if not passed:
                all_passed = False

        return EvalResult(
            case_id=case.case_id,
            passed=all_passed,
            evaluator_results=evaluator_results,
            scores=scores,
            details=all_details,
            answer=answer,
            session_id=session_id,
            elapsed_ms=elapsed_ms,
            token_count=token_count,
        )