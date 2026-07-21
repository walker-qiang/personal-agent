"""LLMEvaluator — LLM-as-Judge for quality evaluation.

Evaluates agent output quality using a pipeline LLM across four dimensions:
- accuracy: Are the facts correct and consistent with the reference?
- completeness: Does the answer cover all aspects of the question?
- relevance: Is the answer directly addressing the user's question?
- conciseness: Is the answer appropriately concise without fluff?

Unlike DeterministicEvaluator (free, fast), this evaluator makes real LLM calls
and should be used for quality assessment, not CI gating.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import Evaluator
from ..case import EvalCase

logger = logging.getLogger("matrix.evaluation")


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI agent responses. Your task is to judge the quality of an agent's answer to a user's question.

You will receive:
1. The user's question
2. The agent's answer
3. Optionally, a reference answer (what the answer should contain)

Rate the answer on these four dimensions on a scale of 0.0 to 1.0:

- **accuracy** (0.0-1.0): Are the facts correct? Does the answer contain any hallucinations or fabricated information? If the answer says "I don't know" or "I couldn't find", that is accurate (not fabricated), but lowers completeness.

- **completeness** (0.0-1.0): Does the answer cover all aspects of the user's question? Are all key points addressed? If the user asked for multiple things, are all of them answered?

- **relevance** (0.0-1.0): Is the answer directly addressing the user's question? Or is it off-topic, rambling, or including irrelevant information?

- **conciseness** (0.0-1.0): Is the answer appropriately concise? Does it avoid unnecessary fluff, repetition, or excessive detail? A long answer is fine if all content is valuable.

Output ONLY a JSON object with this exact structure:
{
  "accuracy": 0.85,
  "completeness": 0.90,
  "relevance": 0.95,
  "conciseness": 0.80,
  "overall": 0.88,
  "reasoning": "Brief explanation of your ratings (1-2 sentences per dimension)"
}

The overall score should be a weighted average: accuracy × 0.35 + completeness × 0.30 + relevance × 0.20 + conciseness × 0.15.
"""


class LLMEvaluator(Evaluator):
    """LLM-as-Judge evaluator for answer quality assessment.

    Uses a pipeline LLM to evaluate the agent's answer against the
    expected behavior. This is more expensive (LLM calls) but provides
    nuanced quality assessment that deterministic rules cannot capture.

    Usage:
        eval = LLMEvaluator(pipeline_llm)
        passed, score, details = eval.evaluate(case, events, answer)
    """

    name = "llm_judge"

    def __init__(self, llm: Any) -> None:
        """Initialize with a pipeline LLM client.

        Args:
            llm: An LLMClient instance (typically the pipeline LLM).
        """
        self._llm = llm

    def evaluate(
        self,
        case: EvalCase,
        events: list[dict[str, Any]],
        answer: str,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Evaluate the answer quality using LLM judgment.

        Returns (passed, score, details) where passed=True means overall >= 0.7.
        """
        if not answer or answer.strip() == "":
            return (False, 0.0, {"error": "empty_answer", "dimensions": {}})

        try:
            user_prompt = self._build_judge_prompt(case, answer)
            scores = self._llm.complete_json(
                JUDGE_SYSTEM_PROMPT,
                [{"role": "user", "content": user_prompt}],
                temperature=0.1,
            )
        except Exception as e:
            logger.warning("LLMEvaluator: LLM call failed: %s", e)
            return (False, 0.0, {"error": f"LLM call failed: {e}", "dimensions": {}})

        overall = scores.get("overall", 0.0)
        passed = overall >= 0.7

        details = {
            "dimensions": {
                "accuracy": scores.get("accuracy", 0.0),
                "completeness": scores.get("completeness", 0.0),
                "relevance": scores.get("relevance", 0.0),
                "conciseness": scores.get("conciseness", 0.0),
            },
            "overall": overall,
            "reasoning": scores.get("reasoning", ""),
        }

        return (passed, overall, details)

    def _build_judge_prompt(self, case: EvalCase, answer: str) -> str:
        """Build the judge prompt with question, answer, and optional reference."""
        parts = [
            f"## User Question\n{case.user_input}",
            f"\n## Agent Answer\n{answer}",
        ]

        # Include reference info from expected behavior
        ref_parts = []
        if case.expected.must_include:
            ref_parts.append(f"Should include: {', '.join(case.expected.must_include)}")
        if case.expected.must_not_include:
            ref_parts.append(f"Should NOT include: {', '.join(case.expected.must_not_include)}")
        if case.expected.required_tools:
            ref_parts.append(f"Should use tools: {', '.join(case.expected.required_tools)}")

        if ref_parts:
            parts.append(f"\n## Reference Criteria\n" + "\n".join(ref_parts))

        parts.append("\nPlease evaluate the answer and return the JSON scores.")
        return "\n".join(parts)

    @staticmethod
    def _parse_judge_result(raw: str) -> dict[str, Any]:
        """Parse the LLM judge output into a scores dict."""
        try:
            cleaned = raw.strip()
            # Extract from markdown fence if present
            if cleaned.startswith("```"):
                import re
                fence_match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```",
                    cleaned, flags=re.DOTALL,
                )
                if fence_match:
                    cleaned = fence_match.group(1)
            scores = json.loads(cleaned)

            # Validate required fields
            required = ["accuracy", "completeness", "relevance", "conciseness"]
            for field in required:
                if field not in scores:
                    scores[field] = 0.0
                scores[field] = float(scores[field])

            if "overall" not in scores:
                scores["overall"] = round(
                    scores["accuracy"] * 0.35
                    + scores["completeness"] * 0.30
                    + scores["relevance"] * 0.20
                    + scores["conciseness"] * 0.15,
                    2,
                )

            return scores
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("LLMEvaluator: parse failed: %s, raw=%s...", e, raw[:100])
            return {
                "accuracy": 0.0,
                "completeness": 0.0,
                "relevance": 0.0,
                "conciseness": 0.0,
                "overall": 0.0,
                "reasoning": f"Parse error: {e}",
            }