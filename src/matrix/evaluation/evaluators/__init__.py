"""Evaluator base classes and EvalResult."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result of evaluating a single EvalCase."""

    case_id: str
    passed: bool
    evaluator_results: dict[str, bool] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    session_id: str = ""
    elapsed_ms: float = 0.0
    token_count: int = 0


class Evaluator(ABC):
    """Base class for all evaluators.

    Each evaluator receives the full event stream and the final answer,
    and returns (passed, score, details).
    """

    name: str = ""

    @abstractmethod
    def evaluate(
        self,
        case: "EvalCase",  # noqa: F821
        events: list[dict[str, Any]],
        answer: str,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Evaluate a single case.

        Args:
            case: The EvalCase definition.
            events: All SSE events from stream_chat().
            answer: The final answer text (all token events concatenated).

        Returns:
            (passed, score, details) tuple.
        """
        ...