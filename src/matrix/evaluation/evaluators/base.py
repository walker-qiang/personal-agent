"""Evaluator base class and EvalResult."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..case import EvalCase


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
    returning (passed, score, details).
    """

    name: str = ""

    @abstractmethod
    def evaluate(
        self,
        case: EvalCase,
        events: list[dict[str, Any]],
        answer: str,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Evaluate a single case.

        Returns (passed, score, details).
        """
        ...