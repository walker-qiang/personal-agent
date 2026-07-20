"""Agent evaluation framework — EvalCase → EvalRunner → Evaluator chain → Metrics → Report.

All evaluators operate on events collected from ChatService.stream_chat() and
the final answer text. The evaluation module does NOT modify any agent logic.
"""

from .case import Difficulty, EvalCase, ExpectedBehavior, Outcome
from .evaluators.base import EvalResult, Evaluator
from .evaluators.deterministic import DeterministicEvaluator
from .metrics import MetricsCalculator, compute_metrics
from .reporter import ReportFormat, Reporter
from .runner import EvalRunner

__all__ = [
    "Difficulty",
    "EvalCase",
    "EvalResult",
    "EvalRunner",
    "Evaluator",
    "ExpectedBehavior",
    "DeterministicEvaluator",
    "MetricsCalculator",
    "ReportFormat",
    "Reporter",
    "Outcome",
    "compute_metrics",
]