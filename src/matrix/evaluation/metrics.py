"""Metrics calculator for evaluation results.

Provides both a MetricsCalculator class (stateful, for advanced use) and a
standalone compute_metrics() function (stateless, for most use cases).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .case import EvalCase
from .evaluators.base import EvalResult


@dataclass
class MetricsSummary:
    """Aggregated metrics across all evaluation results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    avg_elapsed_ms: float = 0.0
    avg_token_count: float = 0.0
    evaluator_pass_rates: dict[str, float] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_risk: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_tag: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed_cases: list[str] = field(default_factory=list)


def compute_metrics(results: list[EvalResult], cases: list[EvalCase]) -> MetricsSummary:
    """Compute aggregated metrics from evaluation results.

    Slices by difficulty, risk, tag, and evaluator for granular analysis.
    """
    case_map = {c.case_id: c for c in cases}
    total = len(results)
    if total == 0:
        return MetricsSummary()

    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    # Average elapsed time and token count
    avg_elapsed = sum(r.elapsed_ms for r in results) / total
    avg_tokens = sum(r.token_count for r in results) / total

    # Per-evaluator pass rates
    eval_pass_rates: dict[str, float] = {}
    if results:
        evaluator_names = list(results[0].evaluator_results.keys())
        for name in evaluator_names:
            passed_count = sum(1 for r in results if r.evaluator_results.get(name, False))
            eval_pass_rates[name] = passed_count / total

    # By difficulty
    by_difficulty = _slice_by(results, case_map, "difficulty")

    # By risk
    by_risk = _slice_by(results, case_map, "risk")

    # By tag (a case can have multiple tags, each counted independently)
    by_tag = _slice_by_tag(results, case_map)

    # Failed cases
    failed_cases = [r.case_id for r in results if not r.passed]

    return MetricsSummary(
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=passed / total if total > 0 else 0.0,
        avg_elapsed_ms=round(avg_elapsed, 1),
        avg_token_count=round(avg_tokens, 1),
        evaluator_pass_rates=eval_pass_rates,
        by_difficulty=by_difficulty,
        by_risk=by_risk,
        by_tag=by_tag,
        failed_cases=failed_cases,
    )


def _slice_by(
    results: list[EvalResult],
    case_map: dict[str, EvalCase],
    attr: str,
) -> dict[str, dict[str, Any]]:
    """Slice results by a case attribute (difficulty, risk)."""
    buckets: dict[str, dict[str, Any]] = {}
    for r in results:
        case = case_map.get(r.case_id)
        if case:
            key = getattr(case, attr, "unknown")
            if key not in buckets:
                buckets[key] = {"total": 0, "passed": 0}
            buckets[key]["total"] += 1
            if r.passed:
                buckets[key]["passed"] += 1
    for key, stats in buckets.items():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 3) if stats["total"] > 0 else 0.0
    return buckets


def _slice_by_tag(
    results: list[EvalResult],
    case_map: dict[str, EvalCase],
) -> dict[str, dict[str, Any]]:
    """Slice results by tag (cases can have multiple tags)."""
    buckets: dict[str, dict[str, Any]] = {}
    for r in results:
        case = case_map.get(r.case_id)
        if case:
            for tag in case.tags:
                if tag not in buckets:
                    buckets[tag] = {"total": 0, "passed": 0}
                buckets[tag]["total"] += 1
                if r.passed:
                    buckets[tag]["passed"] += 1
    for tag, stats in buckets.items():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 3) if stats["total"] > 0 else 0.0
    return buckets


class MetricsCalculator:
    """Stateful metrics calculator (for advanced use where you want to
    accumulate results across multiple runs).

    For most use cases, prefer the standalone compute_metrics() function.
    """

    def __init__(self):
        self._all_results: list[EvalResult] = []
        self._all_cases: list[EvalCase] = []

    def add(self, results: list[EvalResult], cases: list[EvalCase]) -> None:
        """Accumulate results from a run."""
        self._all_results.extend(results)
        case_ids = {c.case_id for c in self._all_cases}
        for c in cases:
            if c.case_id not in case_ids:
                self._all_cases.append(c)

    def compute(self) -> MetricsSummary:
        """Compute metrics from all accumulated results."""
        return compute_metrics(self._all_results, self._all_cases)

    def reset(self) -> None:
        """Clear all accumulated results."""
        self._all_results = []
        self._all_cases = []