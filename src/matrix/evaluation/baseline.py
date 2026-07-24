"""Baseline management — load, compare, and update evaluation baselines.

Baselines are JSON files stored in tests/baselines/ that capture a known-good
state of evaluation results. They enable regression detection by comparing
current results against the baseline.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RegressionReport:
    """Result of comparing current evaluation against baseline."""

    has_regression: bool
    new_passes: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    still_failing: list[str] = field(default_factory=list)
    new_cases: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_console(self) -> str:
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  Regression Report")
        lines.append("=" * 60)
        lines.append("")

        if self.regressions:
            lines.append(f"  Regressions ({len(self.regressions)}):")
            for cid in self.regressions:
                lines.append(f"    ✗ {cid}")
            lines.append("")

        if self.new_passes:
            lines.append(f"  Improvements ({len(self.new_passes)}):")
            for cid in self.new_passes:
                lines.append(f"    ✓ {cid}")
            lines.append("")

        if self.still_failing:
            lines.append(f"  Still failing ({len(self.still_failing)}):")
            for cid in self.still_failing:
                lines.append(f"    ~ {cid}")
            lines.append("")

        if self.new_cases:
            lines.append(f"  New cases ({len(self.new_cases)}):")
            for cid in self.new_cases:
                lines.append(f"    + {cid}")
            lines.append("")

        lines.append("-" * 60)
        if self.has_regression:
            lines.append(f"  Result: REGRESSION DETECTED ({len(self.regressions)} case(s) degraded)")
        else:
            lines.append("  Result: No regression detected.")
        lines.append("=" * 60)

        return "\n".join(lines)


@dataclass
class QualityRegressionReport:
    """Result of comparing quality scores against baseline."""

    has_regression: bool
    avg_score_delta: float = 0.0
    dimension_deltas: dict[str, float] = field(default_factory=dict)
    case_regressions: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_console(self) -> str:
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  Quality Assessment Report")
        lines.append("=" * 60)
        lines.append("")

        # Overall score delta
        arrow = "↑" if self.avg_score_delta > 0 else ("↓" if self.avg_score_delta < 0 else "→")
        lines.append(f"  Average score delta: {arrow} {abs(self.avg_score_delta):.3f}")
        lines.append("")

        # Per-dimension deltas
        if self.dimension_deltas:
            lines.append("  Dimension deltas:")
            for dim, delta in self.dimension_deltas.items():
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                lines.append(f"    {dim:20s} {arrow} {abs(delta):.3f}")
            lines.append("")

        # Case-level regressions
        if self.case_regressions:
            lines.append(f"  Case regressions ({len(self.case_regressions)}):")
            for cr in self.case_regressions:
                cid = cr["case_id"]
                old = cr.get("baseline_score", 0)
                new = cr.get("current_score", 0)
                lines.append(f"    ✗ {cid}: {old:.3f} → {new:.3f} (Δ={new - old:.3f})")
            lines.append("")

        lines.append("-" * 60)
        if self.has_regression:
            lines.append("  Result: QUALITY REGRESSION DETECTED")
        else:
            lines.append("  Result: No quality regression detected.")
        lines.append("=" * 60)

        return "\n".join(lines)


def load_baseline(path: Path) -> dict[str, Any] | None:
    """Load a baseline JSON file. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_baseline(path: Path, data: dict[str, Any]) -> None:
    """Save a baseline JSON file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_git_commit() -> str:
    """Get current git commit hash (short)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_regression_baseline(
    results: list[Any],
    cases: list[Any],
) -> dict[str, Any]:
    """Build a regression baseline from EvalResults."""
    case_results: dict[str, dict[str, Any]] = {}
    for r in results:
        case_results[r.case_id] = {"passed": r.passed}

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    return {
        "version": datetime.now(timezone.utc).strftime("%Y-%m-%d-v%H%M"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
        },
        "case_results": case_results,
    }


def compare_regression(
    results: list[Any],
    baseline: dict[str, Any],
) -> RegressionReport:
    """Compare current results against regression baseline."""
    baseline_cases = baseline.get("case_results", {})

    new_passes: list[str] = []
    regressions: list[str] = []
    still_failing: list[str] = []
    new_cases: list[str] = []

    for r in results:
        baseline_entry = baseline_cases.get(r.case_id)
        if baseline_entry is None:
            new_cases.append(r.case_id)
            continue

        was_pass = baseline_entry.get("passed", False)
        if was_pass and not r.passed:
            regressions.append(r.case_id)
        elif not was_pass and r.passed:
            new_passes.append(r.case_id)
        elif not was_pass and not r.passed:
            still_failing.append(r.case_id)

    # Compute current summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    return RegressionReport(
        has_regression=len(regressions) > 0,
        new_passes=new_passes,
        regressions=regressions,
        still_failing=still_failing,
        new_cases=new_cases,
        summary={
            "current_pass_rate": passed / total if total > 0 else 0.0,
            "baseline_pass_rate": baseline.get("summary", {}).get("pass_rate", 0.0),
            "total": total,
            "passed": passed,
        },
    )


def build_quality_baseline(
    results: list[Any],
) -> dict[str, Any]:
    """Build a quality assessment baseline from EvalResults with LLM scores."""
    case_scores: dict[str, dict[str, Any]] = {}
    dimension_totals: dict[str, float] = {}
    overall_sum = 0.0
    count = 0

    for r in results:
        llm_details = r.details.get("llm_judge", {})
        dimensions = llm_details.get("dimensions", {})
        overall = llm_details.get("overall", 0.0)

        case_scores[r.case_id] = {
            "overall": overall,
            "dimensions": dimensions,
        }

        if overall > 0:
            overall_sum += overall
            count += 1
            for dim, val in dimensions.items():
                dimension_totals[dim] = dimension_totals.get(dim, 0.0) + val

    avg_overall = overall_sum / count if count > 0 else 0.0
    avg_dimensions = {k: v / count for k, v in dimension_totals.items()} if count > 0 else {}

    return {
        "version": datetime.now(timezone.utc).strftime("%Y-%m-%d-v%H%M"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "summary": {
            "total": len(results),
            "avg_quality_score": avg_overall,
            "dimensions": avg_dimensions,
        },
        "case_scores": case_scores,
    }


def compare_quality(
    results: list[Any],
    baseline: dict[str, Any],
    case_threshold: float = 0.15,
    avg_threshold: float = 0.05,
    dim_threshold: float = 0.10,
) -> QualityRegressionReport:
    """Compare current quality scores against baseline."""
    baseline_scores = baseline.get("case_scores", {})
    baseline_summary = baseline.get("summary", {})
    baseline_avg = baseline_summary.get("avg_quality_score", 0.0)
    baseline_dims = baseline_summary.get("dimensions", {})

    case_regressions: list[dict[str, Any]] = []
    current_dim_totals: dict[str, float] = {}
    overall_sum = 0.0
    count = 0

    for r in results:
        llm_details = r.details.get("llm_judge", {})
        current_overall = llm_details.get("overall", 0.0)
        current_dims = llm_details.get("dimensions", {})

        if current_overall > 0:
            overall_sum += current_overall
            count += 1
            for dim, val in current_dims.items():
                current_dim_totals[dim] = current_dim_totals.get(dim, 0.0) + val

        baseline_entry = baseline_scores.get(r.case_id)
        if baseline_entry:
            baseline_overall = baseline_entry.get("overall", 0.0)
            delta = current_overall - baseline_overall
            if delta < -case_threshold:
                case_regressions.append({
                    "case_id": r.case_id,
                    "baseline_score": baseline_overall,
                    "current_score": current_overall,
                    "delta": delta,
                })

    current_avg = overall_sum / count if count > 0 else 0.0
    avg_delta = current_avg - baseline_avg

    current_dims_avg = {k: v / count for k, v in current_dim_totals.items()} if count > 0 else {}
    dim_deltas = {}
    for dim, current_val in current_dims_avg.items():
        baseline_val = baseline_dims.get(dim, 0.0)
        dim_deltas[dim] = current_val - baseline_val

    has_regression = (
        len(case_regressions) > 0
        or avg_delta < -avg_threshold
        or any(d < -dim_threshold for d in dim_deltas.values())
    )

    return QualityRegressionReport(
        has_regression=has_regression,
        avg_score_delta=avg_delta,
        dimension_deltas=dim_deltas,
        case_regressions=case_regressions,
        summary={
            "current_avg": current_avg,
            "baseline_avg": baseline_avg,
            "current_dimensions": current_dims_avg,
            "baseline_dimensions": baseline_dims,
        },
    )
