"""Reporter — formats evaluation results for human or machine consumption.

Supports two output formats:
- console: color-coded terminal output with pass/fail details
- json: machine-readable JSON for CI/CD pipelines
"""

from __future__ import annotations

import json
from typing import Any, Literal

from .case import EvalCase
from .evaluators.base import EvalResult
from .metrics import MetricsSummary

ReportFormat = Literal["json", "console"]


class Reporter:
    """Formats evaluation results into reports.

    Usage:
        reporter = Reporter("console")
        print(reporter.generate(results, summary))
    """

    def __init__(self, format: ReportFormat = "console"):
        self._format = format

    def generate(self, results: list[EvalResult], summary: MetricsSummary) -> str:
        """Generate a report string from results and summary."""
        if self._format == "json":
            return self._json_report(results, summary)
        return self._console_report(results, summary)

    # ---- JSON ----

    def _json_report(self, results: list[EvalResult], summary: MetricsSummary) -> str:
        data = {
            "summary": {
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "pass_rate": summary.pass_rate,
                "avg_elapsed_ms": summary.avg_elapsed_ms,
                "avg_token_count": summary.avg_token_count,
                "evaluator_pass_rates": summary.evaluator_pass_rates,
                "by_difficulty": summary.by_difficulty,
                "by_risk": summary.by_risk,
                "by_tag": summary.by_tag,
                "failed_cases": summary.failed_cases,
            },
            "results": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "evaluator_results": r.evaluator_results,
                    "scores": r.scores,
                    "details": _serialize_details(r.details),
                    "answer": r.answer,
                    "session_id": r.session_id,
                    "elapsed_ms": r.elapsed_ms,
                    "token_count": r.token_count,
                }
                for r in results
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    # ---- Console ----

    def _console_report(self, results: list[EvalResult], summary: MetricsSummary) -> str:
        lines: list[str] = []

        # Header
        lines.append("=" * 60)
        lines.append("  Evaluation Report")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append("── Summary ──")
        lines.append(f"  Total:       {summary.total}")
        lines.append(f"  Passed:      {summary.passed}  \033[32m✓\033[0m")
        if summary.failed > 0:
            lines.append(f"  Failed:      {summary.failed}  \033[31m✗\033[0m")
        else:
            lines.append(f"  Failed:      {summary.failed}")
        lines.append(f"  Pass Rate:   {summary.pass_rate:.1%}")
        lines.append(f"  Avg Time:    {summary.avg_elapsed_ms:.0f}ms")
        lines.append(f"  Avg Tokens:  {summary.avg_token_count:.0f}")
        lines.append("")

        # Per-evaluator pass rates
        if summary.evaluator_pass_rates:
            lines.append("── Evaluator Pass Rates ──")
            for name, rate in summary.evaluator_pass_rates.items():
                icon = "\033[32m✓\033[0m" if rate == 1.0 else "\033[31m✗\033[0m"
                lines.append(f"  {name:20s} {rate:.1%}  {icon}")
            lines.append("")

        # Per-case results
        lines.append("── Case Results ──")
        for r in results:
            status = "\033[32mPASS\033[0m" if r.passed else "\033[31mFAIL\033[0m"
            lines.append(f"  [{status}] {r.case_id}  ({r.elapsed_ms}ms)")

            # Show failed checks
            if not r.passed:
                for eval_name, eval_result in r.evaluator_results.items():
                    if not eval_result:
                        details = r.details.get(eval_name, {})
                        checks = details.get("checks", {})
                        failed_checks = [k for k, v in checks.items() if not v]
                        lines.append(f"         {eval_name}: failed checks: {', '.join(failed_checks)}")
                        # Show specifics
                        for check_name in failed_checks:
                            check_detail = details.get(check_name)
                            if check_detail:
                                lines.append(f"           {check_name}: {_format_detail(check_name, check_detail)}")
            lines.append("")

        # By difficulty
        if summary.by_difficulty:
            lines.append("── By Difficulty ──")
            for diff, stats in sorted(summary.by_difficulty.items()):
                lines.append(f"  {diff:10s} {stats['passed']}/{stats['total']}  ({stats['pass_rate']:.1%})")
            lines.append("")

        # By risk
        if summary.by_risk:
            lines.append("── By Risk ──")
            for risk in ["critical", "high", "medium", "low"]:
                if risk in summary.by_risk:
                    stats = summary.by_risk[risk]
                    lines.append(f"  {risk:10s} {stats['passed']}/{stats['total']}  ({stats['pass_rate']:.1%})")
            lines.append("")

        # Footer
        lines.append("=" * 60)
        if summary.failed == 0:
            lines.append("  All cases passed! \033[32m✓\033[0m")
        else:
            lines.append(f"  {summary.failed} case(s) failed. Review details above.")
        lines.append("=" * 60)

        return "\n".join(lines)


def _serialize_details(details: dict[str, Any]) -> dict[str, Any]:
    """Convert details to JSON-serializable form (sets → lists)."""
    result: dict[str, Any] = {}
    for k, v in details.items():
        if isinstance(v, dict):
            result[k] = _serialize_dict(v)
        else:
            result[k] = v
    return result


def _serialize_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, set):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = _serialize_dict(v)
        else:
            out[k] = v
    return out


def _format_detail(check_name: str, detail: Any) -> str:
    """Format a check detail for console display."""
    if isinstance(detail, dict):
        parts = []
        for k, v in detail.items():
            if isinstance(v, list) and v:
                parts.append(f"{k}={v}")
            elif isinstance(v, bool):
                parts.append(f"{k}={v}")
            elif v:
                parts.append(f"{k}={v}")
        return ", ".join(parts)
    return str(detail)