"""Tests for baseline management — regression and quality baseline logic.

Covers:
- Dataset validation (all 20 cases load and parse correctly)
- Regression baseline: build, save, load, compare (no regression / regression / improvement / new cases)
- Quality baseline: build, save, load, compare (no regression / score drop / dimension regression)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from matrix.evaluation.baseline import (
    QualityRegressionReport,
    RegressionReport,
    build_quality_baseline,
    build_regression_baseline,
    compare_quality,
    compare_regression,
    load_baseline,
    save_baseline,
)
from matrix.evaluation.case import EvalCase
from matrix.evaluation.evaluators.base import EvalResult


# ---- Helpers ----------------------------------------------------------------

def _make_result(
    case_id: str,
    passed: bool,
    llm_overall: float = 0.0,
    llm_dims: dict | None = None,
) -> EvalResult:
    """Create a minimal EvalResult for baseline testing."""
    details = {}
    if llm_overall > 0 or llm_dims:
        details["llm_judge"] = {
            "overall": llm_overall,
            "dimensions": llm_dims or {
                "accuracy": llm_overall,
                "completeness": llm_overall,
                "relevance": llm_overall,
                "conciseness": llm_overall,
            },
        }
    return EvalResult(
        case_id=case_id,
        passed=passed,
        evaluator_results={"deterministic": passed},
        scores={"deterministic": 1.0 if passed else 0.0},
        details=details,
        answer="test answer",
        elapsed_ms=100,
    )


def _make_case(case_id: str, **kw) -> EvalCase:
    return EvalCase(case_id=case_id, user_input=f"test {case_id}", **kw)


# ---- Dataset validation -----------------------------------------------------

class TestDatasetValidation:
    """Verify the eval_dataset.json is valid and complete."""

    def test_dataset_loads(self):
        """Dataset file exists and is valid JSON."""
        dataset_path = Path(__file__).parent / "baselines" / "eval_dataset.json"
        if not dataset_path.exists():
            pytest.skip("eval_dataset.json not found")
        with open(dataset_path) as f:
            data = json.load(f)
        assert "cases" in data
        assert len(data["cases"]) == 20

    def test_all_cases_parse(self):
        """All 20 cases can be parsed into EvalCase objects."""
        dataset_path = Path(__file__).parent / "baselines" / "eval_dataset.json"
        if not dataset_path.exists():
            pytest.skip("eval_dataset.json not found")
        with open(dataset_path) as f:
            data = json.load(f)
        cases = [EvalCase.from_dict(c) for c in data["cases"]]
        assert len(cases) == 20
        # Verify case IDs are unique
        ids = {c.case_id for c in cases}
        assert len(ids) == 20

    def test_case_id_prefixes(self):
        """Case IDs follow naming conventions by category."""
        dataset_path = Path(__file__).parent / "baselines" / "eval_dataset.json"
        if not dataset_path.exists():
            pytest.skip("eval_dataset.json not found")
        with open(dataset_path) as f:
            data = json.load(f)
        prefixes = ["conv_", "finance_", "web_", "media_", "multi_", "edge_"]
        for case in data["cases"]:
            assert any(case["case_id"].startswith(p) for p in prefixes), \
                f"Case {case['case_id']} doesn't match any prefix"

    def test_all_outcomes_are_answer(self):
        """All cases expect outcome=answer (no abstain/tool_error in dataset)."""
        dataset_path = Path(__file__).parent / "baselines" / "eval_dataset.json"
        if not dataset_path.exists():
            pytest.skip("eval_dataset.json not found")
        with open(dataset_path) as f:
            data = json.load(f)
        for case in data["cases"]:
            assert case["expected"]["outcome"] == "answer"

    def test_difficulty_distribution(self):
        """Dataset has a mix of difficulty levels."""
        dataset_path = Path(__file__).parent / "baselines" / "eval_dataset.json"
        if not dataset_path.exists():
            pytest.skip("eval_dataset.json not found")
        with open(dataset_path) as f:
            data = json.load(f)
        difficulties = {c["difficulty"] for c in data["cases"]}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties


# ---- Regression baseline ----------------------------------------------------

class TestRegressionBaseline:
    """Tests for build_regression_baseline and compare_regression."""

    def test_build_baseline_all_pass(self):
        """Build baseline when all cases pass."""
        results = [
            _make_result("c1", True),
            _make_result("c2", True),
            _make_result("c3", True),
        ]
        cases = [_make_case("c1"), _make_case("c2"), _make_case("c3")]
        baseline = build_regression_baseline(results, cases)

        assert baseline["summary"]["total"] == 3
        assert baseline["summary"]["passed"] == 3
        assert baseline["summary"]["failed"] == 0
        assert baseline["summary"]["pass_rate"] == 1.0
        assert baseline["case_results"]["c1"]["passed"] is True
        assert baseline["case_results"]["c2"]["passed"] is True
        assert baseline["case_results"]["c3"]["passed"] is True

    def test_build_baseline_mixed(self):
        """Build baseline with mixed pass/fail."""
        results = [
            _make_result("c1", True),
            _make_result("c2", False),
        ]
        cases = [_make_case("c1"), _make_case("c2")]
        baseline = build_regression_baseline(results, cases)

        assert baseline["summary"]["passed"] == 1
        assert baseline["summary"]["failed"] == 1
        assert baseline["summary"]["pass_rate"] == 0.5
        assert baseline["case_results"]["c2"]["passed"] is False

    def test_compare_no_regression(self):
        """No regression when pass/fail status is unchanged."""
        results = [_make_result("c1", True), _make_result("c2", True)]
        baseline = {
            "summary": {"pass_rate": 1.0},
            "case_results": {
                "c1": {"passed": True},
                "c2": {"passed": True},
            },
        }
        report = compare_regression(results, baseline)
        assert not report.has_regression
        assert len(report.regressions) == 0
        assert len(report.new_passes) == 0
        assert len(report.still_failing) == 0
        assert len(report.new_cases) == 0

    def test_compare_with_regression(self):
        """Regression detected when a previously passing case now fails."""
        results = [_make_result("c1", False), _make_result("c2", True)]
        baseline = {
            "summary": {"pass_rate": 1.0},
            "case_results": {
                "c1": {"passed": True},
                "c2": {"passed": True},
            },
        }
        report = compare_regression(results, baseline)
        assert report.has_regression
        assert "c1" in report.regressions
        assert len(report.regressions) == 1

    def test_compare_with_improvement(self):
        """Improvement detected when a previously failing case now passes."""
        results = [_make_result("c1", True), _make_result("c2", True)]
        baseline = {
            "summary": {"pass_rate": 0.5},
            "case_results": {
                "c1": {"passed": False},
                "c2": {"passed": True},
            },
        }
        report = compare_regression(results, baseline)
        assert not report.has_regression
        assert "c1" in report.new_passes
        assert len(report.new_passes) == 1

    def test_compare_still_failing(self):
        """Still failing cases are tracked separately."""
        results = [_make_result("c1", False), _make_result("c2", True)]
        baseline = {
            "summary": {"pass_rate": 0.5},
            "case_results": {
                "c1": {"passed": False},
                "c2": {"passed": True},
            },
        }
        report = compare_regression(results, baseline)
        assert not report.has_regression
        assert "c1" in report.still_failing
        assert len(report.still_failing) == 1

    def test_compare_new_cases(self):
        """Cases not in baseline are reported as new."""
        results = [
            _make_result("c1", True),
            _make_result("c2", True),
            _make_result("c3", True),
        ]
        baseline = {
            "summary": {"pass_rate": 1.0},
            "case_results": {
                "c1": {"passed": True},
                "c2": {"passed": True},
            },
        }
        report = compare_regression(results, baseline)
        assert not report.has_regression
        assert "c3" in report.new_cases
        assert len(report.new_cases) == 1

    def test_compare_mixed_scenario(self):
        """Mixed scenario: regression + improvement + new case + still failing."""
        results = [
            _make_result("c1", False),  # regression (was pass)
            _make_result("c2", True),   # improvement (was fail)
            _make_result("c3", False),  # still failing
            _make_result("c4", True),   # new case
        ]
        baseline = {
            "summary": {"pass_rate": 0.33},
            "case_results": {
                "c1": {"passed": True},
                "c2": {"passed": False},
                "c3": {"passed": False},
            },
        }
        report = compare_regression(results, baseline)
        assert report.has_regression
        assert "c1" in report.regressions
        assert "c2" in report.new_passes
        assert "c3" in report.still_failing
        assert "c4" in report.new_cases

    def test_regression_report_to_console(self):
        """Console output contains key sections."""
        report = RegressionReport(
            has_regression=True,
            regressions=["case_a"],
            new_passes=["case_b"],
            still_failing=["case_c"],
            new_cases=["case_d"],
            summary={"current_pass_rate": 0.8, "baseline_pass_rate": 0.9},
        )
        output = report.to_console()
        assert "Regression Report" in output
        assert "case_a" in output
        assert "case_b" in output
        assert "REGRESSION DETECTED" in output


# ---- Quality baseline -------------------------------------------------------

class TestQualityBaseline:
    """Tests for build_quality_baseline and compare_quality."""

    def test_build_quality_baseline(self):
        """Build quality baseline from results with LLM scores."""
        results = [
            _make_result("c1", True, llm_overall=0.85, llm_dims={
                "accuracy": 0.9, "completeness": 0.8, "relevance": 0.85, "conciseness": 0.8,
            }),
            _make_result("c2", True, llm_overall=0.75, llm_dims={
                "accuracy": 0.8, "completeness": 0.7, "relevance": 0.75, "conciseness": 0.7,
            }),
        ]
        baseline = build_quality_baseline(results)

        assert baseline["summary"]["total"] == 2
        assert baseline["summary"]["avg_quality_score"] == pytest.approx(0.80, abs=0.01)
        dims = baseline["summary"]["dimensions"]
        assert "accuracy" in dims
        assert dims["accuracy"] == pytest.approx(0.85, abs=0.01)

    def test_build_quality_baseline_no_llm_scores(self):
        """Build quality baseline when no LLM scores are present."""
        results = [_make_result("c1", True), _make_result("c2", True)]
        baseline = build_quality_baseline(results)

        assert baseline["summary"]["avg_quality_score"] == 0.0
        assert baseline["summary"]["dimensions"] == {}

    def test_compare_quality_no_regression(self):
        """No quality regression when scores are within thresholds."""
        results = [
            _make_result("c1", True, llm_overall=0.85),
            _make_result("c2", True, llm_overall=0.80),
        ]
        baseline = {
            "summary": {
                "avg_quality_score": 0.82,
                "dimensions": {"accuracy": 0.85, "completeness": 0.80},
            },
            "case_scores": {
                "c1": {"overall": 0.83, "dimensions": {}},
                "c2": {"overall": 0.81, "dimensions": {}},
            },
        }
        report = compare_quality(results, baseline)
        assert not report.has_regression
        assert len(report.case_regressions) == 0

    def test_compare_quality_case_regression(self):
        """Quality regression when a case score drops beyond threshold."""
        results = [
            _make_result("c1", True, llm_overall=0.60),  # was 0.85, dropped 0.25
            _make_result("c2", True, llm_overall=0.80),
        ]
        baseline = {
            "summary": {
                "avg_quality_score": 0.82,
                "dimensions": {},
            },
            "case_scores": {
                "c1": {"overall": 0.85, "dimensions": {}},
                "c2": {"overall": 0.80, "dimensions": {}},
            },
        }
        report = compare_quality(results, baseline, case_threshold=0.15)
        assert report.has_regression
        assert len(report.case_regressions) == 1
        assert report.case_regressions[0]["case_id"] == "c1"

    def test_compare_quality_avg_regression(self):
        """Quality regression when average score drops beyond threshold."""
        results = [
            _make_result("c1", True, llm_overall=0.70),
            _make_result("c2", True, llm_overall=0.72),
        ]
        baseline = {
            "summary": {
                "avg_quality_score": 0.85,
                "dimensions": {},
            },
            "case_scores": {
                "c1": {"overall": 0.85, "dimensions": {}},
                "c2": {"overall": 0.85, "dimensions": {}},
            },
        }
        report = compare_quality(results, baseline, avg_threshold=0.05)
        assert report.has_regression
        assert report.avg_score_delta < -0.05

    def test_compare_quality_dimension_regression(self):
        """Quality regression when a dimension score drops beyond threshold."""
        results = [
            _make_result("c1", True, llm_overall=0.80, llm_dims={
                "accuracy": 0.60, "completeness": 0.80, "relevance": 0.80, "conciseness": 0.80,
            }),
        ]
        baseline = {
            "summary": {
                "avg_quality_score": 0.80,
                "dimensions": {"accuracy": 0.85, "completeness": 0.80, "relevance": 0.80, "conciseness": 0.80},
            },
            "case_scores": {
                "c1": {"overall": 0.80, "dimensions": {"accuracy": 0.85}},
            },
        }
        report = compare_quality(results, baseline, dim_threshold=0.10)
        assert report.has_regression
        assert "accuracy" in report.dimension_deltas
        assert report.dimension_deltas["accuracy"] < -0.10

    def test_quality_report_to_console(self):
        """Console output contains key sections."""
        report = QualityRegressionReport(
            has_regression=True,
            avg_score_delta=-0.15,
            dimension_deltas={"accuracy": -0.20, "completeness": 0.05},
            case_regressions=[
                {"case_id": "case_x", "baseline_score": 0.85, "current_score": 0.60, "delta": -0.25},
            ],
        )
        output = report.to_console()
        assert "Quality Assessment Report" in output
        assert "QUALITY REGRESSION DETECTED" in output
        assert "case_x" in output
        assert "accuracy" in output


# ---- Save / Load roundtrip --------------------------------------------------

class TestBaselineSaveLoad:
    """Tests for save_baseline and load_baseline."""

    def test_save_and_load_regression(self, tmp_path):
        """Regression baseline can be saved and loaded."""
        results = [_make_result("c1", True), _make_result("c2", False)]
        cases = [_make_case("c1"), _make_case("c2")]
        baseline = build_regression_baseline(results, cases)

        path = tmp_path / "regression_baseline.json"
        save_baseline(path, baseline)
        assert path.exists()

        loaded = load_baseline(path)
        assert loaded is not None
        assert loaded["summary"]["total"] == 2
        assert loaded["case_results"]["c1"]["passed"] is True
        assert loaded["case_results"]["c2"]["passed"] is False

    def test_save_and_load_quality(self, tmp_path):
        """Quality baseline can be saved and loaded."""
        results = [
            _make_result("c1", True, llm_overall=0.85, llm_dims={
                "accuracy": 0.9, "completeness": 0.8, "relevance": 0.85, "conciseness": 0.8,
            }),
        ]
        baseline = build_quality_baseline(results)

        path = tmp_path / "quality_baseline.json"
        save_baseline(path, baseline)
        assert path.exists()

        loaded = load_baseline(path)
        assert loaded is not None
        assert loaded["summary"]["avg_quality_score"] == pytest.approx(0.85, abs=0.01)

    def test_load_nonexistent_returns_none(self, tmp_path):
        """Loading a non-existent file returns None."""
        path = tmp_path / "nonexistent.json"
        assert load_baseline(path) is None

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_baseline creates parent directories."""
        results = [_make_result("c1", True)]
        cases = [_make_case("c1")]
        baseline = build_regression_baseline(results, cases)

        path = tmp_path / "nested" / "dir" / "baseline.json"
        save_baseline(path, baseline)
        assert path.exists()


# ---- Git commit mock --------------------------------------------------------

class TestGitCommit:
    """Test that git commit is captured in baseline."""

    def test_git_commit_in_baseline(self):
        """Baseline includes git commit hash."""
        with patch("matrix.evaluation.baseline.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "abc1234\n"
            mock_run.return_value.returncode = 0

            results = [_make_result("c1", True)]
            cases = [_make_case("c1")]
            baseline = build_regression_baseline(results, cases)

            assert baseline["git_commit"] == "abc1234"

    def test_git_commit_fallback(self):
        """Baseline uses 'unknown' when git is not available."""
        from matrix.evaluation.baseline import get_git_commit
        with patch("matrix.evaluation.baseline.subprocess.run", side_effect=FileNotFoundError):
            commit = get_git_commit()
            assert commit == "unknown"
