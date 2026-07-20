"""Integration tests for the evaluation framework — full pipeline:
EvalCase → EvalRunner → DeterministicEvaluator → Metrics → Reporter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from matrix.chat import ChatService
from matrix.config import AgentConfig
from matrix.evaluation import (
    DeterministicEvaluator,
    EvalCase,
    EvalResult,
    EvalRunner,
    ExpectedBehavior,
    MetricsCalculator,
    Reporter,
    compute_metrics,
)
from matrix.evaluation.case import Difficulty, Outcome, Risk
from matrix.tools import ToolRegistry
from matrix.tools.finance import register_all


class FakeLLM:
    """Fake LLM that returns predefined responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[tuple[str, list[dict]]] = []
        self.provider = "test"
        self.model = "test-model"

    def complete(self, system: str, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append(("complete", messages))
        if not self.responses:
            return ""
        return self.responses.pop(0)

    def stream_complete(self, system: str, messages: list[dict[str, str]], **kwargs):
        self.calls.append(("stream", messages))
        text = self.responses.pop(0) if self.responses else ""
        for ch in text:
            yield ch

    def function_call(self, system, messages, tools, tool_choice="auto", **kwargs):
        from matrix.llm import FunctionCallResult
        self.calls.append(("function_call", messages))
        text = self.responses.pop(0) if self.responses else ""
        return FunctionCallResult(content=text, tool_calls=[])


@pytest.fixture
def eval_chat_service(tmp_cache_path: Path) -> ChatService:
    """Create a ChatService with fake LLM for evaluation testing."""
    config = AgentConfig(
        root_path=tmp_cache_path.parent,
        cache_path=tmp_cache_path,
        trace_path=tmp_cache_path.parent / "trace.jsonl",
        store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
        checkpoint_path=str(tmp_cache_path.parent / "var" / "agent" / "checkpoints.db"),
        skills_base_dir=tmp_cache_path.parent / "skills",
        host="127.0.0.1",
        port=0,
        deepseek_api_key="test-key",
    )
    registry = ToolRegistry()
    register_all(registry, tmp_cache_path)
    from matrix.tools.web import register_all as register_web
    from matrix.tools.agnes import register_all as register_agnes
    register_web(registry)
    register_agnes(registry)

    service = ChatService(config, registry)
    # Override LLMs with fakes
    service._default_llm = FakeLLM(["当前持仓健康，共3个持仓。"])
    service._pipeline_llm = FakeLLM(["[]"])
    return service


class TestEvalCase:
    def test_from_dict_minimal(self):
        data = {"case_id": "c1", "user_input": "hello"}
        case = EvalCase.from_dict(data)
        assert case.case_id == "c1"
        assert case.user_input == "hello"
        assert case.expected.outcome == "answer"
        assert case.difficulty == "easy"

    def test_from_dict_full(self):
        data = {
            "case_id": "c2",
            "user_input": "search AI",
            "expected": {
                "outcome": "answer",
                "must_include": ["AI"],
                "must_not_include": ["error"],
                "required_tools": ["web_search"],
                "forbidden_tools": ["delete_data"],
                "expected_agent": "commander",
                "min_evidence": 2,
            },
            "tags": ["search"],
            "difficulty": "medium",
            "risk": "high",
        }
        case = EvalCase.from_dict(data)
        assert case.expected.must_include == ["AI"]
        assert case.expected.must_not_include == ["error"]
        assert case.expected.required_tools == ["web_search"]
        assert case.expected.forbidden_tools == ["delete_data"]
        assert case.expected.expected_agent == "commander"
        assert case.expected.min_evidence == 2
        assert case.tags == ["search"]
        assert case.difficulty == "medium"
        assert case.risk == "high"

    def test_to_dict_roundtrip(self):
        original = EvalCase(
            case_id="c1",
            user_input="test",
            expected=ExpectedBehavior(
                must_include=["hello"],
                required_tools=["web_search"],
            ),
            tags=["tag1", "tag2"],
            difficulty="hard",
            risk="critical",
        )
        data = original.to_dict()
        restored = EvalCase.from_dict(data)
        assert restored.case_id == original.case_id
        assert restored.expected.must_include == original.expected.must_include
        assert restored.expected.required_tools == original.expected.required_tools
        assert restored.tags == original.tags

    def test_load_smoke_dataset(self):
        """Verify smoke.json is valid and can be loaded."""
        import os
        smoke_path = Path(__file__).parent.parent / "src" / "matrix" / "evaluation" / "datasets" / "smoke.json"
        if not smoke_path.exists():
            pytest.skip("smoke.json not found")
        with open(smoke_path) as f:
            data = json.load(f)
        cases = [EvalCase.from_dict(c) for c in data["cases"]]
        assert len(cases) == 5
        assert cases[0].case_id == "smoke_greeting"
        assert cases[2].expected.required_tools == ["web_search"]


class TestEvalRunner:
    def test_run_single_case(self, eval_chat_service):
        runner = EvalRunner(eval_chat_service, [DeterministicEvaluator()])
        case = EvalCase(
            case_id="greeting",
            user_input="你好",
            expected=ExpectedBehavior(must_not_include=["error"]),
        )
        results = runner.run([case])
        assert len(results) == 1
        assert results[0].case_id == "greeting"
        assert results[0].passed
        assert "deterministic" in results[0].evaluator_results
        assert results[0].elapsed_ms > 0

    def test_run_multiple_cases(self, eval_chat_service):
        runner = EvalRunner(eval_chat_service, [DeterministicEvaluator()])
        cases = [
            EvalCase(case_id="c1", user_input="a", expected=ExpectedBehavior(must_not_include=["error"])),
            EvalCase(case_id="c2", user_input="b", expected=ExpectedBehavior(must_not_include=["error"])),
            EvalCase(case_id="c3", user_input="c", expected=ExpectedBehavior(must_not_include=["error"])),
        ]
        results = runner.run(cases)
        assert len(results) == 3
        assert all(r.passed for r in results)

    def test_failed_case_detected(self, eval_chat_service):
        runner = EvalRunner(eval_chat_service, [DeterministicEvaluator()])
        case = EvalCase(
            case_id="fail",
            user_input="hello",
            expected=ExpectedBehavior(must_include=["nonexistent_keyword_xyz"]),
        )
        results = runner.run([case])
        assert len(results) == 1
        assert not results[0].passed
        assert "nonexistent_keyword_xyz" in str(results[0].details)

    def test_no_evaluators_always_passes(self, eval_chat_service):
        runner = EvalRunner(eval_chat_service, [])
        case = EvalCase(case_id="c1", user_input="hello")
        results = runner.run([case])
        assert results[0].passed
        assert results[0].evaluator_results == {}

    def test_empty_cases_returns_empty(self, eval_chat_service):
        runner = EvalRunner(eval_chat_service, [DeterministicEvaluator()])
        results = runner.run([])
        assert results == []


class TestMetrics:
    def test_compute_metrics_all_pass(self):
        results = [
            EvalResult(case_id="c1", passed=True, evaluator_results={"det": True}, scores={"det": 1.0}, elapsed_ms=100),
            EvalResult(case_id="c2", passed=True, evaluator_results={"det": True}, scores={"det": 1.0}, elapsed_ms=200),
        ]
        cases = [
            EvalCase(case_id="c1", user_input="a", difficulty="easy", risk="low"),
            EvalCase(case_id="c2", user_input="b", difficulty="medium", risk="high"),
        ]
        summary = compute_metrics(results, cases)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.failed == 0
        assert summary.pass_rate == 1.0
        assert summary.avg_elapsed_ms == 150.0
        assert summary.failed_cases == []

    def test_compute_metrics_partial_fail(self):
        results = [
            EvalResult(case_id="c1", passed=True, evaluator_results={"det": True}, scores={"det": 1.0}, elapsed_ms=100),
            EvalResult(case_id="c2", passed=False, evaluator_results={"det": False}, scores={"det": 0.5}, elapsed_ms=300),
            EvalResult(case_id="c3", passed=True, evaluator_results={"det": True}, scores={"det": 1.0}, elapsed_ms=200),
        ]
        cases = [
            EvalCase(case_id="c1", user_input="a", difficulty="easy", risk="low"),
            EvalCase(case_id="c2", user_input="b", difficulty="hard", risk="critical"),
            EvalCase(case_id="c3", user_input="c", difficulty="easy", risk="low"),
        ]
        summary = compute_metrics(results, cases)
        assert summary.total == 3
        assert summary.passed == 2
        assert summary.failed == 1
        assert summary.pass_rate == pytest.approx(2 / 3)
        assert summary.failed_cases == ["c2"]
        assert summary.by_difficulty["easy"]["pass_rate"] == 1.0
        assert summary.by_difficulty["hard"]["pass_rate"] == 0.0
        assert summary.by_risk["critical"]["pass_rate"] == 0.0

    def test_compute_metrics_empty(self):
        summary = compute_metrics([], [])
        assert summary.total == 0
        assert summary.pass_rate == 0.0

    def test_compute_metrics_by_tag(self):
        results = [
            EvalResult(case_id="c1", passed=True, evaluator_results={"det": True}, scores={"det": 1.0}, elapsed_ms=100),
            EvalResult(case_id="c2", passed=False, evaluator_results={"det": False}, scores={"det": 0.0}, elapsed_ms=100),
        ]
        cases = [
            EvalCase(case_id="c1", user_input="a", tags=["greeting", "basic"]),
            EvalCase(case_id="c2", user_input="b", tags=["search", "basic"]),
        ]
        summary = compute_metrics(results, cases)
        assert summary.by_tag["basic"]["total"] == 2
        assert summary.by_tag["basic"]["passed"] == 1
        assert summary.by_tag["greeting"]["pass_rate"] == 1.0
        assert summary.by_tag["search"]["pass_rate"] == 0.0

    def test_metrics_calculator_accumulates(self):
        calc = MetricsCalculator()
        # First run
        calc.add(
            [EvalResult(case_id="c1", passed=True, evaluator_results={"det": True}, scores={"det": 1.0}, elapsed_ms=100)],
            [EvalCase(case_id="c1", user_input="a")],
        )
        # Second run
        calc.add(
            [EvalResult(case_id="c2", passed=False, evaluator_results={"det": False}, scores={"det": 0.0}, elapsed_ms=200)],
            [EvalCase(case_id="c2", user_input="b")],
        )
        summary = calc.compute()
        assert summary.total == 2
        assert summary.passed == 1

        calc.reset()
        assert calc.compute().total == 0


class TestReporter:
    def test_json_report(self):
        results = [
            EvalResult(
                case_id="c1", passed=True,
                evaluator_results={"det": True}, scores={"det": 1.0},
                answer="hello", session_id="s1", elapsed_ms=100, token_count=3,
            ),
        ]
        cases = [EvalCase(case_id="c1", user_input="hi")]
        summary = compute_metrics(results, cases)
        reporter = Reporter("json")
        report = reporter.generate(results, summary)
        data = json.loads(report)
        assert data["summary"]["total"] == 1
        assert data["summary"]["passed"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["case_id"] == "c1"
        assert data["results"][0]["answer"] == "hello"

    def test_console_report(self):
        results = [
            EvalResult(
                case_id="c1", passed=True,
                evaluator_results={"deterministic": True}, scores={"deterministic": 1.0},
                answer="ok", elapsed_ms=50,
            ),
        ]
        cases = [EvalCase(case_id="c1", user_input="hi")]
        summary = compute_metrics(results, cases)
        reporter = Reporter("console")
        report = reporter.generate(results, summary)
        assert "Evaluation Report" in report
        assert "c1" in report
        assert "PASS" in report

    def test_console_report_with_failure(self):
        results = [
            EvalResult(
                case_id="bad", passed=False,
                evaluator_results={"deterministic": False}, scores={"deterministic": 0.5},
                answer="error", elapsed_ms=50,
                details={
                    "deterministic": {
                        "checks": {"outcome": True, "must_include": False},
                        "must_include": {"missing": ["keyword"]},
                    },
                },
            ),
        ]
        cases = [EvalCase(case_id="bad", user_input="hi")]
        summary = compute_metrics(results, cases)
        reporter = Reporter("console")
        report = reporter.generate(results, summary)
        assert "FAIL" in report
        assert "bad" in report
        assert "must_include" in report


class TestFullPipeline:
    """End-to-end: ChatService + EvalRunner + Metrics + Reporter."""

    def test_full_pipeline(self, eval_chat_service):
        # 1. Define cases
        cases = [
            EvalCase(
                case_id="greeting",
                user_input="你好",
                expected=ExpectedBehavior(must_not_include=["error"]),
                tags=["basic"],
                difficulty="easy",
            ),
            EvalCase(
                case_id="holdings",
                user_input="查询持仓",
                expected=ExpectedBehavior(must_not_include=["error"]),
                tags=["finance"],
                difficulty="easy",
            ),
        ]

        # 2. Run evaluation
        runner = EvalRunner(eval_chat_service, [DeterministicEvaluator()])
        results = runner.run(cases)

        # 3. Compute metrics
        summary = compute_metrics(results, cases)

        # 4. Generate reports
        console_reporter = Reporter("console")
        json_reporter = Reporter("json")

        console_report = console_reporter.generate(results, summary)
        json_report = json_reporter.generate(results, summary)

        # 5. Verify
        assert len(results) == 2
        assert all(r.passed for r in results)
        assert summary.total == 2
        assert summary.passed == 2
        assert summary.pass_rate == 1.0

        # Console report contains key info
        assert "Evaluation Report" in console_report
        assert "greeting" in console_report
        assert "holdings" in console_report

        # JSON report is valid
        data = json.loads(json_report)
        assert data["summary"]["total"] == 2
        assert len(data["results"]) == 2