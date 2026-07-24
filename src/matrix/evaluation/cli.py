"""Evaluation CLI — quality gate commands for personal-agent.

Commands:
    check-skills            Validate skill definitions (Layer 1)
    regression              Run regression evaluation (Layer 2)
    quality                 Run quality assessment (Layer 3)
    update-baseline <type>  Update baseline file (regression | quality)
    list-cases              List evaluation cases in dataset

Usage:
    python -m matrix.evaluation.cli check-skills
    python -m matrix.evaluation.cli regression
    python -m matrix.evaluation.cli regression --no-baseline
    python -m matrix.evaluation.cli quality --cases smoke_greeting
    python -m matrix.evaluation.cli update-baseline regression
    python -m matrix.evaluation.cli list-cases
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _resolve_skills_dir() -> Path:
    """Resolve skills_base_dir using the same logic as config.py.

    Does NOT require JWT_SECRET or other env vars — only needs
    MATRIX_SKILLS_BASE_DIR or the default personal-assets/技能/ path.
    """
    from matrix.config import ENV_SKILLS_BASE_DIR, find_root

    root = find_root(Path.cwd())
    skills_raw = os.environ.get(ENV_SKILLS_BASE_DIR, "").strip()
    if skills_raw:
        p = Path(skills_raw).expanduser()
        return p if p.is_absolute() else root / p
    return root / ".." / "personal-assets" / "技能"


def _default_dataset_path() -> Path:
    """Default eval dataset path."""
    from matrix.config import find_root

    root = find_root(Path.cwd())
    return root / "tests" / "baselines" / "eval_dataset.json"


def _default_baseline_path(baseline_type: str) -> Path:
    """Default baseline file path."""
    from matrix.config import find_root

    root = find_root(Path.cwd())
    return root / "tests" / "baselines" / f"{baseline_type}_baseline.json"


# ---- Layer 1: check-skills ----


def cmd_check_skills(args: argparse.Namespace) -> int:
    """Validate all skill definitions in skills_base_dir."""
    from matrix.skills import load_skills

    skills_dir = _resolve_skills_dir()

    if not skills_dir.exists():
        print(f"✗ Skills directory not found: {skills_dir}")
        return 1

    print(f"Scanning: {skills_dir}")
    skills = load_skills(skills_dir)

    if not skills:
        print("✗ No skills found.")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0

    for skill in skills:
        checked += 1
        skill_dir = skills_dir / skill.name

        # Check title (empty title is an error; title == name is normal behavior)
        if not skill.title:
            errors.append(f"  {skill.name}: title is empty")

        # Check description (empty description is a warning, reduces match quality)
        if not skill.description:
            warnings.append(f"  {skill.name}: description is empty (reduces skill matching)")

        # Check workflow steps have tool names
        for i, step in enumerate(skill.workflow):
            if "tool" in step and not step["tool"]:
                errors.append(f"  {skill.name}: step {step.get('step', i+1)} has empty tool name")

        # Check referenced knowledge files exist
        for kf in skill.knowledge_files:
            kpath = skill_dir / "references" / kf
            if not kpath.exists():
                errors.append(f"  {skill.name}: missing knowledge file: {kf}")

        # Check referenced script files exist
        for sf in skill.script_files:
            spath = skill_dir / "scripts" / sf
            if not spath.exists():
                errors.append(f"  {skill.name}: missing script file: {sf}")

        has_err = any(e.startswith(f"  {skill.name}:") for e in errors)
        has_warn = any(w.startswith(f"  {skill.name}:") for w in warnings)
        if has_err:
            status = "✗"
        elif has_warn:
            status = "⚠"
        else:
            status = "✓"
        print(f"  {status} {skill.name} (workflow: {len(skill.workflow)} steps, "
              f"knowledge: {len(skill.knowledge_files)}, scripts: {len(skill.script_files)})")

    print("")
    if warnings:
        print(f"⚠ {len(warnings)} warning(s) (non-blocking):")
        for w in warnings:
            print(f"  {w}")
        print("")

    if errors:
        print(f"✗ {len(errors)} error(s) found in {checked} skill(s):")
        for e in errors:
            print(f"  {e}")
        return 1

    print(f"✓ All {checked} skill(s) valid.")
    return 0


# ---- Layer 2: regression ----


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load eval cases from JSON dataset file."""
    if not path.exists():
        print(f"✗ Dataset not found: {path}")
        print(f"  Fallback to smoke dataset.")
        from matrix.config import find_root

        root = find_root(Path.cwd())
        path = root / "src" / "matrix" / "evaluation" / "datasets" / "smoke.json"
        if not path.exists():
            print(f"✗ Smoke dataset also not found: {path}")
            sys.exit(1)

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def _filter_cases(
    cases: list[dict[str, Any]],
    case_ids: list[str] | None,
    tags: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter cases by case_ids or tags."""
    if case_ids:
        return [c for c in cases if c["case_id"] in case_ids]
    if tags:
        return [c for c in cases if any(t in c.get("tags", []) for t in tags)]
    return cases


def _create_chat_service() -> Any:
    """Create a ChatService instance from .env config."""
    from dotenv import load_dotenv

    load_dotenv()

    from matrix.config import load_config
    from matrix.chat import ChatService
    from matrix.tools import ToolRegistry
    from matrix.tools.finance import register_all as register_finance
    from matrix.tools.web import register_all as register_web
    from matrix.tools.agnes import register_all as register_agnes

    config = load_config()
    registry = ToolRegistry()
    register_finance(registry, config.cache_path)
    register_web(registry)
    register_agnes(registry)

    # Register code tools if enabled
    if config.code_sandbox_enabled:
        from matrix.tools.code import register_all as register_code
        register_code(registry, config)

    return ChatService(config, registry)


def cmd_regression(args: argparse.Namespace) -> int:
    """Run regression evaluation with deterministic evaluator."""
    from matrix.evaluation import (
        DeterministicEvaluator,
        EvalCase,
        EvalRunner,
        Reporter,
        compute_metrics,
    )
    from matrix.evaluation.baseline import (
        compare_regression,
        load_baseline,
        build_regression_baseline,
        save_baseline,
    )

    dataset_path = Path(args.dataset) if args.dataset else _default_dataset_path()
    raw_cases = _load_dataset(dataset_path)
    raw_cases = _filter_cases(raw_cases, args.cases, args.tags)

    if not raw_cases:
        print("✗ No cases to run.")
        return 1

    cases = [EvalCase.from_dict(c) for c in raw_cases]
    print(f"Running {len(cases)} case(s)...\n")

    chat_service = _create_chat_service()
    evaluator = DeterministicEvaluator()
    runner = EvalRunner(chat_service, [evaluator])

    results = runner.run(cases)
    summary = compute_metrics(results, cases)

    # Print results
    reporter = Reporter(args.format)
    print(reporter.generate(results, summary))

    if args.no_baseline:
        return 0 if summary.failed == 0 else 1

    # Compare with baseline
    baseline_path = Path(args.baseline) if args.baseline else _default_baseline_path("regression")
    baseline = load_baseline(baseline_path)

    if baseline is None:
        print(f"\n⚠ No baseline found at {baseline_path}")
        print("  Run 'python -m matrix.evaluation.cli update-baseline regression' to create one.")
        return 0 if summary.failed == 0 else 1

    print(f"\nComparing with baseline: {baseline_path}")
    report = compare_regression(results, baseline)
    print(report.to_console())

    return 1 if report.has_regression else 0


# ---- Layer 3: quality ----


def cmd_quality(args: argparse.Namespace) -> int:
    """Run quality assessment with LLM-as-Judge evaluator."""
    from matrix.evaluation import (
        DeterministicEvaluator,
        EvalCase,
        EvalRunner,
        LLMEvaluator,
        Reporter,
        compute_metrics,
    )
    from matrix.evaluation.baseline import (
        compare_quality,
        load_baseline,
    )
    from matrix.llm import build_llm_client

    dataset_path = Path(args.dataset) if args.dataset else _default_dataset_path()
    raw_cases = _load_dataset(dataset_path)
    raw_cases = _filter_cases(raw_cases, args.cases, args.tags)

    if not raw_cases:
        print("✗ No cases to run.")
        return 1

    cases = [EvalCase.from_dict(c) for c in raw_cases]
    print(f"Running {len(cases)} case(s) with LLM-as-Judge...\n")

    chat_service = _create_chat_service()
    config = chat_service.config

    # Use the pipeline LLM for judging (free Agnes model)
    judge_llm = build_llm_client(
        provider=config.pipeline_provider,
        agnes_api_key=config.agnes_api_key,
        deepseek_api_key=config.deepseek_api_key,
        anthropic_api_key=config.anthropic_api_key,
        model=config.pipeline_model,
        agnes_base_url=config.agnes_base_url,
        deepseek_base_url=config.deepseek_base_url,
        max_tokens=config.agent_max_tokens,
        timeout_sec=config.agent_model_timeout_sec,
        max_message_chars=config.max_message_chars,
    )

    evaluators = [DeterministicEvaluator(), LLMEvaluator(judge_llm)]
    runner = EvalRunner(chat_service, evaluators)

    results = runner.run(cases)
    summary = compute_metrics(results, cases)

    # Print results
    reporter = Reporter(args.format)
    print(reporter.generate(results, summary))

    if args.no_baseline:
        return 0 if summary.failed == 0 else 1

    # Compare with baseline
    baseline_path = Path(args.baseline) if args.baseline else _default_baseline_path("quality")
    baseline = load_baseline(baseline_path)

    if baseline is None:
        print(f"\n⚠ No baseline found at {baseline_path}")
        print("  Run 'python -m matrix.evaluation.cli update-baseline quality' to create one.")
        return 0 if summary.failed == 0 else 1

    print(f"\nComparing with baseline: {baseline_path}")
    report = compare_quality(results, baseline)
    print(report.to_console())

    return 1 if report.has_regression else 0


# ---- update-baseline ----


def cmd_update_baseline(args: argparse.Namespace) -> int:
    """Update a baseline file with current results."""
    baseline_type = args.type

    if baseline_type not in ("regression", "quality"):
        print(f"✗ Invalid baseline type: {baseline_type}")
        print("  Valid types: regression, quality")
        return 1

    from matrix.evaluation import (
        DeterministicEvaluator,
        EvalCase,
        EvalRunner,
        compute_metrics,
    )
    from matrix.evaluation.baseline import (
        build_regression_baseline,
        build_quality_baseline,
        save_baseline,
    )

    dataset_path = Path(args.dataset) if args.dataset else _default_dataset_path()
    raw_cases = _load_dataset(dataset_path)
    cases = [EvalCase.from_dict(c) for c in raw_cases]

    print(f"Running {len(cases)} case(s) to generate baseline...\n")

    chat_service = _create_chat_service()

    if baseline_type == "regression":
        evaluator = DeterministicEvaluator()
        runner = EvalRunner(chat_service, [evaluator])
        results = runner.run(cases)
        baseline_data = build_regression_baseline(results, cases)
    else:
        from matrix.evaluation import LLMEvaluator
        from matrix.llm import build_llm_client

        config = chat_service.config
        judge_llm = build_llm_client(
            provider=config.pipeline_provider,
            agnes_api_key=config.agnes_api_key,
            deepseek_api_key=config.deepseek_api_key,
            anthropic_api_key=config.anthropic_api_key,
            model=config.pipeline_model,
            agnes_base_url=config.agnes_base_url,
            deepseek_base_url=config.deepseek_base_url,
            max_tokens=config.agent_max_tokens,
            timeout_sec=config.agent_model_timeout_sec,
            max_message_chars=config.max_message_chars,
        )
        evaluators = [DeterministicEvaluator(), LLMEvaluator(judge_llm)]
        runner = EvalRunner(chat_service, evaluators)
        results = runner.run(cases)
        baseline_data = build_quality_baseline(results)

    baseline_path = _default_baseline_path(baseline_type)
    save_baseline(baseline_path, baseline_data)

    summary = baseline_data.get("summary", {})
    print(f"\n✓ Baseline saved: {baseline_path}")
    print(f"  Version: {baseline_data.get('version', 'unknown')}")
    print(f"  Git commit: {baseline_data.get('git_commit', 'unknown')}")

    if baseline_type == "regression":
        print(f"  Total: {summary.get('total', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Pass rate: {summary.get('pass_rate', 0):.1%}")
    else:
        print(f"  Avg quality score: {summary.get('avg_quality_score', 0):.3f}")
        dims = summary.get("dimensions", {})
        for dim, val in dims.items():
            print(f"    {dim}: {val:.3f}")

    print(f"\n  Commit this file to git:")
    print(f"  git add {baseline_path} && git commit -m 'chore: update {baseline_type} baseline'")

    return 0


# ---- list-cases ----


def cmd_list_cases(args: argparse.Namespace) -> int:
    """List all cases in the evaluation dataset."""
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset_path()
    raw_cases = _load_dataset(dataset_path)

    print(f"Dataset: {dataset_path}")
    print(f"Total cases: {len(raw_cases)}\n")
    print(f"{'Case ID':<30} {'Difficulty':<10} {'Risk':<10} {'Tags'}")
    print("-" * 80)

    for c in raw_cases:
        cid = c["case_id"]
        diff = c.get("difficulty", "easy")
        risk = c.get("risk", "low")
        tags = ", ".join(c.get("tags", []))
        print(f"{cid:<30} {diff:<10} {risk:<10} {tags}")

    return 0


# ---- Main ----


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="matrix.evaluation.cli",
        description="Quality gate CLI for personal-agent evaluation.",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # check-skills
    sub.add_parser("check-skills", help="Validate skill definitions (Layer 1)")

    # regression
    p_reg = sub.add_parser("regression", help="Run regression evaluation (Layer 2)")
    p_reg.add_argument("--dataset", help="Dataset JSON file path")
    p_reg.add_argument("--baseline", help="Baseline JSON file path")
    p_reg.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison")
    p_reg.add_argument("--format", choices=["console", "json"], default="console", help="Output format")
    p_reg.add_argument("--cases", help="Comma-separated case IDs to run")
    p_reg.add_argument("--tags", help="Comma-separated tags to filter")

    # quality
    p_qual = sub.add_parser("quality", help="Run quality assessment (Layer 3)")
    p_qual.add_argument("--dataset", help="Dataset JSON file path")
    p_qual.add_argument("--baseline", help="Baseline JSON file path")
    p_qual.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison")
    p_qual.add_argument("--format", choices=["console", "json"], default="console", help="Output format")
    p_qual.add_argument("--cases", help="Comma-separated case IDs to run")
    p_qual.add_argument("--tags", help="Comma-separated tags to filter")

    # update-baseline
    p_ub = sub.add_parser("update-baseline", help="Update baseline file")
    p_ub.add_argument("type", choices=["regression", "quality"], help="Baseline type")
    p_ub.add_argument("--dataset", help="Dataset JSON file path")

    # list-cases
    p_lc = sub.add_parser("list-cases", help="List evaluation cases")
    p_lc.add_argument("--dataset", help="Dataset JSON file path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Parse comma-separated filters
    if hasattr(args, "cases") and args.cases:
        args.cases = [c.strip() for c in args.cases.split(",")]
    if hasattr(args, "tags") and args.tags:
        args.tags = [t.strip() for t in args.tags.split(",")]

    commands = {
        "check-skills": cmd_check_skills,
        "regression": cmd_regression,
        "quality": cmd_quality,
        "update-baseline": cmd_update_baseline,
        "list-cases": cmd_list_cases,
    }

    exit_code = commands[args.command](args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
