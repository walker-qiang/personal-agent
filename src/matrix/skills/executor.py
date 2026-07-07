"""Skill executor: run predefined skill workflows."""

from __future__ import annotations

import json
import time
from typing import Any

from ..tools import FinanceToolError, ToolRegistry
from .loader import SkillDefinition


def execute_skill(
    skill: SkillDefinition,
    tools: ToolRegistry,
    trace: Any = None,
) -> dict[str, Any]:
    """Execute a skill's workflow steps sequentially."""
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    findings: list[str] = []

    for step in skill.workflow:
        tool_name = step.get("tool", "")
        if not tool_name:
            continue

        started = time.perf_counter()
        try:
            result = tools.call(tool_name, step.get("arguments", {}))
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            results.append({
                "step": step["step"],
                "name": tool_name,
                "arguments": step.get("arguments", {}),
                "result": result,
                "elapsed_ms": elapsed_ms,
            })
            if trace:
                trace.record({
                    "ok": True,
                    "skill": skill.name,
                    "step": step["step"],
                    "tool": tool_name,
                    "elapsed_ms": elapsed_ms,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
        except FinanceToolError as err:
            errors.append(f"Step {step['step']} ({tool_name}): {err}")
            if trace:
                trace.record({
                    "ok": False,
                    "skill": skill.name,
                    "step": step["step"],
                    "tool": tool_name,
                    "error": str(err),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })

    return {
        "skill": skill.name,
        "title": skill.title,
        "steps_executed": len(results),
        "results": results,
        "errors": errors,
        "findings": findings,
    }