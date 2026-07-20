"""Skill executor: run predefined skill workflows with parameter bindings.

Parameter bindings eliminate the "data搬运" problem: instead of LLM copying
data between steps, the system resolves cross-step bindings deterministically.

Two binding mechanisms:
1. Template syntax in arguments: {{step_N.output.field.path}}
   Example: web_fetch(url={{step_1.output.items[0].url}})
2. Explicit parameterBindings in frontmatter:
   parameterBindings:
     - from: step_1
       field: "output.items"
       to: step_2
       param: "items"
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from ..tools import FinanceToolError, ToolRegistry
from .loader import SkillDefinition

logger = logging.getLogger("matrix.skills")

# Template pattern: {{step_N.output.field.path}} or {{step_N.output}}
_TEMPLATE_RE = re.compile(r"\{\{(step_\d+)\.(output(?:\.\S+)?)\}\}")


def execute_skill(
    skill: SkillDefinition,
    tools: ToolRegistry,
    trace: Any = None,
) -> dict[str, Any]:
    """Execute a skill's workflow steps sequentially with parameter bindings.

    Before each step executes, template references in arguments are resolved
    against previous step results. This eliminates the need for LLM to
    "搬运" data between steps.
    """
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    findings: list[str] = []
    # Index: {step_key: output_value} for template resolution
    step_outputs: dict[str, Any] = {}

    for step in skill.workflow:
        tool_name = step.get("tool", "")
        if not tool_name:
            continue

        step_num = step.get("step", len(results) + 1)
        step_key = f"step_{step_num}"

        # Resolve parameter bindings for this step
        arguments = _resolve_arguments(
            step.get("arguments", {}),
            step_outputs,
            step_key,
            skill.parameter_bindings,
        )

        started = time.perf_counter()
        try:
            result = tools.call(tool_name, arguments)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

            # Store output for downstream steps to reference
            step_outputs[step_key] = {"output": result}

            results.append({
                "step": step_num,
                "name": tool_name,
                "arguments": arguments,
                "result": result,
                "elapsed_ms": elapsed_ms,
            })
            if trace:
                trace.record({
                    "ok": True,
                    "skill": skill.name,
                    "step": step_num,
                    "tool": tool_name,
                    "elapsed_ms": elapsed_ms,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
        except FinanceToolError as err:
            errors.append(f"Step {step_num} ({tool_name}): {err}")
            step_outputs[step_key] = {"error": str(err)}
            if trace:
                trace.record({
                    "ok": False,
                    "skill": skill.name,
                    "step": step_num,
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


def _resolve_arguments(
    arguments: dict[str, Any],
    step_outputs: dict[str, Any],
    current_step: str,
    bindings: list[dict[str, str]],
) -> dict[str, Any]:
    """Resolve template references and explicit bindings in step arguments.

    Template syntax: {{step_N.output.field.path}}
    Example: {{step_1.output.items[0].url}} resolves to the actual URL value.

    Returns the resolved arguments dict with all templates replaced.
    """
    resolved: dict[str, Any] = {}

    for key, value in arguments.items():
        if isinstance(value, str):
            resolved[key] = _resolve_template(value, step_outputs)
        elif isinstance(value, (list, dict)):
            resolved[key] = _resolve_nested(value, step_outputs)
        else:
            resolved[key] = value

    # Apply explicit parameterBindings from frontmatter
    for binding in bindings:
        if binding.get("to", "") == current_step:
            from_step = binding.get("from", "")
            field = binding.get("field", "")
            param = binding.get("param", "")
            if from_step in step_outputs and param:
                value = _resolve_field_path(
                    step_outputs[from_step], field, ""
                )
                if value is not None:
                    resolved[param] = value

    return resolved


def _resolve_template(text: str, step_outputs: dict[str, Any]) -> str:
    """Replace {{step_N.output.field}} templates with actual values.

    If the template resolves to a non-string value, it's JSON-serialized.
    If the template can't be resolved (step not found), the original
    template is preserved as-is — this allows LLM to fill in the value
    as a fallback.
    """
    def _replace(match: re.Match) -> str:
        step_key = match.group(1)
        field_path = match.group(2)

        if step_key not in step_outputs:
            logger.debug("Template resolution: %s not found, preserving template", step_key)
            return match.group(0)  # preserve template as-is

        value = _resolve_field_path(step_outputs[step_key], field_path, None)
        if value is None:
            logger.debug("Template resolution: %s.%s returned None", step_key, field_path)
            return match.group(0)

        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    return _TEMPLATE_RE.sub(_replace, text)


def _resolve_nested(
    obj: Any, step_outputs: dict[str, Any],
) -> Any:
    """Recursively resolve templates in nested data structures."""
    if isinstance(obj, dict):
        return {k: _resolve_nested(v, step_outputs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_nested(v, step_outputs) for v in obj]
    if isinstance(obj, str):
        return _resolve_template(obj, step_outputs)
    return obj


def _resolve_field_path(
    container: dict[str, Any], field_path: str, default: Any,
) -> Any:
    """Resolve a dot-separated field path within a container dict.

    Supports simple array indexing: items[0].url
    Example: "output.items[0].url" → container["output"]["items"][0]["url"]
    """
    if not field_path:
        return container

    # Split on dots, but handle array indexing
    parts = re.split(r"\.", field_path)
    current = container

    for part in parts:
        array_match = re.match(r"(\w+)\[(\d+)\]", part)
        if array_match:
            key = array_match.group(1)
            idx = int(array_match.group(2))
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return default
        else:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

    return current