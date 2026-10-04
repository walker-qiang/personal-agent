"""Validation and compilation for application-level DAG delegation plans."""

from __future__ import annotations

import re
from typing import Any, Iterable


_OUTPUT_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_STEP_REF_RE = re.compile(r"^(0|[1-9][0-9]*):([1-9][0-9]*)$")


class PlanValidationError(ValueError):
    """A delegation plan cannot be safely scheduled."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)


def compile_dag_plan(
    plan: Any,
    *,
    plan_revision: int = 0,
    expected_revision: int | None = None,
    completed_step_refs: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize and validate a delegation plan before scheduling it.

    The compiler copies step dictionaries, fills only backward-compatible
    defaults, and never mutates the LLM response or graph state.
    """
    _validate_revision_number(plan_revision, "plan_revision")
    if expected_revision is not None:
        _validate_revision_number(expected_revision, "expected_revision")
        if plan_revision != expected_revision:
            raise PlanValidationError(
                "plan_revision_conflict",
                (
                    "plan revision conflict: "
                    f"expected {expected_revision}, got {plan_revision}"
                ),
                details={
                    "expected_revision": expected_revision,
                    "actual_revision": plan_revision,
                },
            )

    if not isinstance(plan, list):
        raise PlanValidationError(
            "plan_not_a_list",
            "delegation plan must be a JSON array",
        )

    compiled: list[dict[str, Any]] = []
    step_ids: set[int] = set()
    output_keys: set[str] = set()

    for index, raw_step in enumerate(plan):
        if not isinstance(raw_step, dict):
            raise PlanValidationError(
                "step_not_an_object",
                f"DAG step {index + 1} must be an object",
                details={"index": index},
            )

        step = dict(raw_step)
        if "step" not in step:
            step["step"] = index + 1
        if "depends_on" not in step:
            step["depends_on"] = []
        if "output_key" not in step:
            step["output_key"] = f"step_{step['step']}"
        if "skill_name" not in step:
            step["skill_name"] = ""

        step_id = step["step"]
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 1:
            raise PlanValidationError(
                "invalid_step_id",
                f"DAG step id must be a positive integer, got {step_id!r}",
                details={"step": step_id},
            )
        if step_id in step_ids:
            raise PlanValidationError(
                "duplicate_step_id",
                f"duplicate DAG step id: {step_id}",
                details={"step": step_id},
            )
        step_ids.add(step_id)

        dependencies = step["depends_on"]
        if not isinstance(dependencies, list):
            raise PlanValidationError(
                "invalid_dependencies",
                f"depends_on for step {step_id} must be an array",
                details={"step": step_id},
            )
        normalized_dependencies: list[int] = []
        for dependency in dependencies:
            if (
                isinstance(dependency, bool)
                or not isinstance(dependency, int)
                or dependency < 1
            ):
                raise PlanValidationError(
                    "invalid_dependency_id",
                    (
                        f"step {step_id} has an invalid dependency id: "
                        f"{dependency!r}"
                    ),
                    details={"step": step_id, "dependency": dependency},
                )
            if dependency in normalized_dependencies:
                raise PlanValidationError(
                    "duplicate_dependency",
                    f"step {step_id} declares dependency {dependency} more than once",
                    details={"step": step_id, "dependency": dependency},
                )
            normalized_dependencies.append(dependency)
        if step_id in normalized_dependencies:
            raise PlanValidationError(
                "self_dependency",
                f"step {step_id} cannot depend on itself",
                details={"step": step_id},
            )

        output_key = step["output_key"]
        if (
            not isinstance(output_key, str)
            or not _OUTPUT_KEY_RE.fullmatch(output_key)
        ):
            raise PlanValidationError(
                "invalid_output_key",
                (
                    f"output_key for step {step_id} must be an ASCII identifier "
                    "(letters, digits, underscore; starting with a letter)"
                ),
                details={"step": step_id, "output_key": output_key},
            )
        if output_key in output_keys:
            raise PlanValidationError(
                "duplicate_output_key",
                f"duplicate output_key: {output_key}",
                details={"step": step_id, "output_key": output_key},
            )
        output_keys.add(output_key)

        step["depends_on"] = normalized_dependencies
        compiled.append(step)

    for step in compiled:
        unknown = [
            dependency
            for dependency in step["depends_on"]
            if dependency not in step_ids
        ]
        if unknown:
            raise PlanValidationError(
                "unknown_dependency",
                (
                    f"step {step['step']} depends on unknown step(s): "
                    f"{unknown}"
                ),
                details={"step": step["step"], "dependencies": unknown},
            )

    _validate_acyclic(compiled)
    _validate_completed_step_refs(
        completed_step_refs or [],
        plan_revision=plan_revision,
        step_ids=step_ids,
    )
    return compiled


def completed_steps_for_revision(
    completed: Iterable[int] | None,
    completed_step_refs: Iterable[str] | None,
    plan_revision: int,
) -> set[int]:
    """Return completion state applicable to the current plan revision."""
    current_prefix = f"{plan_revision}:"
    refs = list(completed_step_refs or [])
    revision_completed = {
        int(ref.split(":", 1)[1])
        for ref in refs
        if ref.startswith(current_prefix)
    }
    if revision_completed:
        return revision_completed
    if refs:
        # A plan revision with only stale refs has no completed steps yet.
        return set()
    return set(completed or [])


def _validate_revision_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PlanValidationError(
            "invalid_plan_revision",
            f"{name} must be a non-negative integer, got {value!r}",
            details={name: value},
        )


def _validate_completed_step_refs(
    refs: Iterable[str],
    *,
    plan_revision: int,
    step_ids: set[int],
) -> None:
    for ref in refs:
        if not isinstance(ref, str):
            raise PlanValidationError(
                "invalid_step_reference",
                f"completed step reference must be a string, got {ref!r}",
                details={"reference": ref},
            )
        match = _STEP_REF_RE.fullmatch(ref)
        if match is None:
            raise PlanValidationError(
                "invalid_step_reference",
                f"invalid completed step reference: {ref!r}",
                details={"reference": ref},
            )
        revision = int(match.group(1))
        step_id = int(match.group(2))
        if revision > plan_revision:
            raise PlanValidationError(
                "plan_revision_conflict",
                (
                    f"completed step reference {ref!r} belongs to a future "
                    f"plan revision {revision}"
                ),
                details={
                    "reference": ref,
                    "plan_revision": plan_revision,
                },
            )
        if revision == plan_revision and step_id not in step_ids:
            raise PlanValidationError(
                "plan_revision_conflict",
                (
                    f"completed step reference {ref!r} does not exist in "
                    f"plan revision {plan_revision}"
                ),
                details={
                    "reference": ref,
                    "plan_revision": plan_revision,
                },
            )


def _validate_acyclic(plan: list[dict[str, Any]]) -> None:
    dependencies = {
        step["step"]: set(step["depends_on"])
        for step in plan
    }
    visited: set[int] = set()
    visiting: set[int] = set()

    def visit(step_id: int, path: list[int]) -> None:
        if step_id in visiting:
            cycle_start = path.index(step_id)
            cycle = path[cycle_start:] + [step_id]
            raise PlanValidationError(
                "cycle_dependency",
                f"DAG plan contains a dependency cycle: {cycle}",
                details={"cycle": cycle},
            )
        if step_id in visited:
            return

        visiting.add(step_id)
        for dependency in dependencies[step_id]:
            visit(dependency, path + [step_id])
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in dependencies:
        visit(step_id, [])
