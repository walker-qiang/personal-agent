"""Budget pre-check: estimate token cost before LLM calls, reject if over budget.

With free-tier models, the threshold is set high (98%) to prevent API errors
while avoiding unnecessary restrictions. The compaction check (85%) still
triggers earlier to keep context quality high.

Design:
- Pre-check: estimate tokens before each LLM call
- Threshold: BUDGET_REJECT_THRESHOLD = 0.98 (free model, nearly full)
- Compaction: triggered at COMPACTION_THRESHOLD = 0.85 (from compaction.py)
- Rejection: triggered when budget > BUDGET_REJECT_THRESHOLD
- 降级顺序: drop preview → try compaction → reject
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import json
import logging
from typing import Any

from ..llm.truncate import estimate_tokens
from .compaction import CONTEXT_WINDOW_TOKENS

logger = logging.getLogger("matrix.context")

# Budget thresholds
BUDGET_REJECT_THRESHOLD = 0.98  # Free model: near-full, just prevent errors
BUDGET_WARN_THRESHOLD = 0.90    # Warn when approaching limit
RESERVE_OUTPUT_TOKENS = 4096    # Reserve for the model's response


@dataclass(frozen=True)
class BudgetMeasurement:
    """Provider-neutral measurements used for admission and diagnostics."""

    input_bytes: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


def measure_budget(
    texts: Iterable[str],
    *,
    output_tokens: int = 0,
    overhead_tokens: int = 0,
) -> BudgetMeasurement:
    """Measure a request using the same conservative estimator everywhere."""
    values = [str(value) for value in texts]
    input_bytes = sum(len(value.encode("utf-8")) for value in values)
    input_tokens = sum(estimate_tokens(value) for value in values) + max(0, overhead_tokens)
    output_tokens = max(0, output_tokens)
    return BudgetMeasurement(
        input_bytes=input_bytes,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def measure_json_budget(
    value: Any,
    *,
    output_tokens: int = 0,
    overhead_tokens: int = 0,
) -> BudgetMeasurement:
    """Serialize a JSON request once, then measure its actual UTF-8 payload."""
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return measure_budget(
        [encoded],
        output_tokens=output_tokens,
        overhead_tokens=overhead_tokens,
    )


def check_budget(
    messages: list[dict[str, Any]],
    system_prompt: str,
    context_window: int = CONTEXT_WINDOW_TOKENS,
    reserve_output: int = RESERVE_OUTPUT_TOKENS,
) -> tuple[bool, float, str]:
    """Check if the estimated token usage exceeds the budget.

    Args:
        messages: The conversation messages.
        system_prompt: The system prompt text.
        context_window: Maximum context window size in tokens.
        reserve_output: Tokens to reserve for the model's response.

    Returns:
        (ok, usage_ratio, message) — ok=False means the call should be rejected.
    """
    # Estimate tokens
    total = estimate_tokens(system_prompt)
    for msg in messages:
        content = msg.get("content", "")
        total += estimate_tokens(str(content))
        # Also estimate tool call definitions if present
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            total += estimate_tokens(str(fn.get("name", "")))
            total += estimate_tokens(str(fn.get("arguments", "")))

    available = context_window - reserve_output
    ratio = total / available if available > 0 else 1.0

    if ratio >= BUDGET_REJECT_THRESHOLD:
        return (
            False,
            ratio,
            f"PROMPT_BUDGET_EXCEEDED: estimated {total}/{available} tokens "
            f"({ratio:.1%} >= {BUDGET_REJECT_THRESHOLD:.0%} threshold)",
        )
    elif ratio >= BUDGET_WARN_THRESHOLD:
        return (
            True,
            ratio,
            f"Budget warning: {ratio:.1%} of context window used",
        )
    else:
        return (True, ratio, "ok")


def check_budget_compact(
    messages: list[dict[str, Any]],
    system_prompt: str,
    context_window: int = CONTEXT_WINDOW_TOKENS,
) -> tuple[bool, str]:
    """Check budget and suggest action.

    Returns:
        (proceed, action) where action is one of:
        - "ok": proceed normally
        - "warn": proceed but log warning
        - "compact": should trigger compaction
        - "reject": must reject the call
    """
    ok, ratio, msg = check_budget(messages, system_prompt, context_window)

    if not ok:
        return (False, "reject")
    elif ratio >= 0.85:  # COMPACTION_THRESHOLD
        return (True, "compact")
    elif ratio >= BUDGET_WARN_THRESHOLD:
        return (True, "warn")
    else:
        return (True, "ok")
