"""bucket_allocation tool — allocation target vs actual comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import ToolDefinition
from .holdings import holdings_summary


def bucket_allocation(cache_path: str = "") -> dict[str, Any]:
    """Show allocation targets vs actual for each bucket."""
    summary = holdings_summary(cache_path=cache_path)
    return {
        "currency": summary["currency"],
        "total_balance_cents": summary["total_balance_cents"],
        "total_balance_yuan": summary["total_balance_yuan"],
        "buckets": summary["buckets"],
    }


tool_definition = ToolDefinition(
    name="finance.bucket_allocation",
    description="Return current bucket allocation and configured targets.",
    input_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=bucket_allocation,
)