"""L4 DataBus: on-demand retrieval of stored data via global index.

Maintains a global data index in the system prompt, prefetches data
based on step dependency declarations, and provides lightweight inspection
tools (outline, search, context) as alternatives to full retrieval.

Design:
- Index table in system prompt: maps refId → tool, summary
- Prefetch: small data (<4096 chars) auto-injected, large data gets enhanced summary
- Inspection tools: outline, search, context — lightweight alternatives to get_stored_data
- Budget-controlled: 4096 chars max for prefetched data, with prioritized降级
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .tool_result_store import ToolResultRefStore

logger = logging.getLogger("matrix.context")

# Budgets
PREFETCH_BUDGET_CHARS = 4096
SMALL_DATA_THRESHOLD = 4096
ENHANCED_SUMMARY_MAX = 1000


def build_data_index(
    ref_store: ToolResultRefStore,
    messages: list[dict[str, Any]],
) -> str:
    """Build a data index to inject into the system prompt.

    Scans messages for __refId references and builds a compact index
    mapping refId → summary. This lets the LLM know what data is available
    without consuming the full data.
    """
    ref_ids: set[str] = set()
    for msg in messages:
        content = str(msg.get("content", ""))
        if "__refId" in content:
            import re
            for match in re.finditer(r'"__refId"\s*:\s*"([^"]+)"', content):
                ref_ids.add(match.group(1))

    if not ref_ids:
        return ""

    entries: list[str] = []
    for ref_id in sorted(ref_ids):
        summary = ref_store.get_summary(ref_id)
        if summary:
            entries.append(f"- `{ref_id}`: {summary}")

    if not entries:
        return ""

    return (
        "## Available Data (use get_stored_data to retrieve)\n"
        + "\n".join(entries)
        + "\n"
    )


def prefetch_for_step(
    ref_store: ToolResultRefStore,
    ref_ids: list[str],
) -> str:
    """Prefetch data for a step based on declared dependencies.

    Small data (<=SMALL_DATA_THRESHOLD chars) is injected directly.
    Large data gets an enhanced summary with structure hints.

    Returns a string to inject into the system prompt.
    Budget: PREFETCH_BUDGET_CHARS.
    """
    if not ref_ids:
        return ""

    parts: list[str] = []
    used = 0

    for ref_id in ref_ids:
        if used >= PREFETCH_BUDGET_CHARS:
            break

        data = ref_store.get(ref_id)
        if data is None:
            continue

        data_str = json.dumps(data, ensure_ascii=False, default=str)
        if len(data_str) <= SMALL_DATA_THRESHOLD:
            entry = f"## Prefetched: {ref_id}\n```json\n{data_str}\n```\n"
            if used + len(entry) <= PREFETCH_BUDGET_CHARS:
                parts.append(entry)
                used += len(entry)
                continue

        # Large data: enhanced summary
        summary = _build_enhanced_summary(data)
        entry = f"## Data Summary: {ref_id}\n{summary}\n"
        if used + len(entry) <= PREFETCH_BUDGET_CHARS:
            parts.append(entry)
            used += len(entry)

    return "\n".join(parts) if parts else ""


def _build_enhanced_summary(data: Any) -> str:
    """Build a structured summary of data for the LLM.

    Unlike the simple summary in ToolResultRefStore, this includes
    structure hints (first few items, key names) to help LLM reasoning.
    """
    if isinstance(data, list):
        count = len(data)
        if count == 0:
            return "Empty list"
        lines = [f"Array, {count} items total"]
        for i, item in enumerate(data[:3]):
            if isinstance(item, dict):
                preview = json.dumps(item, ensure_ascii=False)[:200]
            else:
                preview = str(item)[:200]
            lines.append(f"  [{i}]: {preview}")
        if count > 3:
            lines.append(f"  ... and {count - 3} more items")
        return "\n".join(lines)

    if isinstance(data, dict):
        keys = list(data.keys())
        lines = [f"Object with {len(keys)} keys"]
        for k in keys[:10]:
            v = data[k]
            if isinstance(v, (str, int, float, bool)):
                lines.append(f"  {k}: {str(v)[:100]}")
            elif isinstance(v, list):
                lines.append(f"  {k}: [Array of {len(v)} items]")
            elif isinstance(v, dict):
                lines.append(f"  {k}: {{Object with {len(v)} keys}}")
            else:
                lines.append(f"  {k}: {type(v).__name__}")
        if len(keys) > 10:
            lines.append(f"  ... and {len(keys) - 10} more keys")
        return "\n".join(lines)

    return str(data)[:ENHANCED_SUMMARY_MAX]


def extract_ref_ids_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Extract all __refId values from tool messages."""
    import re
    ref_ids: list[str] = []
    seen: set[str] = set()

    for msg in messages:
        content = str(msg.get("content", ""))
        for match in re.finditer(r'"__refId"\s*:\s*"([^"]+)"', content):
            ref_id = match.group(1)
            if ref_id not in seen:
                seen.add(ref_id)
                ref_ids.append(ref_id)

    return ref_ids