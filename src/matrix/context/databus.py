"""L4 DataBus: on-demand retrieval of stored data via global index.

Maintains a global data index in the system prompt so the LLM knows what
data is available without consuming the full data.
"""

from __future__ import annotations

import logging
from typing import Any

from .tool_result_store import ToolResultRefStore

logger = logging.getLogger("matrix.context")


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