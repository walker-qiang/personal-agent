"""Context management: ToolResultRefStore, Compaction, DataBus.

L1: ToolResultRefStore — store large tool results externally, keep ref in context.
L2: SemanticCompressor — LLM-based semantic compression of large results (future).
L3: Compaction — conversation-level compression via structured handoff.
L4: DataBus — on-demand retrieval of stored data (future).
"""

from .tool_result_store import ToolResultRefStore, StoredResult, make_get_stored_data_tool
from .compaction import (
    should_compact,
    compact_messages,
    estimate_usage_ratio,
    COMPACTION_THRESHOLD,
    COMPACTION_TARGET,
    CONTEXT_WINDOW_TOKENS,
)

from .databus import build_data_index, prefetch_for_step, extract_ref_ids_from_messages
from .budget import check_budget, check_budget_compact, BUDGET_REJECT_THRESHOLD, BUDGET_WARN_THRESHOLD

__all__ = [
    "ToolResultRefStore",
    "StoredResult",
    "make_get_stored_data_tool",
    "should_compact",
    "compact_messages",
    "estimate_usage_ratio",
    "build_data_index",
    "prefetch_for_step",
    "extract_ref_ids_from_messages",
    "check_budget",
    "check_budget_compact",
    "BUDGET_REJECT_THRESHOLD",
    "BUDGET_WARN_THRESHOLD",
    "COMPACTION_THRESHOLD",
    "COMPACTION_TARGET",
    "CONTEXT_WINDOW_TOKENS",
]