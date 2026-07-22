"""Context management: ToolResultRefStore, Compaction, DataBus.

L1: ToolResultRefStore — store large tool results externally, keep ref in context.
L2: SemanticCompressor — LLM-based semantic compression of large results (future).
L3: Compaction — conversation-level compression via structured handoff.
L4: DataBus — on-demand retrieval of stored data (future).
"""

from .tool_result_store import ToolResultRefStore, StoredResult, make_get_stored_data_tool
from .compaction import (
    compact_messages,
    COMPACTION_THRESHOLD,
    COMPACTION_TARGET,
    CONTEXT_WINDOW_TOKENS,
)

from .databus import build_data_index
from .budget import check_budget, check_budget_compact, BUDGET_REJECT_THRESHOLD, BUDGET_WARN_THRESHOLD

__all__ = [
    "ToolResultRefStore",
    "StoredResult",
    "make_get_stored_data_tool",
    "compact_messages",
    "build_data_index",
    "check_budget",
    "check_budget_compact",
    "BUDGET_REJECT_THRESHOLD",
    "BUDGET_WARN_THRESHOLD",
    "COMPACTION_THRESHOLD",
    "COMPACTION_TARGET",
    "CONTEXT_WINDOW_TOKENS",
]