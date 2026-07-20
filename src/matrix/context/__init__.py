"""Context management: ToolResultRefStore, SemanticCompressor, DataBus.

L1: ToolResultRefStore — store large tool results externally, keep ref in context.
L2: SemanticCompressor — LLM-based semantic compression of large results (future).
L3: Compaction — conversation-level compression (future).
L4: DataBus — on-demand retrieval of stored data (future).
"""

from .tool_result_store import ToolResultRefStore, StoredResult, make_get_stored_data_tool

__all__ = ["ToolResultRefStore", "StoredResult", "make_get_stored_data_tool"]