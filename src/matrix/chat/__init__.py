"""Chat service package.

Re-exports ChatService and utilities for backward compatibility
with the original chat.py module.
"""

from __future__ import annotations

from ._service import ChatService
from ._utils import (
    MEMORY_EXTRACTION_PROMPT,
    preview_json,
    result_count,
    timestamp,
)

__all__ = [
    "ChatService",
    "MEMORY_EXTRACTION_PROMPT",
    "preview_json",
    "result_count",
    "timestamp",
]