"""L1 ToolResultRefStore: externalize large tool results from LLM context.

Design principles (from the article):
- Single-representation principle: data appears in exactly ONE form in context
- Deterministic over intelligent: no LLM in the storage pipeline
- Threshold-based: >8000 chars or >10 array elements triggers externalization
- Preview is for debugging only, never enters LLM context
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("matrix.context")


@dataclass
class StoredResult:
    """A stored tool result with metadata."""
    ref_id: str
    tool_name: str
    original_length: int
    summary: str
    preview: str  # for debugging only, never enters LLM prompt
    stored_at: float
    ttl_seconds: int = 3600  # 1 hour default


_STORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS stored_results (
    ref_id       TEXT PRIMARY KEY,
    tool_name    TEXT NOT NULL,
    full_data    TEXT NOT NULL,
    summary      TEXT NOT NULL DEFAULT '',
    preview      TEXT NOT NULL DEFAULT '',
    original_length INTEGER NOT NULL DEFAULT 0,
    stored_at    REAL NOT NULL,
    ttl_seconds  INTEGER NOT NULL DEFAULT 3600
);

CREATE INDEX IF NOT EXISTS idx_stored_results_stored_at
    ON stored_results(stored_at);
"""

# Thresholds
MAX_CHARS_INLINE = 8000        # >8000 chars → externalize
MAX_ARRAY_ITEMS_INLINE = 10    # >10 items → externalize
PREVIEW_MAX_CHARS = 12000      # preview cap (for debugging, never in prompt)


class ToolResultRefStore:
    """Thread-safe SQLite store for large tool results.

    Usage:
        store = ToolResultRefStore(db_path)
        ref = store.store("web_search", large_result)
        # In context: only the ref object is kept
        data = store.get(ref.ref_id)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_STORE_SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ---- Public API ----

    def should_store(self, result: Any) -> bool:
        """Check if a tool result should be externalized.

        Returns True if the result exceeds inline thresholds.
        """
        result_str = self._to_string(result)
        char_count = len(result_str)

        if char_count > MAX_CHARS_INLINE:
            return True

        # Check array element count
        if isinstance(result, list) and len(result) > MAX_ARRAY_ITEMS_INLINE:
            return True

        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list) and len(v) > MAX_ARRAY_ITEMS_INLINE:
                    return True

        return False

    def store(self, tool_name: str, result: Any, ttl_seconds: int = 3600) -> StoredResult:
        """Store a large tool result external to the context.

        Returns a StoredResult with the ref_id and summary.
        The original result is stored in SQLite; only the ref object
        should be included in the LLM context.
        """
        result_str = self._to_string(result)
        ref_id = uuid.uuid4().hex[:16]
        original_length = len(result_str)
        summary = self._build_summary(result)
        preview = result_str[:PREVIEW_MAX_CHARS]
        stored_at = time.time()

        with self._lock:
            self._get_conn().execute(
                """INSERT INTO stored_results
                   (ref_id, tool_name, full_data, summary, preview,
                    original_length, stored_at, ttl_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (ref_id, tool_name, result_str, summary, preview,
                 original_length, stored_at, ttl_seconds),
            )
            self._get_conn().commit()

        logger.debug(
            "stored_result: ref_id=%s tool=%s original_len=%d summary_len=%d",
            ref_id, tool_name, original_length, len(summary),
        )

        return StoredResult(
            ref_id=ref_id,
            tool_name=tool_name,
            original_length=original_length,
            summary=summary,
            preview=preview,
            stored_at=stored_at,
            ttl_seconds=ttl_seconds,
        )

    def get(self, ref_id: str) -> Any | None:
        """Retrieve the full stored data by ref_id.

        Returns the parsed JSON object, or None if not found or expired.
        """
        with self._lock:
            row = self._get_conn().execute(
                """SELECT full_data, stored_at, ttl_seconds
                   FROM stored_results WHERE ref_id = ?""",
                (ref_id,),
            ).fetchone()

        if row is None:
            return None

        full_data, stored_at, ttl_seconds = row
        if time.time() - stored_at > ttl_seconds:
            self._delete(ref_id)
            return None

        try:
            return json.loads(full_data)
        except (json.JSONDecodeError, TypeError):
            return full_data

    def get_summary(self, ref_id: str) -> str | None:
        """Get just the summary for a stored result."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT summary FROM stored_results WHERE ref_id = ?",
                (ref_id,),
            ).fetchone()
        return row[0] if row else None

    def build_ref_object(self, stored: StoredResult) -> dict[str, Any]:
        """Build the reference object to include in LLM context.

        This is the ONLY representation of the data that enters the LLM context.
        Per the single-representation principle, the full data is NEVER
        included alongside this ref object.
        """
        return {
            "__stored": True,
            "__refId": stored.ref_id,
            "__toolType": stored.tool_name,
            "__originalLength": stored.original_length,
            "__summary": stored.summary,
            "__hint": (
                f"Call get_stored_data(refId=\"{stored.ref_id}\") "
                f"to retrieve the full data"
            ),
        }

    def cleanup_expired(self) -> int:
        """Delete expired stored results. Returns count deleted."""
        now = time.time()
        with self._lock:
            cur = self._get_conn().execute(
                "DELETE FROM stored_results WHERE stored_at + ttl_seconds < ?",
                (now,),
            )
            self._get_conn().commit()
            count = cur.rowcount
            if count > 0:
                logger.debug("cleanup_expired: deleted %d expired results", count)
            return count

    # ---- Internal ----

    def _delete(self, ref_id: str) -> None:
        with self._lock:
            self._get_conn().execute(
                "DELETE FROM stored_results WHERE ref_id = ?", (ref_id,),
            )
            self._get_conn().commit()

    @staticmethod
    def _to_string(result: Any) -> str:
        """Convert any result to a JSON string for storage."""
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    @staticmethod
    def _build_summary(result: Any) -> str:
        """Build a compact summary of the result.

        This summary is deterministic — no LLM involved.
        It provides enough info for the LLM to decide whether to fetch full data.
        """
        if isinstance(result, dict):
            keys = list(result.keys())[:20]
            key_summary = ", ".join(keys)
            if len(result) > 20:
                key_summary += f" ... (+{len(result) - 20} more)"
            return f"Object with {len(result)} keys: {{{key_summary}}}"

        if isinstance(result, list):
            item_count = len(result)
            if item_count == 0:
                return "Empty list"
            # Show type of first item
            first_type = type(result[0]).__name__
            return f"Array of {item_count} {first_type} items"

        if isinstance(result, str):
            return f"String, {len(result)} chars"

        return f"{type(result).__name__}, {len(str(result))} chars"


def make_get_stored_data_tool(store: ToolResultRefStore):
    """Create a tool definition for retrieving stored data by refId.

    This tool is registered in the ToolRegistry and exposed to the LLM.
    """
    def get_stored_data(refId: str) -> dict[str, Any]:
        """Retrieve full data that was externalized from the context.

        Args:
            refId: The reference ID from a __refId field in a stored result reference.

        Returns:
            The full stored data, or an error dict if not found.
        """
        data = store.get(refId)
        if data is None:
            return {"error": f"Stored data not found or expired: {refId}"}
        return {"refId": refId, "data": data}

    return get_stored_data