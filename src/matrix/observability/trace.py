"""Observability: structured tracing with SQLite storage and web panel.

Replaces the old JSONL TraceLogger with SQLite-backed structured storage
that supports querying by session_id, event type, and time range.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class TraceStore:
    """SQLite-backed trace event store. Thread-safe, WAL mode."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ---- public API ----

    def record(self, event: dict[str, Any]) -> None:
        """Record a trace event. Thread-safe, fast append."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO trace_events
                   (session_id, event_type, node_name, agent_id, tool_name,
                    ok, elapsed_ms, arguments, result_preview, error,
                    span_id, parent_span_id, ts)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.get("session_id", ""),
                    event.get("event_type", "unknown"),
                    event.get("node_name"),
                    event.get("agent_id"),
                    event.get("tool_name"),
                    _to_int(event.get("ok")),
                    event.get("elapsed_ms"),
                    _json_dump(event.get("arguments")),
                    _truncate(event.get("result")),
                    event.get("error"),
                    event.get("span_id"),
                    event.get("parent_span_id"),
                    event.get("ts", _now_ts()),
                ),
            )
            conn.commit()

    def query(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query trace events with optional filters."""
        conditions = []
        params: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT * FROM trace_events {where} "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent sessions with summary stats."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT
                 session_id,
                 MIN(ts) AS started,
                 MAX(ts) AS ended,
                 COUNT(*) AS total_events,
                 SUM(CASE WHEN ok = 0 THEN 1 ELSE 0 END) AS errors,
                 SUM(CASE WHEN event_type = 'tool_call' THEN 1 ELSE 0 END) AS tool_calls
               FROM trace_events
               WHERE session_id != ''
               GROUP BY session_id
               ORDER BY MIN(id) DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def session_detail(self, session_id: str) -> list[dict[str, Any]]:
        """Get all events for a session, ordered chronologically."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM trace_events WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Get overall trace statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM trace_events").fetchone()[0]
        errors = conn.execute(
            "SELECT COUNT(*) FROM trace_events WHERE ok = 0"
        ).fetchone()[0]
        sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM trace_events WHERE session_id != ''"
        ).fetchone()[0]
        return {
            "total_events": total,
            "total_errors": errors,
            "total_sessions": sessions,
        }

    def close(self) -> None:
        pass  # SQLite connections are short-lived per call

    # ---- internal ----

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL DEFAULT '',
            event_type TEXT NOT NULL DEFAULT 'unknown',
            node_name TEXT,
            agent_id TEXT,
            tool_name TEXT,
            ok INTEGER,
            elapsed_ms REAL,
            arguments TEXT,
            result_preview TEXT,
            error TEXT,
            span_id TEXT,
            parent_span_id TEXT,
            ts TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )""")
        # Migrate: add span_id column if missing (pre-existing DBs)
        _add_column_if_missing(conn, "trace_events", "span_id", "TEXT")
        _add_column_if_missing(conn, "trace_events", "parent_span_id", "TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trace_session ON trace_events(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trace_ts ON trace_events(ts)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trace_type ON trace_events(event_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trace_span ON trace_events(span_id)"
        )
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


# Keep TraceLogger as a thin compatibility alias
class TraceLogger(TraceStore):
    """Backward-compatible alias for TraceStore.

    The old JSONL TraceLogger has been replaced with SQLite-backed
    TraceStore.  This class exists so existing code that references
    TraceLogger continues to work without changes.
    """

    def __init__(self, path: Path):
        # Convert trace.jsonl path to trace.db path
        db_path = path.with_suffix(".db")
        super().__init__(db_path)


# ---- helpers ----

def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    return int(value)


def _json_dump(obj: Any) -> str | None:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)[:1000]


def _truncate(obj: Any, max_len: int = 500) -> str | None:
    if obj is None:
        return None
    s = str(obj)
    return s[:max_len] + ("..." if len(s) > max_len else "")


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    # Parse JSON fields
    for field in ("arguments",):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Add a column if it doesn't exist (safe migration for pre-existing DBs)."""
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")