"""Observability: structured tracing with SQLite storage and web panel.

Replaces the old JSONL TraceLogger with SQLite-backed structured storage
that supports querying by session_id, event type, and time range.

Now with OpenTelemetry-standardized span storage and OTLP export support.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .otel import (
    Span,
    Tracer,
    OTLPExporter,
    TraceContext,
    normalize_legacy_event,
    event_to_span_kind,
    event_to_status,
    STATUS_UNSET,
    SPAN_KIND_INTERNAL,
)

logger = logging.getLogger("matrix.observability")


class TraceStore:
    """SQLite-backed trace event store with OTel-standardized span support.

    Dual-mode storage:
    1. Legacy events: ``record()`` stores flat event dicts (backward compatible)
    2. OTel spans: ``record_span()`` stores OTel Span objects with full
       trace context (trace_id, span_id, parent_span_id, attributes, etc.)

    OTel spans can be exported to external backends via OTLP/JSON.
    """

    def __init__(
        self,
        db_path: Path,
        sanitizer: object | None = None,
        *,
        service_name: str = "matrix-agent",
        otlp_endpoint: str = "",
        otlp_export: bool = False,
    ):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._sanitizer = sanitizer  # TraceSanitizer or None
        self._tracer = Tracer(service_name=service_name)
        self._exporter = OTLPExporter(
            endpoint=otlp_endpoint if otlp_export else "",
            service_name=service_name,
        )
        self._otlp_export = otlp_export
        self._init_db()

    # ---- public API ----

    def record(self, event: dict[str, Any]) -> None:
        """Record a trace event. Thread-safe, fast append."""
        with self._lock:
            # ---- TRACE PRIVACY ----
            if self._sanitizer:
                event["arguments"] = self._sanitizer.sanitize(event.get("arguments"))
                event["result"] = self._sanitizer.sanitize(event.get("result"))
            # ---- END TRACE PRIVACY ----
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

    def record_span(self, span: Span) -> None:
        """Record an OTel Span to SQLite and optionally export via OTLP.

        The span is stored in a dedicated ``otel_spans`` table with
        standard OTel fields, while also being indexed for querying
        by trace_id, session_id, etc.
        """
        with self._lock:
            conn = self._get_conn()
            attrs_json = json.dumps(span.attributes, ensure_ascii=False, default=str)
            events_json = json.dumps(span.events, ensure_ascii=False, default=str)
            resource_json = json.dumps(span.resource, ensure_ascii=False, default=str)

            # Extract session.id from attributes for indexing
            session_id = span.attributes.get("session.id", "")

            conn.execute(
                """INSERT INTO otel_spans
                   (trace_id, span_id, parent_span_id, name, kind,
                    start_time_unix_nano, end_time_unix_nano,
                    status_code, status_message,
                    attributes, events, resource, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    span.trace_id,
                    span.span_id,
                    span.parent_span_id,
                    span.name,
                    span.kind,
                    span.start_time_unix_nano,
                    span.end_time_unix_nano,
                    span.status_code,
                    span.status_message,
                    attrs_json,
                    events_json,
                    resource_json,
                    session_id,
                ),
            )
            conn.commit()

        # Async OTLP export (non-blocking)
        if self._otlp_export:
            try:
                self._exporter.export([span])
            except Exception as e:
                logger.debug("otlp_export_error: %s", e)

    @property
    def tracer(self) -> Tracer:
        """Access the internal OTel Tracer for creating spans."""
        return self._tracer

    def start_span(
        self,
        name: str,
        *,
        session_id: str = "",
        agent_id: str = "",
        kind: int = SPAN_KIND_INTERNAL,
        **attributes: Any,
    ) -> Span:
        """Start an OTel span. Use ``end_span()`` to finalize.

        Convenience method that delegates to the internal Tracer.
        """
        return self._tracer.start_span(
            name, session_id=session_id, agent_id=agent_id,
            kind=kind, **attributes,
        )

    def end_span(self, span: Span, *, status: int | None = None, message: str = "") -> None:
        """End a span and record it to SQLite + OTLP.

        Args:
            span: The span to end
            status: OTel StatusCode (optional, defaults to OK if no error)
            message: Status message (for errors)
        """
        self._tracer.end_span(span)
        if status is not None:
            span.set_status(status, message)
        elif span.status_code == STATUS_UNSET:
            span.set_status(1)  # STATUS_OK
        self.record_span(span)

    def query_spans(
        self,
        trace_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query OTel spans by trace_id or session_id."""
        conditions = []
        params: list[Any] = []
        if trace_id:
            conditions.append("trace_id = ?")
            params.append(trace_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT * FROM otel_spans {where} "
            "ORDER BY start_time_unix_nano DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [_span_row_to_dict(r) for r in rows]

    def export_otlp(self) -> list[dict[str, Any]]:
        """Get buffered OTLP exports (for debugging/testing)."""
        return self._exporter.get_buffered()

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
            "CREATE INDEX IF NOT EXISTS idx_trace_span ON trace_events(span_id)"
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
        # OTel spans table (standardized span storage)
        conn.execute("""CREATE TABLE IF NOT EXISTS otel_spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            span_id TEXT NOT NULL,
            parent_span_id TEXT NOT NULL DEFAULT '',
            name TEXT NOT NULL DEFAULT '',
            kind INTEGER NOT NULL DEFAULT 0,
            start_time_unix_nano INTEGER NOT NULL,
            end_time_unix_nano INTEGER NOT NULL DEFAULT 0,
            status_code INTEGER NOT NULL DEFAULT 0,
            status_message TEXT NOT NULL DEFAULT '',
            attributes TEXT NOT NULL DEFAULT '{}',
            events TEXT NOT NULL DEFAULT '[]',
            resource TEXT NOT NULL DEFAULT '{}',
            session_id TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        )""")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_otel_trace ON otel_spans(trace_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_otel_session ON otel_spans(session_id)"
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

    def __init__(
        self,
        path: Path,
        sanitizer: object | None = None,
        *,
        service_name: str = "matrix-agent",
        otlp_endpoint: str = "",
        otlp_export: bool = False,
    ):
        # Convert trace.jsonl path to trace.db path
        db_path = path.with_suffix(".db")
        super().__init__(
            db_path,
            sanitizer=sanitizer,
            service_name=service_name,
            otlp_endpoint=otlp_endpoint,
            otlp_export=otlp_export,
        )


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


def _span_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an otel_spans row to a dict, parsing JSON fields."""
    d = dict(row)
    for field in ("attributes", "events", "resource"):
        val = d.get(field)
        if val and isinstance(val, str):
            try:
                d[field] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
    # Add convenience fields
    start = d.get("start_time_unix_nano", 0)
    end = d.get("end_time_unix_nano", 0)
    if start and end:
        d["elapsed_ms"] = round((end - start) / 1e6, 3)
    else:
        d["elapsed_ms"] = 0.0
    return d