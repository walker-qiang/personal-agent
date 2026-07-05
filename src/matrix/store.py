"""SQLite-backed session and conversation history store.

Replaces the in-memory ``dict[str, list[dict]]`` with a durable store that
survives server restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    msg_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);
"""


class SessionStore:
    """Thread-safe SQLite store for conversation sessions and messages."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    # ---- Connection management ----

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ---- Session CRUD ----

    def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent sessions ordered by updated_at desc."""
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT id, title, created_at, updated_at, msg_count "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "title": r[1] or _default_title(r[0]),
                "created_at": r[2],
                "updated_at": r[3],
                "msg_count": r[4],
            }
            for r in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT id, title, created_at, updated_at, msg_count FROM sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "title": row[1] or _default_title(row[0]),
            "created_at": row[2],
            "updated_at": row[3],
            "msg_count": row[4],
        }

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cur = self._get_conn().execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._get_conn().commit()
            return cur.rowcount > 0

    def update_title(self, session_id: str, title: str) -> None:
        """Set the session title (e.g. from first user message)."""
        with self._lock:
            self._get_conn().execute(
                "UPDATE sessions SET title=? WHERE id=? AND (title='' OR title IS NULL)",
                (title, session_id),
            )
            self._get_conn().commit()

    # ---- Message CRUD ----

    def save_message(self, session_id: str, role: str, content: str) -> None:
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            # Upsert session
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, msg_count) "
                "VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  updated_at=excluded.updated_at, "
                "  msg_count=sessions.msg_count + 1",
                (session_id, "", now, now),
            )
            # Insert message
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            conn.commit()

    def get_history(self, session_id: str, max_turns: int = 8) -> list[dict[str, str]]:
        """Return the last N turns of conversation history."""
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT role, content FROM messages "
                "WHERE session_id=? ORDER BY created_at ASC LIMIT ?",
                (session_id, max_turns * 2),
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def reset(self, session_id: str) -> None:
        """Delete all messages for a session (keeps session metadata)."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute(
                "UPDATE sessions SET msg_count=0, updated_at=? WHERE id=?",
                (time.time(), session_id),
            )
            conn.commit()

    def prune(self, max_age_days: int = 30) -> int:
        """Delete sessions older than max_age_days. Returns count deleted."""
        cutoff = time.time() - max_age_days * 86400
        with self._lock:
            cur = self._get_conn().execute(
                "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
            )
            self._get_conn().commit()
            return cur.rowcount

    def backfill_titles(self) -> int:
        """Set titles for sessions that have messages but no title."""
        with self._lock:
            cur = self._get_conn().execute(
                "UPDATE sessions SET title = ("
                "  SELECT substr(content, 1, 30) FROM messages "
                "  WHERE messages.session_id = sessions.id AND messages.role = 'user' "
                "  ORDER BY messages.created_at ASC LIMIT 1"
                ") WHERE title = '' OR title IS NULL"
            )
            self._get_conn().commit()
            return cur.rowcount


def _default_title(session_id: str) -> str:
    # Show last 8 chars of session ID (after the hyphen prefix)
    return session_id.split("-", 1)[-1][:8]