"""SQLite-backed session and conversation history store.

Multi-user support: users table with bcrypt password hashes, user_id column
on sessions and messages for data isolation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT '',
    title       TEXT NOT NULL DEFAULT '',
    provider    TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    msg_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL DEFAULT '',
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS user_profile (
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""


class SessionStore:
    """Thread-safe SQLite store for users, sessions, and messages."""

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
            self._migrate()
            self._conn.commit()
        return self._conn

    def _migrate(self) -> None:
        """Add any missing columns to existing tables."""
        assert self._conn is not None

        # sessions: add provider, model, user_id if missing
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(sessions)").fetchall()]
        if "provider" not in cols:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN provider TEXT NOT NULL DEFAULT ''")
        if "model" not in cols:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN model TEXT NOT NULL DEFAULT ''")
        if "user_id" not in cols:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")

        # messages: add user_id if missing
        msg_cols = [r[1] for r in self._conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "user_id" not in msg_cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")

        # Create idx_sessions_user after user_id column is guaranteed to exist
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at)"
        )

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ---- User CRUD ----

    def get_user(self, username: str) -> dict[str, Any] | None:
        """Get a user by username (id). Returns dict with id, password_hash or None."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT id, password_hash FROM users WHERE id=?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "password_hash": row[1]}

    def create_user(self, user_id: str, password_hash: str) -> bool:
        """Create a new user. Returns True if created, False if already exists."""
        with self._lock:
            try:
                self._get_conn().execute(
                    "INSERT INTO users (id, password_hash, created_at) VALUES (?, ?, ?)",
                    (user_id, password_hash, time.time()),
                )
                self._get_conn().commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def user_exists(self, user_id: str) -> bool:
        """Check if a user exists."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT 1 FROM users WHERE id=?",
                (user_id,),
            ).fetchone()
        return row is not None

    def user_count(self) -> int:
        """Return the total number of users."""
        with self._lock:
            row = self._get_conn().execute("SELECT COUNT(*) FROM users").fetchone()
        return row[0] if row else 0

    # ---- Session CRUD ----

    def list_sessions(self, user_id: str = "", limit: int = 20) -> list[dict[str, Any]]:
        """Return recent sessions for a user, ordered by updated_at desc."""
        with self._lock:
            if user_id:
                rows = self._get_conn().execute(
                    "SELECT id, title, created_at, updated_at, msg_count "
                    "FROM sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            else:
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
                "turn_count": r[4] // 2,
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
            "turn_count": row[4] // 2,
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

    # ---- Provider ----

    def get_provider(self, session_id: str) -> str:
        """Get the LLM provider assigned to a session."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT provider FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        return row[0] if row and row[0] else ""

    def get_model(self, session_id: str) -> str:
        """Get the LLM model assigned to a session."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT model FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
        return row[0] if row and row[0] else ""

    def set_provider(self, session_id: str, provider: str, model: str = "", user_id: str = "") -> None:
        """Set the LLM provider and optionally model for a session."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO sessions (id, user_id, provider, model, created_at, updated_at, msg_count) "
                "VALUES (?, ?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(id) DO UPDATE SET provider=excluded.provider"
                + (", model=excluded.model" if model else "")
                + ", updated_at=excluded.updated_at",
                (session_id, user_id, provider, model, time.time(), time.time()),
            )
            conn.commit()

    # ---- Message CRUD ----

    def save_message(self, session_id: str, role: str, content: str, user_id: str = "") -> None:
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            # Upsert session
            conn.execute(
                "INSERT INTO sessions (id, user_id, title, created_at, updated_at, msg_count) "
                "VALUES (?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  updated_at=excluded.updated_at, "
                "  msg_count=sessions.msg_count + 1",
                (session_id, user_id, "", now, now),
            )
            # Insert message
            conn.execute(
                "INSERT INTO messages (session_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, role, content, now),
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
                "UPDATE sessions SET title = COALESCE(("
                "  SELECT substr(content, 1, 30) FROM messages "
                "  WHERE messages.session_id = sessions.id AND messages.role = 'user' "
                "  ORDER BY messages.created_at ASC LIMIT 1"
                "), '') WHERE title = '' OR title IS NULL"
            )
            self._get_conn().commit()
            return cur.rowcount

    # ---- User Profile (long-term memory) ----

    def get_profile(self, user_id: str) -> dict[str, str]:
        """Return all key-value pairs for a user."""
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT key, value FROM user_profile WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def upsert_profile(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a profile entry."""
        with self._lock:
            self._get_conn().execute(
                "INSERT INTO user_profile (user_id, key, value, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (user_id, key, value, time.time()),
            )
            self._get_conn().commit()

    def delete_profile_key(self, user_id: str, key: str) -> bool:
        """Delete a profile entry. Returns True if deleted."""
        with self._lock:
            cur = self._get_conn().execute(
                "DELETE FROM user_profile WHERE user_id=? AND key=?",
                (user_id, key),
            )
            self._get_conn().commit()
            return cur.rowcount > 0

    def sync_profile_from_file(self, user_id: str, json_path: str) -> int:
        """Load profile from JSON file into SQLite. Returns count of entries synced."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0
        if not isinstance(data, dict):
            return 0
        count = 0
        for key, value in data.items():
            if isinstance(value, str) and key.strip() and value.strip():
                self.upsert_profile(user_id, key.strip(), value.strip())
                count += 1
        return count

    def sync_profile_to_file(self, user_id: str, json_path: str) -> bool:
        """Export SQLite profile to JSON file. Returns True on success."""
        profile = self.get_profile(user_id)
        if not profile:
            return False
        try:
            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            return True
        except OSError:
            return False


def _default_title(session_id: str) -> str:
    return session_id.split("-", 1)[-1][:8]