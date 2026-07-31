"""SQLite-backed session and conversation history store.

Multi-user support: users table with bcrypt password hashes, user_id column
on sessions and messages for data isolation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
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
    leaf_id     TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    msg_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  TEXT NOT NULL DEFAULT '',
    parent_id   TEXT,
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
    memory_type TEXT NOT NULL DEFAULT 'preference',
    created_at  REAL NOT NULL DEFAULT 0,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""


class SessionStore:
    """Thread-safe SQLite store for users, sessions, and messages."""

    # Memory decay: half-life in seconds (30 days)
    MEMORY_HALF_LIFE = 30 * 86400

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

        # user_profile: add memory_type and created_at if missing
        prof_cols = [r[1] for r in self._conn.execute("PRAGMA table_info(user_profile)").fetchall()]
        if "memory_type" not in prof_cols:
            self._conn.execute(
                "ALTER TABLE user_profile ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'preference'"
            )
        if "created_at" not in prof_cols:
            self._conn.execute(
                "ALTER TABLE user_profile ADD COLUMN created_at REAL NOT NULL DEFAULT 0"
            )

        # sessions: add hidden column if missing
        if "hidden" not in cols:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0"
            )

        # Create idx_sessions_user after user_id column is guaranteed to exist
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at)"
        )

        # Phase 7: Session tree — add message_id, parent_id, leaf_id
        msg_cols = [r[1] for r in self._conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "message_id" not in msg_cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN message_id TEXT NOT NULL DEFAULT ''")
        if "parent_id" not in msg_cols:
            self._conn.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT")
        # Backfill message_id for existing rows
        self._conn.execute(
            "UPDATE messages SET message_id = 'm' || CAST(id AS TEXT) WHERE message_id = ''"
        )
        # Backfill parent_id: each message's parent is the previous message in the same session
        self._conn.execute("""
            UPDATE messages SET parent_id = (
                SELECT m2.message_id FROM messages m2
                WHERE m2.session_id = messages.session_id
                  AND m2.id < messages.id
                ORDER BY m2.id DESC LIMIT 1
            ) WHERE parent_id IS NULL
        """)

        sess_cols = [r[1] for r in self._conn.execute("PRAGMA table_info(sessions)").fetchall()]
        if "leaf_id" not in sess_cols:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN leaf_id TEXT")
        # Backfill leaf_id: last message in each session
        self._conn.execute("""
            UPDATE sessions SET leaf_id = (
                SELECT message_id FROM messages
                WHERE messages.session_id = sessions.id
                ORDER BY messages.id DESC LIMIT 1
            ) WHERE leaf_id IS NULL
        """)

        # Create indexes for message_id and parent_id (after columns are added)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_message_id ON messages(message_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_parent_id ON messages(parent_id)"
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

    def list_sessions(self, user_id: str = "", limit: int = 20,
                      include_hidden: bool = False) -> list[dict[str, Any]]:
        """Return recent sessions for a user, ordered by updated_at desc.

        By default excludes hidden (archived) sessions. Set include_hidden=True
        to show all sessions.
        """
        with self._lock:
            if user_id:
                if include_hidden:
                    rows = self._get_conn().execute(
                        "SELECT id, title, created_at, updated_at, msg_count, hidden "
                        "FROM sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
                        (user_id, limit),
                    ).fetchall()
                else:
                    rows = self._get_conn().execute(
                        "SELECT id, title, created_at, updated_at, msg_count, hidden "
                        "FROM sessions WHERE user_id=? AND hidden=0 ORDER BY updated_at DESC LIMIT ?",
                        (user_id, limit),
                    ).fetchall()
            else:
                if include_hidden:
                    rows = self._get_conn().execute(
                        "SELECT id, title, created_at, updated_at, msg_count, hidden "
                        "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                else:
                    rows = self._get_conn().execute(
                        "SELECT id, title, created_at, updated_at, msg_count, hidden "
                        "FROM sessions WHERE hidden=0 ORDER BY updated_at DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
        return [
            {
                "id": r[0],
                "title": r[1] or _default_title(r[0]),
                "created_at": r[2],
                "updated_at": r[3],
                "turn_count": r[4] // 2,
                "hidden": bool(r[5]),
            }
            for r in rows
        ]

    def get_session(self, session_id: str, user_id: str = "") -> dict[str, Any] | None:
        with self._lock:
            if user_id:
                row = self._get_conn().execute(
                    "SELECT id, title, created_at, updated_at, msg_count, hidden "
                    "FROM sessions WHERE id=? AND user_id=?",
                    (session_id, user_id),
                ).fetchone()
            else:
                row = self._get_conn().execute(
                    "SELECT id, title, created_at, updated_at, msg_count, hidden "
                    "FROM sessions WHERE id=?",
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
            "hidden": bool(row[5]),
        }

    def delete_session(self, session_id: str, user_id: str = "") -> bool:
        with self._lock:
            if user_id:
                cur = self._get_conn().execute(
                    "DELETE FROM sessions WHERE id=? AND user_id=?",
                    (session_id, user_id),
                )
            else:
                cur = self._get_conn().execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._get_conn().commit()
            return cur.rowcount > 0

    def batch_delete_sessions(self, session_ids: list[str], user_id: str = "") -> int:
        """Delete multiple sessions and their messages. Returns count deleted."""
        if not session_ids:
            return 0
        with self._lock:
            conn = self._get_conn()
            placeholders = ",".join("?" for _ in session_ids)
            sql = f"DELETE FROM sessions WHERE id IN ({placeholders})"
            params: tuple[Any, ...] = tuple(session_ids)
            if user_id:
                sql += " AND user_id=?"
                params += (user_id,)
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount

    def batch_set_hidden(self, session_ids: list[str], hidden: bool = True, user_id: str = "") -> int:
        """Set hidden (archive) state for multiple sessions. Returns count updated."""
        if not session_ids:
            return 0
        val = 1 if hidden else 0
        with self._lock:
            conn = self._get_conn()
            placeholders = ",".join("?" for _ in session_ids)
            sql = f"UPDATE sessions SET hidden=? WHERE id IN ({placeholders})"
            params: tuple[Any, ...] = (val, *session_ids)
            if user_id:
                sql += " AND user_id=?"
                params += (user_id,)
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.rowcount

    def update_title(self, session_id: str, title: str, user_id: str = "") -> None:
        """Set the session title (e.g. from first user message)."""
        with self._lock:
            sql = "UPDATE sessions SET title=? WHERE id=? AND (title='' OR title IS NULL)"
            params: tuple[Any, ...] = (title, session_id)
            if user_id:
                sql += " AND user_id=?"
                params += (user_id,)
            self._get_conn().execute(sql, params)
            self._get_conn().commit()

    # ---- Provider ----

    def get_provider(self, session_id: str, user_id: str = "") -> str:
        """Get the LLM provider assigned to a session."""
        with self._lock:
            if user_id:
                row = self._get_conn().execute(
                    "SELECT provider FROM sessions WHERE id=? AND user_id=?",
                    (session_id, user_id),
                ).fetchone()
            else:
                row = self._get_conn().execute(
                    "SELECT provider FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
        return row[0] if row and row[0] else ""

    def get_model(self, session_id: str, user_id: str = "") -> str:
        """Get the LLM model assigned to a session."""
        with self._lock:
            if user_id:
                row = self._get_conn().execute(
                    "SELECT model FROM sessions WHERE id=? AND user_id=?",
                    (session_id, user_id),
                ).fetchone()
            else:
                row = self._get_conn().execute(
                    "SELECT model FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
        return row[0] if row and row[0] else ""

    def set_provider(self, session_id: str, provider: str, model: str = "", user_id: str = "") -> bool:
        """Set the LLM provider and optionally model for a session."""
        with self._lock:
            conn = self._get_conn()
            if user_id:
                existing = conn.execute(
                    "SELECT user_id FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
                if existing is not None and existing[0] != user_id:
                    return False
            conn.execute(
                "INSERT INTO sessions (id, user_id, provider, model, created_at, updated_at, msg_count) "
                "VALUES (?, ?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(id) DO UPDATE SET provider=excluded.provider"
                + (", model=excluded.model" if model else "")
                + ", updated_at=excluded.updated_at"
                + ", user_id=CASE WHEN sessions.user_id='' THEN excluded.user_id ELSE sessions.user_id END",
                (session_id, user_id, provider, model, time.time(), time.time()),
            )
            conn.commit()
            return True

    # ---- Message CRUD ----

    def save_message(self, session_id: str, role: str, content: str, user_id: str = "") -> str:
        """Append a message to the session tree.

        Returns the message_id of the newly created message.
        The message's parent_id is the session's current leaf_id.
        """
        now = time.time()
        msg_id = uuid.uuid4().hex[:12]
        with self._lock:
            conn = self._get_conn()
            # Get current leaf_id for parent
            row = conn.execute(
                "SELECT leaf_id, user_id FROM sessions WHERE id=?", (session_id,),
            ).fetchone()
            if row and user_id and row[1] not in ("", user_id):
                raise ValueError(f"session belongs to another user: {session_id}")
            parent_id = row[0] if row else None

            # Upsert session
            conn.execute(
                "INSERT INTO sessions (id, user_id, title, created_at, updated_at, msg_count, leaf_id) "
                "VALUES (?, ?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  updated_at=excluded.updated_at, "
                "  msg_count=sessions.msg_count + 1, "
                "  leaf_id=excluded.leaf_id, "
                "  user_id=CASE WHEN sessions.user_id='' THEN excluded.user_id ELSE sessions.user_id END",
                (session_id, user_id, "", now, now, msg_id),
            )
            # Insert message
            conn.execute(
                "INSERT INTO messages (message_id, parent_id, session_id, user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (msg_id, parent_id, session_id, user_id, role, content, now),
            )
            conn.commit()
        return msg_id

    def get_history(self, session_id: str, max_turns: int = 8, user_id: str = "") -> list[dict[str, str]]:
        """Return conversation history by traversing from leaf to root.

        Walks the parent_id chain from the session's leaf_id backward,
        collecting up to max_turns*2 messages, then reverses to
        chronological order.
        """
        with self._lock:
            conn = self._get_conn()
            # Get current leaf_id
            if user_id:
                row = conn.execute(
                    "SELECT leaf_id FROM sessions WHERE id=? AND user_id=?",
                    (session_id, user_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT leaf_id FROM sessions WHERE id=?", (session_id,),
                ).fetchone()
            if row is None:
                return []

            leaf_id = row[0]
            if not leaf_id:
                # Fallback: no leaf_id set, use linear query
                rows = conn.execute(
                    "SELECT role, content FROM messages "
                    "WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                    (session_id, max_turns * 2),
                ).fetchall()
                rows = list(reversed(rows))
                return [{"role": r[0], "content": r[1], "message_id": ""} for r in rows]

            # Traverse from leaf to root
            path: list[dict[str, str]] = []
            current = leaf_id
            while current and len(path) < max_turns * 2:
                msg_row = conn.execute(
                    "SELECT message_id, parent_id, role, content FROM messages WHERE message_id=?",
                    (current,),
                ).fetchone()
                if msg_row is None:
                    break
                path.append({"role": msg_row[2], "content": msg_row[3], "message_id": msg_row[0]})
                current = msg_row[1]  # parent_id

            path.reverse()
            return path

    def branch(self, session_id: str, from_message_id: str, user_id: str = "") -> bool:
        """Move the session's leaf_id back to a specific message.

        This creates a fork point — new messages will be appended as
        children of from_message_id. Existing messages on the old branch
        are preserved (append-only).

        Returns True if the branch was successful, False if the message
        was not found in this session.
        """
        with self._lock:
            conn = self._get_conn()
            sql = "SELECT 1 FROM messages WHERE message_id=? AND session_id=?"
            params: tuple[Any, ...] = (from_message_id, session_id)
            if user_id:
                sql += " AND user_id=?"
                params += (user_id,)
            row = conn.execute(sql, params).fetchone()
            if row is None:
                return False
            sql = "UPDATE sessions SET leaf_id=?, updated_at=? WHERE id=?"
            params = (from_message_id, time.time(), session_id)
            if user_id:
                sql += " AND user_id=?"
                params += (user_id,)
            conn.execute(sql, params)
            conn.commit()
            return True

    def get_leaf_id(self, session_id: str, user_id: str = "") -> str | None:
        """Return the current leaf_id for a session."""
        with self._lock:
            if user_id:
                row = self._get_conn().execute(
                    "SELECT leaf_id FROM sessions WHERE id=? AND user_id=?",
                    (session_id, user_id),
                ).fetchone()
            else:
                row = self._get_conn().execute(
                    "SELECT leaf_id FROM sessions WHERE id=?", (session_id,),
                ).fetchone()
            return row[0] if row else None

    def get_branches(self, session_id: str, user_id: str = "") -> list[dict[str, Any]]:
        """Return all fork points (messages with multiple children) in a session."""
        with self._lock:
            sql = (
                "SELECT parent_id, COUNT(*) as cnt FROM messages "
                "WHERE session_id=? AND parent_id IS NOT NULL"
            )
            params: tuple[Any, ...] = (session_id,)
            if user_id:
                sql += " AND user_id=?"
                params += (user_id,)
            sql += " GROUP BY parent_id HAVING cnt > 1"
            rows = self._get_conn().execute(sql, params).fetchall()
            return [{"fork_point": r[0], "branch_count": r[1]} for r in rows]

    def reset(self, session_id: str, user_id: str = "") -> bool:
        """Delete all messages for a session (keeps session metadata)."""
        with self._lock:
            conn = self._get_conn()
            where = "session_id=?"
            params: tuple[Any, ...] = (session_id,)
            if user_id:
                where += " AND user_id=?"
                params += (user_id,)
            conn.execute(f"DELETE FROM messages WHERE {where}", params)
            sql = "UPDATE sessions SET msg_count=0, leaf_id=NULL, updated_at=? WHERE id=?"
            update_params: tuple[Any, ...] = (time.time(), session_id)
            if user_id:
                sql += " AND user_id=?"
                update_params += (user_id,)
            cur = conn.execute(sql, update_params)
            conn.commit()
            return cur.rowcount > 0

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

    def get_profile(self, user_id: str) -> dict[str, str]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT key, value FROM user_profile WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_all_memories(self, user_id: str) -> list[dict]:
        """Return all memories with full metadata for evolution processing.

        Returns list of dicts with keys: key, value, memory_type, created_at, updated_at
        """
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT key, value, memory_type, created_at, updated_at "
                "FROM user_profile WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [
            {
                "key": r[0],
                "value": r[1],
                "memory_type": r[2],
                "created_at": r[3],
                "updated_at": r[4],
            }
            for r in rows
        ]

    def count_memories(self, user_id: str) -> int:
        """Return total number of memories for a user."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT COUNT(*) FROM user_profile WHERE user_id=?",
                (user_id,),
            ).fetchone()
        return row[0] if row else 0

    def get_policies(self, user_id: str) -> dict[str, str]:
        """Return policy-type memories (hard rules, highest priority)."""
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT key, value FROM user_profile "
                "WHERE user_id=? AND memory_type='policy' ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_preferences(self, user_id: str) -> dict[str, str]:
        """Return preference-type memories (user preferences, lower priority)."""
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT key, value FROM user_profile "
                "WHERE user_id=? AND memory_type='preference' ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_profile_formatted(self, user_id: str) -> str:
        """Return profile as formatted text for system prompt injection.

        Policies are presented first and marked as [HARD RULE] because they
        cannot be overridden by user instructions. Preferences are presented
        as [PREFERENCE], with time-decay weights applied.

        Memories older than the half-life (30 days) are down-weighted by 50%,
        and those older than 4 half-lives (120 days) are excluded entirely
        unless they are policies (which never decay).
        """
        now = time.time()
        policies = self._get_decayed_memories(user_id, "policy", now)
        preferences = self._get_decayed_memories(user_id, "preference", now)

        parts: list[str] = []
        if policies:
            parts.append("## Hard Rules (DO NOT override these)")
            for k, v, weight, updated_at in policies:
                age_days = (now - updated_at) / 86400
                age_hint = ""
                if age_days > 90:
                    age_hint = f" [established {int(age_days)}d ago]"
                parts.append(f"- [HARD RULE] {k}: {v}{age_hint}")
        if preferences:
            parts.append("\n## User Preferences")
            for k, v, weight, _ in preferences:
                if weight < 0.5:
                    parts.append(f"- [PREFERENCE, fading] {k}: {v}")
                else:
                    parts.append(f"- [PREFERENCE] {k}: {v}")
        return "\n".join(parts) if parts else ""

    def _get_decayed_memories(
        self, user_id: str, memory_type: str, now: float,
    ) -> list[tuple[str, str, float, float]]:
        """Get memories with time-decay weights applied.

        Decay formula: weight = 0.5^(age / half_life)
        - age = 0 days → weight = 1.0
        - age = 30 days → weight = 0.5
        - age = 60 days → weight = 0.25
        - age >= 120 days → excluded (weight < 0.0625) unless policy

        Returns list of (key, value, weight, updated_at) sorted by weight descending.
        Policies never decay below 1.0.
        """
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT key, value, updated_at FROM user_profile "
                "WHERE user_id=? AND memory_type=? ORDER BY updated_at DESC",
                (user_id, memory_type),
            ).fetchall()

        results: list[tuple[str, str, float, float]] = []
        for key, value, updated_at in rows:
            age = now - updated_at
            if memory_type == "policy":
                weight = 1.0  # Policies never decay
            else:
                weight = 2.0 ** (-age / self.MEMORY_HALF_LIFE)
                if weight < 0.0625:  # 4 half-lives = 120 days
                    continue  # Exclude very stale memories
            results.append((key, value, round(weight, 3), updated_at))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def _get_updated_at(self, user_id: str, key: str) -> float:
        """Get the updated_at timestamp for a specific profile entry."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT updated_at FROM user_profile WHERE user_id=? AND key=?",
                (user_id, key),
            ).fetchone()
        return row[0] if row else 0.0

    def upsert_profile(self, user_id: str, key: str, value: str,
                       memory_type: str = "preference") -> None:
        """Insert or update a profile entry with memory type."""
        now = time.time()
        with self._lock:
            # Get existing created_at so we don't overwrite it
            existing = self._get_conn().execute(
                "SELECT created_at FROM user_profile WHERE user_id=? AND key=?",
                (user_id, key),
            ).fetchone()
            created_at = existing[0] if existing and existing[0] > 0 else now

            self._get_conn().execute(
                "INSERT INTO user_profile (user_id, key, value, memory_type, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET "
                "value=excluded.value, memory_type=excluded.memory_type, updated_at=excluded.updated_at",
                (user_id, key, value, memory_type, created_at, now),
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
