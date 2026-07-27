"""Tests for SessionStore session tree features (Phase 7).

Tests message_id / parent_id chain, get_history tree traversal,
branch, get_leaf_id, get_branches, and schema migration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from matrix.store import SessionStore


class TestSessionTree:
    """Tests for session tree structure (message_id, parent_id, leaf_id)."""

    @pytest.fixture
    def store(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = Path(f.name)
        s = SessionStore(str(db_path))
        yield s
        s.close()
        db_path.unlink(missing_ok=True)

    def test_save_message_returns_message_id(self, store):
        msg_id = store.save_message("s1", "user", "hello")
        assert msg_id, "save_message should return a message_id"
        assert len(msg_id) == 12, f"message_id should be 12 hex chars, got {msg_id}"

    def test_save_message_sets_parent_id(self, store):
        """Each new message's parent_id should be the previous leaf_id."""
        msg1 = store.save_message("s1", "user", "first")
        msg2 = store.save_message("s1", "assistant", "second")
        msg3 = store.save_message("s1", "user", "third")

        leaf = store.get_leaf_id("s1")
        assert leaf == msg3, "leaf_id should be the latest message"

        history = store.get_history("s1", max_turns=5)
        assert len(history) == 3
        assert history[0]["content"] == "first"
        assert history[1]["content"] == "second"
        assert history[2]["content"] == "third"

    def test_get_history_traverses_tree(self, store):
        """get_history should traverse from leaf to root via parent_id."""
        for i in range(10):
            store.save_message("s1", "user" if i % 2 == 0 else "assistant", f"msg{i}")

        history = store.get_history("s1", max_turns=3)
        # Should return at most 3 turns = 6 messages
        assert len(history) == 6
        # Should be chronological (oldest first)
        assert history[0]["content"] == "msg4"  # 10 - 6 = 4
        assert history[-1]["content"] == "msg9"

    def test_get_history_respects_max_turns(self, store):
        for i in range(20):
            store.save_message("s1", "user", f"msg{i}")

        history = store.get_history("s1", max_turns=2)
        assert len(history) == 4  # 2 turns = 4 messages
        assert history[0]["content"] == "msg16"
        assert history[-1]["content"] == "msg19"

    def test_get_history_empty_session(self, store):
        history = store.get_history("nonexistent")
        assert history == []

    def test_get_history_fallback_no_leaf_id(self, store):
        """When a session exists but has no leaf_id, fall back to linear query."""
        # Directly insert a session with empty leaf_id
        import time
        with store._lock:
            conn = store._get_conn()
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, msg_count, leaf_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("s_noleaf", "", time.time(), time.time(), 2, ""),
            )
            conn.execute(
                "INSERT INTO messages (message_id, parent_id, session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("m1", None, "s_noleaf", "user", "fallback_msg", time.time()),
            )
            conn.commit()

        history = store.get_history("s_noleaf")
        assert len(history) == 1
        assert history[0]["content"] == "fallback_msg"


class TestBranch:
    """Tests for branch / fork functionality."""

    @pytest.fixture
    def store(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = Path(f.name)
        s = SessionStore(str(db_path))
        yield s
        s.close()
        db_path.unlink(missing_ok=True)

    def test_branch_moves_leaf_id(self, store):
        msg1 = store.save_message("s1", "user", "original")
        msg2 = store.save_message("s1", "assistant", "reply")
        msg3 = store.save_message("s1", "user", "follow_up")

        assert store.get_leaf_id("s1") == msg3

        # Branch back to msg1
        ok = store.branch("s1", msg1)
        assert ok is True
        assert store.get_leaf_id("s1") == msg1

        # New message should be child of msg1
        msg4 = store.save_message("s1", "user", "new_branch")
        assert store.get_leaf_id("s1") == msg4

        # History should show current branch: msg1 → msg4
        # (msg2, msg3 are on the old branch, not in current traversal)
        history = store.get_history("s1", max_turns=5)
        assert len(history) == 2
        assert history[0]["content"] == "original"
        assert history[1]["content"] == "new_branch"

    def test_branch_nonexistent_message(self, store):
        store.save_message("s1", "user", "hello")
        ok = store.branch("s1", "nonexistent")
        assert ok is False

    def test_branch_nonexistent_session(self, store):
        ok = store.branch("nonexistent", "any_id")
        assert ok is False

    def test_branch_preserves_old_branch(self, store):
        """Branching should preserve old messages (append-only).
        Old branch messages still exist in DB but not on current traversal path."""
        msg1 = store.save_message("s1", "user", "root")
        msg2 = store.save_message("s1", "assistant", "branch_a")
        msg3 = store.save_message("s1", "user", "branch_a_cont")

        # Branch back to root
        store.branch("s1", msg1)
        store.save_message("s1", "assistant", "branch_b")

        # Current branch history
        history = store.get_history("s1", max_turns=10)
        contents = [m["content"] for m in history]
        assert "root" in contents
        assert "branch_b" in contents

        # Old branch messages still exist in DB
        with store._lock:
            conn = store._get_conn()
            rows = conn.execute(
                "SELECT content FROM messages WHERE session_id=? ORDER BY id",
                ("s1",),
            ).fetchall()
        all_contents = [r[0] for r in rows]
        assert "branch_a" in all_contents
        assert "branch_a_cont" in all_contents

    def test_multiple_branches(self, store):
        """Test branching multiple times."""
        msg1 = store.save_message("s1", "user", "root")
        msg2 = store.save_message("s1", "assistant", "a1")
        store.branch("s1", msg1)
        store.save_message("s1", "assistant", "b1")
        store.branch("s1", msg1)
        store.save_message("s1", "assistant", "c1")

        # Now msg1 should have 3 children (a1, b1, c1)
        branches = store.get_branches("s1")
        assert len(branches) == 1
        assert branches[0]["fork_point"] == msg1
        assert branches[0]["branch_count"] == 3


class TestGetBranches:
    """Tests for get_branches fork point detection."""

    @pytest.fixture
    def store(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = Path(f.name)
        s = SessionStore(str(db_path))
        yield s
        s.close()
        db_path.unlink(missing_ok=True)

    def test_no_branches_for_linear_session(self, store):
        for i in range(5):
            store.save_message("s1", "user", f"msg{i}")
        branches = store.get_branches("s1")
        assert branches == []

    def test_no_branches_for_empty_session(self, store):
        branches = store.get_branches("nonexistent")
        assert branches == []

    def test_detects_fork_point(self, store):
        root = store.save_message("s1", "user", "root")
        store.save_message("s1", "assistant", "branch_a")
        store.save_message("s1", "assistant", "branch_a_cont")

        store.branch("s1", root)
        store.save_message("s1", "assistant", "branch_b")

        branches = store.get_branches("s1")
        assert len(branches) == 1
        assert branches[0]["fork_point"] == root
        assert branches[0]["branch_count"] == 2

    def test_multiple_fork_points(self, store):
        """Create a more complex tree with multiple fork points."""
        msg1 = store.save_message("s1", "user", "root")
        msg2 = store.save_message("s1", "assistant", "child1")
        msg3 = store.save_message("s1", "user", "child2")

        # Fork at msg2
        store.branch("s1", msg2)
        store.save_message("s1", "assistant", "fork_at_msg2")

        # Fork at msg1
        store.branch("s1", msg1)
        store.save_message("s1", "assistant", "fork_at_root")

        branches = store.get_branches("s1")
        assert len(branches) == 2  # fork at msg1 and msg2
        fork_points = [b["fork_point"] for b in branches]
        assert msg1 in fork_points
        assert msg2 in fork_points


class TestGetLeafId:
    """Tests for get_leaf_id."""

    @pytest.fixture
    def store(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = Path(f.name)
        s = SessionStore(str(db_path))
        yield s
        s.close()
        db_path.unlink(missing_ok=True)

    def test_leaf_id_updates_on_new_message(self, store):
        msg1 = store.save_message("s1", "user", "first")
        assert store.get_leaf_id("s1") == msg1

        msg2 = store.save_message("s1", "assistant", "second")
        assert store.get_leaf_id("s1") == msg2

    def test_leaf_id_none_for_nonexistent_session(self, store):
        assert store.get_leaf_id("nonexistent") is None

    def test_leaf_id_updates_after_branch(self, store):
        msg1 = store.save_message("s1", "user", "root")
        store.save_message("s1", "assistant", "child")

        store.branch("s1", msg1)
        assert store.get_leaf_id("s1") == msg1


class TestSchemaMigration:
    """Tests that schema migration handles existing databases correctly."""

    @pytest.fixture
    def store(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            db_path = Path(f.name)
        s = SessionStore(str(db_path))
        yield s
        s.close()
        db_path.unlink(missing_ok=True)

    def test_migration_adds_columns(self, store):
        """After migration, message_id, parent_id, leaf_id should exist."""
        with store._lock:
            conn = store._get_conn()
            msg_cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
            sess_cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]

        assert "message_id" in msg_cols, "message_id column should exist after migration"
        assert "parent_id" in msg_cols, "parent_id column should exist after migration"
        assert "leaf_id" in sess_cols, "leaf_id column should exist after migration"

    def test_migration_backfills_message_id(self, store):
        """Messages inserted before migration should get backfilled message_id."""
        import time
        with store._lock:
            conn = store._get_conn()
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, msg_count) "
                "VALUES (?, ?, ?, ?, ?)",
                ("s_backfill", "", time.time(), time.time(), 1),
            )
            conn.commit()

            # Simulate a message without message_id (old schema)
            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                ("s_backfill", "user", "old message", time.time()),
            )
            conn.commit()

        # Re-open to trigger migration
        store.close()
        s2 = SessionStore(str(store._db_path))

        with s2._lock:
            conn = s2._get_conn()
            row = conn.execute(
                "SELECT message_id FROM messages WHERE session_id=? AND role=?",
                ("s_backfill", "user"),
            ).fetchone()
        assert row is not None
        assert row[0] != "", "message_id should be backfilled"
        assert row[0].startswith("m"), f"backfilled message_id should start with 'm', got {row[0]}"

        s2.close()

    def test_migration_backfills_leaf_id(self, store):
        """Sessions should get leaf_id backfilled to last message."""
        import time
        with store._lock:
            conn = store._get_conn()
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at, msg_count) "
                "VALUES (?, ?, ?, ?, ?)",
                ("s_leaf", "", time.time(), time.time(), 1),
            )
            conn.commit()

        # Insert a message with message_id
        msg_id = store.save_message("s_leaf", "user", "test leaf")

        # Re-open to trigger migration
        store.close()
        s2 = SessionStore(str(store._db_path))

        with s2._lock:
            conn = s2._get_conn()
            row = conn.execute(
                "SELECT leaf_id FROM sessions WHERE id=?", ("s_leaf",),
            ).fetchone()
        assert row is not None
        assert row[0] == msg_id, f"leaf_id should be {msg_id}, got {row[0]}"

        s2.close()

    def test_migration_is_idempotent(self, store):
        """Running migration twice should not cause errors."""
        store.save_message("s1", "user", "test")
        # Re-open to trigger migration again
        store.close()
        s2 = SessionStore(str(store._db_path))
        s2.save_message("s1", "assistant", "reply")
        history = s2.get_history("s1")
        assert len(history) == 2
        s2.close()