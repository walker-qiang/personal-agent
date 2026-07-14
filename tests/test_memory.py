"""Tests for long-term memory (user profile) and JSON sync."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from matrix.store import SessionStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "test.db"
        s = SessionStore(str(db_path))
        yield s


class TestProfileCRUD:
    def test_upsert_and_get(self, store):
        store.upsert_profile("alice", "语言偏好", "中文")
        store.upsert_profile("alice", "投资风格", "稳健")
        profile = store.get_profile("alice")
        assert profile["语言偏好"] == "中文"
        assert profile["投资风格"] == "稳健"

    def test_user_isolation(self, store):
        store.upsert_profile("alice", "key", "alice_value")
        store.upsert_profile("bob", "key", "bob_value")
        assert store.get_profile("alice")["key"] == "alice_value"
        assert store.get_profile("bob")["key"] == "bob_value"

    def test_upsert_overwrites(self, store):
        store.upsert_profile("alice", "key", "old")
        store.upsert_profile("alice", "key", "new")
        assert store.get_profile("alice")["key"] == "new"

    def test_delete(self, store):
        store.upsert_profile("alice", "key", "value")
        assert store.delete_profile_key("alice", "key") is True
        assert store.get_profile("alice") == {}

    def test_delete_nonexistent(self, store):
        assert store.delete_profile_key("alice", "nonexistent") is False

    def test_empty_profile(self, store):
        assert store.get_profile("nobody") == {}


class TestProfileJSONSync:
    def test_sync_to_file(self, store):
        store.upsert_profile("alice", "语言", "中文")
        store.upsert_profile("alice", "城市", "北京")
        with tempfile.TemporaryDirectory() as d:
            json_path = str(Path(d) / "alice.json")
            result = store.sync_profile_to_file("alice", json_path)
            assert result is True
            with open(json_path) as f:
                data = json.load(f)
            assert data == {"语言": "中文", "城市": "北京"}

    def test_sync_from_file(self, store):
        with tempfile.TemporaryDirectory() as d:
            json_path = str(Path(d) / "alice.json")
            data = {"语言": "中文", "城市": "北京"}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            count = store.sync_profile_from_file("alice", json_path)
            assert count == 2
            assert store.get_profile("alice") == data

    def test_sync_from_missing_file(self, store):
        count = store.sync_profile_from_file("alice", "/nonexistent/path.json")
        assert count == 0

    def test_sync_from_invalid_json(self, store):
        with tempfile.TemporaryDirectory() as d:
            json_path = str(Path(d) / "bad.json")
            with open(json_path, "w") as f:
                f.write("not json")
            count = store.sync_profile_from_file("alice", json_path)
            assert count == 0

    def test_sync_to_file_empty_profile(self, store):
        with tempfile.TemporaryDirectory() as d:
            json_path = str(Path(d) / "nobody.json")
            result = store.sync_profile_to_file("nobody", json_path)
            assert result is False  # Empty profile, no file created

    def test_sync_roundtrip(self, store):
        """Full roundtrip: SQLite → JSON → SQLite (different user)."""
        store.upsert_profile("alice", "key1", "val1")
        store.upsert_profile("alice", "key2", "val2")
        with tempfile.TemporaryDirectory() as d:
            json_path = str(Path(d) / "alice.json")
            assert store.sync_profile_to_file("alice", json_path)
            # Load into bob
            count = store.sync_profile_from_file("bob", json_path)
            assert count == 2
            assert store.get_profile("bob") == store.get_profile("alice")

    def test_sync_from_file_skips_empty(self, store):
        with tempfile.TemporaryDirectory() as d:
            json_path = str(Path(d) / "alice.json")
            data = {"key1": "v1", "": "empty_key", "key2": "  "}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            count = store.sync_profile_from_file("alice", json_path)
            assert count == 1  # Only key1 survives
            assert store.get_profile("alice") == {"key1": "v1"}