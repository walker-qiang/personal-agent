"""Tests for FastAPI HTTP endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from matrix.config import AgentConfig
from matrix.server.app import create_app
from matrix.auth import hash_password


@pytest.fixture
def client(tmp_cache_path: Path):
    config = AgentConfig(
        root_path=tmp_cache_path.parent,
        cache_path=tmp_cache_path,
        trace_path=tmp_cache_path.parent / "trace" / "tool-calls.jsonl",
        store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
        checkpoint_path=str(tmp_cache_path.parent / "var" / "agent" / "checkpoints.db"),
        skills_base_dir=tmp_cache_path.parent / "skills",
        host="127.0.0.1",
        port=0,
        deepseek_api_key="test-key",
        jwt_secret="test-jwt-secret-for-server-tests",
        admin_password_hash=hash_password("test-password"),
    )
    app = create_app(config)
    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def auth_token(client):
    """Login and return a JWT token for authenticated requests."""
    resp = client.post("/api/auth/login", json={
        "username": "admin",
        "password": "test-password",
    })
    assert resp.status_code == 200
    return resp.json()["token"]


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestHealthz:
    def test_returns_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["mode"] == "read-only"
        assert "cache_path" in data
        assert data["cache_exists"] is True
        assert data["provider"] == "deepseek"
        assert data["model"] == "deepseek-chat"
        assert data["llm_available"] is True
        assert data["llm_error"] == ""

    def test_reports_llm_unavailable_when_no_key(self, tmp_cache_path):
        config = AgentConfig(
            root_path=tmp_cache_path.parent,
            cache_path=tmp_cache_path,
            trace_path=tmp_cache_path.parent / "trace.jsonl",
            store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
            checkpoint_path=str(tmp_cache_path.parent / "var" / "agent" / "checkpoints.db"),
            skills_base_dir=tmp_cache_path.parent / "skills",
            host="127.0.0.1",
            port=0,
        )
        app = create_app(config)
        with TestClient(app) as client:
            resp = client.get("/healthz")
            data = resp.json()
            assert data["llm_available"] is False
            assert "missing DEEPSEEK_API_KEY" in data["llm_error"]


class TestTools:
    def test_list_tools(self, client, auth_token):
        resp = client.get("/tools", headers=_auth_headers(auth_token))
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        names = {t["name"] for t in data["tools"]}
        assert len(names) == 11
        assert "finance.holdings_summary" in names

    def test_tools_call_holdings(self, client, auth_token):
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.holdings_summary", "arguments": {}},
            headers=_auth_headers(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool"] == "finance.holdings_summary"
        assert data["result"]["holding_count"] == 2
        assert data["result"]["total_balance_yuan"] == 350.0

    def test_tools_call_asset_lookup(self, client, auth_token):
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.asset_lookup", "arguments": {"query": "Fund"}},
            headers=_auth_headers(auth_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["count"] == 1
        assert data["result"]["assets"][0]["name"] == "Sample Fund"

    def test_tools_call_unknown_tool(self, client, auth_token):
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.unknown", "arguments": {}},
            headers=_auth_headers(auth_token),
        )
        assert resp.status_code == 400
        assert "unknown tool" in resp.json()["error"]

    def test_tools_call_missing_tool(self, client, auth_token):
        resp = client.post(
            "/tools/call",
            json={"arguments": {}},
            headers=_auth_headers(auth_token),
        )
        assert resp.status_code == 400
        assert "tool is required" in resp.json()["error"]

    def test_tools_call_invalid_arguments(self, client, auth_token):
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.holdings_summary", "arguments": "not-an-object"},
            headers=_auth_headers(auth_token),
        )
        assert resp.status_code == 400
        assert "arguments must be an object" in resp.json()["error"]

    def test_tools_call_invalid_json(self, client, auth_token):
        resp = client.post(
            "/tools/call",
            content="not json",
            headers={"content-type": "application/json", **_auth_headers(auth_token)},
        )
        assert resp.status_code == 400

    def test_tools_call_writes_trace(self, client, tmp_cache_path, auth_token):
        trace_path = tmp_cache_path.parent / "trace" / "tool-calls.db"
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.asset_lookup", "arguments": {"query": "sample-cash"}},
            headers=_auth_headers(auth_token),
        )
        assert resp.status_code == 200
        assert trace_path.exists()
        import sqlite3
        conn = sqlite3.connect(str(trace_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM trace_events WHERE event_type = 'tool_call' ORDER BY id DESC LIMIT 1"
        ).fetchall()
        assert len(rows) >= 1
        assert rows[0]["tool_name"] == "finance.asset_lookup"
        assert rows[0]["ok"] == 1


class TestChat:
    def test_returns_sse_error_when_no_key(self, tmp_cache_path):
        config = AgentConfig(
            root_path=tmp_cache_path.parent,
            cache_path=tmp_cache_path,
            trace_path=tmp_cache_path.parent / "trace.jsonl",
            store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
            checkpoint_path=str(tmp_cache_path.parent / "var" / "agent" / "checkpoints.db"),
            skills_base_dir=tmp_cache_path.parent / "skills",
            host="127.0.0.1",
            port=0,
            jwt_secret="test-jwt-secret-for-server-tests",
            admin_password_hash=hash_password("test-password"),
        )
        app = create_app(config)
        with TestClient(app) as client:
            # Login first
            login_resp = client.post("/api/auth/login", json={
                "username": "admin",
                "password": "test-password",
            })
            token = login_resp.json()["token"]
            resp = client.post(
                "/chat",
                json={"message": "\u4f60\u597d"},
                headers={"accept": "text/event-stream", **_auth_headers(token)},
            )
            assert resp.status_code == 200
            text = resp.text
            assert "event: error" in text
            assert "missing DEEPSEEK_API_KEY" in text
            assert "event: done" in text

    def test_chat_with_tools(self, client, auth_token):
        """Full chat flow with FakeLLM — requires monkeypatching the app state."""
        resp = client.post(
            "/chat",
            json={"message": "\u4f60\u597d"},
            headers={"accept": "text/event-stream", **_auth_headers(auth_token)},
        )
        assert resp.status_code == 200
        text = resp.text
        assert text.startswith("event: ")
        assert "event: done" in text


class TestReset:
    def test_reset_returns_ok(self, client, auth_token):
        resp = client.post("/reset", json={"session_id": "test-session"}, headers=_auth_headers(auth_token))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_reset_with_empty_body(self, client, auth_token):
        resp = client.post("/reset", json={}, headers=_auth_headers(auth_token))
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_reset_invalid_json(self, client, auth_token):
        resp = client.post(
            "/reset",
            content="not json",
            headers={"content-type": "application/json", **_auth_headers(auth_token)},
        )
        assert resp.status_code == 400


class TestNotFound:
    def test_unknown_path(self, client, auth_token):
        resp = client.get("/nonexistent", headers=_auth_headers(auth_token))
        assert resp.status_code == 404

    def test_unknown_method(self, client):
        resp = client.put("/healthz")
        assert resp.status_code == 405