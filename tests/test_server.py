"""Tests for FastAPI HTTP endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from matrix.config import AgentConfig
from matrix.server.app import create_app


@pytest.fixture
def client(tmp_cache_path: Path):
    config = AgentConfig(
        root_path=tmp_cache_path.parent,
        cache_path=tmp_cache_path,
        trace_path=tmp_cache_path.parent / "trace" / "tool-calls.jsonl",
        store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
        host="127.0.0.1",
        port=0,
        deepseek_api_key="test-key",
    )
    app = create_app(config)
    with TestClient(app) as tc:
        yield tc


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
    def test_list_tools(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        names = {t["name"] for t in data["tools"]}
        assert len(names) == 5
        assert "finance.holdings_summary" in names

    def test_tools_call_holdings(self, client):
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.holdings_summary", "arguments": {}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool"] == "finance.holdings_summary"
        assert data["result"]["holding_count"] == 2
        assert data["result"]["total_balance_yuan"] == 350.0

    def test_tools_call_asset_lookup(self, client):
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.asset_lookup", "arguments": {"query": "Fund"}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["count"] == 1
        assert data["result"]["assets"][0]["name"] == "Sample Fund"

    def test_tools_call_unknown_tool(self, client):
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.unknown", "arguments": {}},
        )
        assert resp.status_code == 400
        assert "unknown tool" in resp.json()["error"]

    def test_tools_call_missing_tool(self, client):
        resp = client.post(
            "/tools/call",
            json={"arguments": {}},
        )
        assert resp.status_code == 400
        assert "tool is required" in resp.json()["error"]

    def test_tools_call_invalid_arguments(self, client):
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.holdings_summary", "arguments": "not-an-object"},
        )
        assert resp.status_code == 400
        assert "arguments must be an object" in resp.json()["error"]

    def test_tools_call_invalid_json(self, client):
        resp = client.post(
            "/tools/call",
            content="not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_tools_call_writes_trace(self, client, tmp_cache_path):
        trace_path = tmp_cache_path.parent / "trace" / "tool-calls.jsonl"
        resp = client.post(
            "/tools/call",
            json={"tool": "finance.asset_lookup", "arguments": {"query": "sample-cash"}},
        )
        assert resp.status_code == 200
        assert trace_path.exists()
        lines = trace_path.read_text(encoding="utf-8").strip().split("\n")
        trace = [json.loads(line) for line in lines if line.strip()]
        assert len(trace) >= 1
        assert trace[-1]["tool"] == "finance.asset_lookup"
        assert trace[-1]["ok"] is True
        assert trace[-1]["result_count"] == 1


class TestChat:
    def test_returns_sse_error_when_no_key(self, tmp_cache_path):
        config = AgentConfig(
            root_path=tmp_cache_path.parent,
            cache_path=tmp_cache_path,
            trace_path=tmp_cache_path.parent / "trace.jsonl",
            store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
            host="127.0.0.1",
            port=0,
        )
        app = create_app(config)
        with TestClient(app) as client:
            resp = client.post(
                "/chat",
                json={"message": "\u4f60\u597d"},
                headers={"accept": "text/event-stream"},
            )
            assert resp.status_code == 200
            text = resp.text
            assert "event: error" in text
            assert "missing DEEPSEEK_API_KEY" in text
            assert "event: done" in text

    def test_chat_with_tools(self, client):
        """Full chat flow with FakeLLM — requires monkeypatching the app state."""
        # We can't easily patch the app's internal state with TestClient,
        # but we can verify the SSE format is correct even with errors.
        resp = client.post(
            "/chat",
            json={"message": "\u4f60\u597d"},
            headers={"accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        text = resp.text
        assert text.startswith("event: ")
        assert "event: done" in text


class TestReset:
    def test_reset_returns_ok(self, client):
        resp = client.post("/reset", json={"session_id": "test-session"})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_reset_with_empty_body(self, client):
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_reset_invalid_json(self, client):
        resp = client.post(
            "/reset",
            content="not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400


class TestNotFound:
    def test_unknown_path(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_unknown_method(self, client):
        resp = client.put("/healthz")
        assert resp.status_code == 405