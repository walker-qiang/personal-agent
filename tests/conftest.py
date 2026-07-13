"""Shared test fixtures and utilities."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import pytest

from matrix.config import AgentConfig


def write_cache_fixture(path: Path) -> None:
    """Create a minimal SQLite finance cache for testing."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE assets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fact_id TEXT NOT NULL UNIQUE,
  code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  asset_type TEXT NOT NULL,
  legacy_asset_type TEXT NOT NULL,
  bucket TEXT NOT NULL,
  channel TEXT NOT NULL DEFAULT '',
  currency TEXT NOT NULL,
  risk_level TEXT,
  holding_cost_pct REAL,
  expected_yield_pct REAL,
  notes TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  status TEXT NOT NULL,
  archived_at TEXT
);

CREATE TABLE snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fact_id TEXT NOT NULL UNIQUE,
  asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE RESTRICT,
  asset_fact_id TEXT NOT NULL,
  snapshot_date TEXT NOT NULL,
  balance_cents INTEGER NOT NULL,
  expected_yield_pct REAL,
  actual_yield_pct REAL,
  notes TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  source_ref TEXT,
  correction_of TEXT,
  UNIQUE(asset_id, snapshot_date)
);

CREATE TABLE bucket_targets (
  bucket TEXT PRIMARY KEY,
  target_pct REAL NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL
);
"""
    )
    conn.execute(
        """INSERT INTO assets (
  fact_id, code, name, asset_type, legacy_asset_type, bucket, channel, currency,
  risk_level, holding_cost_pct, expected_yield_pct, notes, created_at, updated_at,
  status, archived_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "ast_sample_cash", "sample-cash", "Sample Cash",
            "cash", "cash-cny", "cash", "Bank", "CNY",
            "R1", None, 1.5, "",
            "2026-05-01T00:00:00Z", "2026-05-01T00:00:00Z",
            "active", None,
        ),
    )
    conn.execute(
        """INSERT INTO assets (
  fact_id, code, name, asset_type, legacy_asset_type, bucket, channel, currency,
  risk_level, holding_cost_pct, expected_yield_pct, notes, created_at, updated_at,
  status, archived_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "ast_sample_fund", "sample-fund", "Sample Fund",
            "fund", "etf-fund", "growth", "Broker", "CNY",
            "R3", None, None, "",
            "2026-05-01T00:00:00Z", "2026-05-01T00:00:00Z",
            "active", None,
        ),
    )
    cash_id = conn.execute("SELECT id FROM assets WHERE fact_id = ?", ("ast_sample_cash",)).fetchone()[0]
    fund_id = conn.execute("SELECT id FROM assets WHERE fact_id = ?", ("ast_sample_fund",)).fetchone()[0]
    snapshots = [
        ("snap_cash_1", cash_id, "ast_sample_cash", "2026-05-01", 10000, 1.5, None, ""),
        ("snap_cash_2", cash_id, "ast_sample_cash", "2026-05-02", 15000, 1.5, None, ""),
        ("snap_fund_1", fund_id, "ast_sample_fund", "2026-05-01", 20000, None, None, ""),
    ]
    for row in snapshots:
        conn.execute(
            """INSERT INTO snapshots (
  fact_id, asset_id, asset_fact_id, snapshot_date, balance_cents,
  expected_yield_pct, actual_yield_pct, notes, created_at, source_ref, correction_of
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (*row, "2026-05-01T00:00:00Z", None, None),
        )
    conn.execute(
        "INSERT INTO bucket_targets (bucket, target_pct, notes, updated_at) VALUES (?, ?, ?, ?)",
        ("cash", 40.0, "", "2026-05-01T00:00:00Z"),
    )
    conn.execute(
        "INSERT INTO bucket_targets (bucket, target_pct, notes, updated_at) VALUES (?, ?, ?, ?)",
        ("growth", 60.0, "", "2026-05-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def tmp_cache_path() -> Any:
    """Create a temporary SQLite finance cache for testing."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "finance.sqlite"
        write_cache_fixture(cache_path)
        yield cache_path


@pytest.fixture
def tmp_dir() -> Any:
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def agent_config(tmp_cache_path: Path) -> AgentConfig:
    """Create an AgentConfig with test values."""
    return AgentConfig(
        root_path=tmp_cache_path.parent,
        cache_path=tmp_cache_path,
        trace_path=tmp_cache_path.parent / "trace" / "tool-calls.jsonl",
        store_path=tmp_cache_path.parent / "var" / "agent" / "sessions.db",
        checkpoint_path=str(tmp_cache_path.parent / "var" / "agent" / "checkpoints.db"),
        skills_dir=tmp_cache_path.parent / "skills" / "investment",
        skills_base_dir=tmp_cache_path.parent / "skills",
        host="127.0.0.1",
        port=0,
        deepseek_api_key="test-key",
    )


@pytest.fixture
async def async_client(agent_config: AgentConfig):
    """Create an async FastAPI test client with auth configured."""
    from httpx import ASGITransport, AsyncClient
    from matrix.server.app import create_app
    from matrix.auth import hash_password

    # Override config to enable auth
    config = agent_config
    object.__setattr__(config, "admin_password_hash", hash_password("test-password"))
    object.__setattr__(config, "jwt_secret", "test-jwt-secret-for-auth-tests")

    app = create_app(config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client