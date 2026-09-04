"""SQLite schema owned by the independent Runtime.

The schema is deliberately additive.  It lives next to the existing session
database, but Runtime tables have their own names and lifecycle.
"""

from __future__ import annotations

import sqlite3
import time


RUNTIME_SCHEMA_VERSION = 3


def migrate_runtime_schema(conn: sqlite3.Connection) -> None:
    """Apply the Runtime schema idempotently in the caller's transaction."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runtime_schema_meta (
            version INTEGER PRIMARY KEY,
            applied_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS session_entries (
            entry_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            parent_entry_id TEXT,
            entry_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_session_entries_owner_session
            ON session_entries(owner_id, session_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_session_entries_parent
            ON session_entries(parent_entry_id);
        CREATE TABLE IF NOT EXISTS orchestration_runs (
            run_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            graph_thread_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_orchestration_runs_owner_session
            ON orchestration_runs(owner_id, session_id, updated_at);
        CREATE TABLE IF NOT EXISTS orchestration_run_steps (
            run_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            status TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY(run_id, step_id)
        );
        CREATE INDEX IF NOT EXISTS idx_orchestration_run_steps_operation
            ON orchestration_run_steps(operation_id);
        CREATE TABLE IF NOT EXISTS runtime_operations (
            operation_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            orchestration_run_id TEXT NOT NULL DEFAULT '',
            operation_scope TEXT NOT NULL DEFAULT 'top_level'
                CHECK(operation_scope IN ('top_level', 'dag_step', 'nested_agent_tool')),
            step_id TEXT,
            phase TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            version INTEGER NOT NULL,
            state_schema_version INTEGER NOT NULL,
            last_event_sequence INTEGER NOT NULL,
            state_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_operations_owner_session
            ON runtime_operations(owner_id, session_id, updated_at);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_runtime_active_top_level
            ON runtime_operations(owner_id, session_id)
            WHERE operation_scope = 'top_level'
              AND phase NOT IN ('completed', 'failed', 'aborted', 'recovery_required');
        CREATE UNIQUE INDEX IF NOT EXISTS uq_runtime_dag_step
            ON runtime_operations(orchestration_run_id, step_id)
            WHERE operation_scope = 'dag_step'
              AND orchestration_run_id <> '' AND step_id IS NOT NULL;
        CREATE TABLE IF NOT EXISTS runtime_events (
            event_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            payload_json TEXT NOT NULL,
            UNIQUE(operation_id, sequence)
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_events_owner_session
            ON runtime_events(owner_id, session_id, timestamp);
        CREATE TABLE IF NOT EXISTS runtime_approvals (
            approval_id TEXT PRIMARY KEY,
            approval_set_id TEXT NOT NULL DEFAULT '',
            owner_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            tool_call_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            sanitized_arguments_json TEXT NOT NULL,
            risk TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            decision TEXT,
            expires_at REAL,
            version INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_approvals_owner_status
            ON runtime_approvals(owner_id, status, created_at);
        CREATE INDEX IF NOT EXISTS idx_runtime_approvals_operation_status
            ON runtime_approvals(operation_id, status);
        CREATE TABLE IF NOT EXISTS runtime_approval_sets (
            approval_set_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            orchestration_run_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_idempotency_key TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_approval_sets_owner_status
            ON runtime_approval_sets(owner_id, status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_runtime_approval_sets_operation
            ON runtime_approval_sets(operation_id);
        CREATE TABLE IF NOT EXISTS runtime_tool_effects (
            operation_id TEXT NOT NULL,
            tool_call_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            recovery_policy TEXT NOT NULL,
            status TEXT NOT NULL,
            idempotency_key TEXT NOT NULL DEFAULT '',
            result_json TEXT,
            error TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY(operation_id, tool_call_id)
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_effects_recovery
            ON runtime_tool_effects(status, updated_at);
        """
    )
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(runtime_approvals)").fetchall()
    }
    if "approval_set_id" not in columns:
        conn.execute(
            "ALTER TABLE runtime_approvals ADD COLUMN approval_set_id TEXT NOT NULL DEFAULT ''"
        )
    approval_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(runtime_approvals)").fetchall()
    }
    for name, definition in (
        ("decided_by", "TEXT NOT NULL DEFAULT ''"),
        ("decided_at", "REAL"),
        ("decision_source", "TEXT NOT NULL DEFAULT ''"),
        ("idempotency_key", "TEXT NOT NULL DEFAULT ''"),
    ):
        if name not in approval_columns:
            conn.execute(f"ALTER TABLE runtime_approvals ADD COLUMN {name} {definition}")
    approval_set_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(runtime_approval_sets)").fetchall()
    }
    if "last_idempotency_key" not in approval_set_columns:
        conn.execute(
            "ALTER TABLE runtime_approval_sets ADD COLUMN "
            "last_idempotency_key TEXT NOT NULL DEFAULT ''"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runtime_approvals_set_status "
        "ON runtime_approvals(approval_set_id, status)"
    )
    row = conn.execute(
        "SELECT version FROM runtime_schema_meta ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None or int(row[0]) < RUNTIME_SCHEMA_VERSION:
        conn.execute(
            "INSERT OR REPLACE INTO runtime_schema_meta(version, applied_at) VALUES (?, ?)",
            (RUNTIME_SCHEMA_VERSION, time.time()),
        )
