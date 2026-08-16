"""SQLite schema owned by the independent Runtime.

The schema is deliberately additive.  It lives next to the existing session
database, but Runtime tables have their own names and lifecycle.
"""

from __future__ import annotations

import sqlite3
import time


RUNTIME_SCHEMA_VERSION = 2


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
    row = conn.execute(
        "SELECT version FROM runtime_schema_meta ORDER BY version DESC LIMIT 1"
    ).fetchone()
    current_version = int(row[0]) if row is not None else 0
    if not _supports_nested_agent_scope(conn):
        _migrate_runtime_operations_for_nested_agents(conn)
    if current_version < RUNTIME_SCHEMA_VERSION:
        conn.execute(
            "INSERT OR REPLACE INTO runtime_schema_meta(version, applied_at) VALUES (?, ?)",
            (RUNTIME_SCHEMA_VERSION, time.time()),
        )


def _supports_nested_agent_scope(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='runtime_operations'"
    ).fetchone()
    return bool(row and row[0] and "nested_agent_tool" in str(row[0]))


def _migrate_runtime_operations_for_nested_agents(conn: sqlite3.Connection) -> None:
    """Rebuild the table to extend its CHECK constraint without losing data.

    SQLite cannot alter a CHECK constraint in place.  Existing installations
    may already have the v1 table, so keep the migration explicit and
    copy every persisted operation before replacing the table.
    """
    conn.execute("DROP INDEX IF EXISTS idx_runtime_operations_owner_session")
    conn.execute("DROP INDEX IF EXISTS uq_runtime_active_top_level")
    conn.execute("DROP INDEX IF EXISTS uq_runtime_dag_step")
    conn.execute("ALTER TABLE runtime_operations RENAME TO runtime_operations_v1")
    conn.execute(
        """
        CREATE TABLE runtime_operations (
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
        )
        """
    )
    conn.execute(
        """
        INSERT INTO runtime_operations (
            operation_id, owner_id, session_id, agent_id,
            orchestration_run_id, operation_scope, step_id, phase,
            turn_index, version, state_schema_version,
            last_event_sequence, state_json, created_at, updated_at
        )
        SELECT
            operation_id, owner_id, session_id, agent_id,
            orchestration_run_id, operation_scope, step_id, phase,
            turn_index, version, state_schema_version,
            last_event_sequence, state_json, created_at, updated_at
        FROM runtime_operations_v1
        """
    )
    conn.execute("DROP TABLE runtime_operations_v1")
    conn.execute(
        """
        CREATE INDEX idx_runtime_operations_owner_session
            ON runtime_operations(owner_id, session_id, updated_at)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX uq_runtime_active_top_level
            ON runtime_operations(owner_id, session_id)
            WHERE operation_scope = 'top_level'
              AND phase NOT IN ('completed', 'failed', 'aborted', 'recovery_required')
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX uq_runtime_dag_step
            ON runtime_operations(orchestration_run_id, step_id)
            WHERE operation_scope = 'dag_step'
              AND orchestration_run_id <> '' AND step_id IS NOT NULL
        """
    )
