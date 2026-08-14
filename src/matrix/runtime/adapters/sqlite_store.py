"""Transactional SQLite implementation of the Runtime store contract."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..domain.approvals import Approval, ApprovalDecision, ApprovalStatus
from ..domain.errors import OperationConflictError
from ..domain.events import RuntimeEvent, RuntimeEventType
from ..domain.operations import OperationPhase, OperationState, StateTransition
from ..domain.tools import RecoveryPolicy, ToolRequest, ToolResult
from .sqlite_schema import migrate_runtime_schema


class SQLiteRuntimeStore:
    """Keep operation snapshots and events in one SQLite transaction."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("BEGIN")
            migrate_runtime_schema(self._conn)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def create(self, operation: OperationState) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                if operation.operation_scope == "top_level" and self.has_active(operation.owner_id, operation.session_id):
                    raise OperationConflictError(
                        f"session already has an active operation: {operation.session_id}"
                    )
                conn.execute(
                    """INSERT INTO runtime_operations
                    (operation_id, owner_id, session_id, agent_id,
                     orchestration_run_id, operation_scope, step_id, phase,
                     turn_index, version, state_schema_version,
                     last_event_sequence, state_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    _operation_values(operation),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise OperationConflictError(str(exc)) from exc
            except Exception:
                conn.rollback()
                raise

    def load(self, owner_id: str, operation_id: str) -> OperationState | None:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM runtime_operations WHERE operation_id=? AND owner_id=?",
                (operation_id, owner_id),
            ).fetchone()
        return _operation_from_row(row) if row else None

    def commit(self, transition: StateTransition) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                _apply_transition(conn, transition)
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise OperationConflictError(str(exc)) from exc
            except Exception:
                conn.rollback()
                raise

    def list_incomplete(self) -> list[OperationState]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT * FROM runtime_operations WHERE phase NOT IN "
                "('completed','failed','aborted','recovery_required') ORDER BY created_at"
            ).fetchall()
        return [_operation_from_row(row) for row in rows]

    def has_active(self, owner_id: str, session_id: str) -> bool:
        row = self._get_conn().execute(
            """SELECT 1 FROM runtime_operations
               WHERE owner_id=? AND session_id=?
                 AND operation_scope='top_level'
                 AND phase NOT IN ('completed','failed','aborted','recovery_required')
               LIMIT 1""",
            (owner_id, session_id),
        ).fetchone()
        return row is not None

    def find_active(self, owner_id: str, session_id: str) -> OperationState | None:
        with self._lock:
            row = self._get_conn().execute(
                """SELECT * FROM runtime_operations
                   WHERE owner_id=? AND session_id=? AND operation_scope='top_level'
                     AND phase NOT IN ('completed','failed','aborted','recovery_required')
                   ORDER BY updated_at DESC LIMIT 1""",
                (owner_id, session_id),
            ).fetchone()
        return _operation_from_row(row) if row else None

    def events(self, owner_id: str, operation_id: str) -> list[RuntimeEvent]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT * FROM runtime_events WHERE owner_id=? AND operation_id=? ORDER BY sequence",
                (owner_id, operation_id),
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    def create_approval(self, approval: Approval) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO runtime_approvals
                (approval_id, owner_id, operation_id, tool_call_id, tool_name,
                 sanitized_arguments_json, risk, status, decision, expires_at,
                 version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    approval.approval_id, approval.owner_id, approval.operation_id,
                    approval.tool_call_id, approval.tool_name,
                    json.dumps(approval.sanitized_arguments, ensure_ascii=False, default=str),
                    approval.risk, approval.status.value,
                    approval.decision.value if approval.decision else None,
                    approval.expires_at, approval.version, time.time(), time.time(),
                ),
            )
            conn.commit()

    def get_approval(self, owner_id: str, approval_id: str) -> Approval | None:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM runtime_approvals WHERE owner_id=? AND approval_id=?",
                (owner_id, approval_id),
            ).fetchone()
        return _approval_from_row(row) if row else None

    def resolve_approval(
        self,
        owner_id: str,
        approval_id: str,
        decision: ApprovalDecision,
        decided_at: float,
        transition: StateTransition,
    ) -> Approval:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM runtime_approvals WHERE owner_id=? AND approval_id=?",
                    (owner_id, approval_id),
                ).fetchone()
                if row is None:
                    raise KeyError(approval_id)
                existing = _approval_from_row(row)
                if row["operation_id"] != transition.new_state.operation_id:
                    raise OperationConflictError("approval operation does not match transition")
                operation = conn.execute(
                    "SELECT owner_id, phase, version FROM runtime_operations WHERE operation_id=?",
                    (row["operation_id"],),
                ).fetchone()
                if operation is None or operation["owner_id"] != owner_id:
                    raise OperationConflictError("approval operation not found or owner mismatch")
                if operation["phase"] != OperationPhase.WAITING_APPROVAL.value:
                    raise OperationConflictError("operation is not waiting for approval")
                if int(operation["version"]) != transition.previous_version:
                    raise OperationConflictError(
                        f"operation version conflict: expected {transition.previous_version}, "
                        f"actual {operation['version']}"
                    )
                if existing.status is ApprovalStatus.EXPIRED:
                    conn.commit()
                    return existing
                if existing.status is not ApprovalStatus.PENDING:
                    raise OperationConflictError("approval is no longer pending")
                if existing.expires_at is not None and existing.expires_at <= decided_at:
                    cur = conn.execute(
                        "UPDATE runtime_approvals SET status=?, decision=NULL, version=version+1, updated_at=? "
                        "WHERE approval_id=? AND owner_id=? AND status=? AND version=?",
                        (
                            ApprovalStatus.EXPIRED.value,
                            decided_at,
                            approval_id,
                            owner_id,
                            ApprovalStatus.PENDING.value,
                            existing.version,
                        ),
                    )
                    if cur.rowcount != 1:
                        raise OperationConflictError("approval compare-and-swap failed")
                    conn.commit()
                    expired = conn.execute(
                        "SELECT * FROM runtime_approvals WHERE approval_id=?", (approval_id,)
                    ).fetchone()
                    return _approval_from_row(expired)

                _apply_transition(conn, transition)
                status = (
                    ApprovalStatus.APPROVED
                    if decision is ApprovalDecision.APPROVE
                    else ApprovalStatus.SKIPPED
                )
                cur = conn.execute(
                    "UPDATE runtime_approvals SET status=?, decision=?, version=version+1, updated_at=? "
                    "WHERE approval_id=? AND owner_id=? AND status=? AND version=?",
                    (
                        status.value,
                        decision.value,
                        decided_at,
                        approval_id,
                        owner_id,
                        ApprovalStatus.PENDING.value,
                        existing.version,
                    ),
                )
                if cur.rowcount != 1:
                    raise OperationConflictError("approval compare-and-swap failed")
                conn.commit()
                updated = conn.execute(
                    "SELECT * FROM runtime_approvals WHERE approval_id=?", (approval_id,)
                ).fetchone()
                return _approval_from_row(updated)
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise OperationConflictError(str(exc)) from exc
            except Exception:
                conn.rollback()
                raise

    def append_session_entry(
        self, owner_id: str, session_id: str, entry_type: str, payload: dict[str, Any],
    ) -> str:
        """Write the new entry stream without changing the legacy message API."""
        entry_id = f"e_{int(time.time() * 1000000)}_{threading.get_ident()}"
        with self._lock:
            conn = self._get_conn()
            parent = conn.execute(
                "SELECT entry_id FROM session_entries WHERE owner_id=? AND session_id=? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1", (owner_id, session_id)
            ).fetchone()
            conn.execute(
                "INSERT INTO session_entries(entry_id, owner_id, session_id, parent_entry_id, entry_type, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (entry_id, owner_id, session_id, parent[0] if parent else None, entry_type,
                 json.dumps(payload, ensure_ascii=False, default=str), time.time()),
            )
            conn.commit()
        return entry_id

    def begin_tool_effect(self, request: ToolRequest, policy: RecoveryPolicy) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO runtime_tool_effects
                    (operation_id, tool_call_id, owner_id, tool_name, recovery_policy,
                     status, idempotency_key, created_at, updated_at)
                    SELECT ?, ?, owner_id, ?, ?, 'executing', ?, ?, ?
                    FROM runtime_operations WHERE operation_id=?""",
                    (request.operation_id, request.call_id, request.name, policy.value,
                     request.idempotency_key, time.time(), time.time(), request.operation_id),
                )
                if conn.execute("SELECT changes()").fetchone()[0] != 1:
                    raise OperationConflictError("operation does not exist for tool effect")
                conn.commit()
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise OperationConflictError(str(exc)) from exc

    def settle_tool_effect(self, request: ToolRequest, result: ToolResult) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """UPDATE runtime_tool_effects SET status=?, result_json=?, error=?, updated_at=?
                   WHERE operation_id=? AND tool_call_id=? AND status='executing'""",
                (
                    "failed" if result.is_error else "settled",
                    json.dumps(result.result, ensure_ascii=False, default=str), result.error,
                    time.time(), request.operation_id, request.call_id,
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0] != 1:
                raise OperationConflictError("tool effect intent does not exist or is settled")
            conn.commit()

    def mark_recovery_required(self, operation_id: str, owner_id: str = "") -> None:
        """Classify an abandoned operation without replaying external effects."""
        with self._lock:
            conn = self._get_conn()
            params: tuple[Any, ...] = (operation_id,)
            sql = "UPDATE runtime_operations SET phase='recovery_required', version=version+1, updated_at=? WHERE operation_id=?"
            params = (time.time(), operation_id)
            if owner_id:
                sql += " AND owner_id=?"
                params += (owner_id,)
            conn.execute(sql, params)
            conn.commit()


def _operation_values(operation: OperationState) -> tuple[Any, ...]:
    return (
        operation.operation_id, operation.owner_id, operation.session_id, operation.agent_id,
        operation.orchestration_run_id, operation.operation_scope, operation.step_id or None,
        operation.phase.value,
        operation.turn_index, operation.version, operation.state_schema_version,
        operation.last_event_sequence, json.dumps(operation.state, ensure_ascii=False, default=str),
        time.time(), time.time(),
    )


def _apply_transition(conn: sqlite3.Connection, transition: StateTransition) -> None:
    """Apply one operation snapshot and its events inside the caller's transaction."""

    current = conn.execute(
        "SELECT * FROM runtime_operations WHERE operation_id=?",
        (transition.new_state.operation_id,),
    ).fetchone()
    if current is None:
        raise OperationConflictError("operation does not exist")
    if int(current["version"]) != transition.previous_version:
        raise OperationConflictError(
            f"operation version conflict: expected {transition.previous_version}, "
            f"actual {current['version']}"
        )
    if transition.new_state.version != transition.previous_version + 1:
        raise OperationConflictError("operation version must increase by one")
    state = transition.new_state
    cur = conn.execute(
        """UPDATE runtime_operations SET phase=?, turn_index=?, version=?,
        state_schema_version=?, last_event_sequence=?, state_json=?, updated_at=?
        WHERE operation_id=? AND owner_id=? AND version=?""",
        (
            state.phase.value,
            state.turn_index,
            state.version,
            state.state_schema_version,
            state.last_event_sequence,
            json.dumps(state.state, ensure_ascii=False, default=str),
            time.time(),
            state.operation_id,
            state.owner_id,
            transition.previous_version,
        ),
    )
    if cur.rowcount != 1:
        raise OperationConflictError("operation compare-and-swap failed")
    for event in transition.events:
        conn.execute(
            """INSERT INTO runtime_events
            (event_id, owner_id, operation_id, session_id, sequence,
             event_type, timestamp, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.owner_id,
                event.operation_id,
                event.session_id,
                event.sequence,
                event.event_type.value,
                event.timestamp,
                json.dumps(event.payload, ensure_ascii=False, default=str),
            ),
        )


def _operation_from_row(row: sqlite3.Row) -> OperationState:
    return OperationState(
        operation_id=row["operation_id"], owner_id=row["owner_id"],
        session_id=row["session_id"], agent_id=row["agent_id"],
        orchestration_run_id=row["orchestration_run_id"],
        operation_scope=row["operation_scope"], step_id=row["step_id"] or "",
        phase=OperationPhase(row["phase"]), turn_index=int(row["turn_index"]),
        version=int(row["version"]), state_schema_version=int(row["state_schema_version"]),
        last_event_sequence=int(row["last_event_sequence"]),
        state=json.loads(row["state_json"] or "{}"),
    )


def _event_from_row(row: sqlite3.Row) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=row["event_id"], owner_id=row["owner_id"], operation_id=row["operation_id"],
        session_id=row["session_id"], sequence=int(row["sequence"]),
        event_type=RuntimeEventType(row["event_type"]), timestamp=float(row["timestamp"]),
        payload=json.loads(row["payload_json"] or "{}"),
    )


def _approval_from_row(row: sqlite3.Row) -> Approval:
    return Approval(
        approval_id=row["approval_id"], owner_id=row["owner_id"], operation_id=row["operation_id"],
        tool_call_id=row["tool_call_id"], tool_name=row["tool_name"],
        sanitized_arguments=json.loads(row["sanitized_arguments_json"] or "{}"),
        risk=row["risk"], status=ApprovalStatus(row["status"]),
        decision=ApprovalDecision(row["decision"]) if row["decision"] else None,
        expires_at=row["expires_at"], version=int(row["version"]),
    )
