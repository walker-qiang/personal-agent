"""Transactional SQLite implementation of the Runtime store contract."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..domain.approvals import (
    Approval,
    ApprovalDecision,
    ApprovalSet,
    ApprovalSetStatus,
    ApprovalStatus,
)
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

    def recover_incomplete(self, reason: str = "process_restart") -> list[OperationState]:
        """Mark non-approval operations left mid-flight as recovery-required.

        The update and its audit event are committed together. Approval waits
        remain resumable; replaying any other phase could duplicate a tool
        effect after a crash between the effect and its settlement.
        """
        recovered_ids: list[tuple[str, str]] = []
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    "SELECT * FROM runtime_operations "
                    "WHERE phase NOT IN ('completed','failed','aborted','recovery_required','waiting_approval') "
                    "ORDER BY created_at"
                ).fetchall()
                for row in rows:
                    previous_phase = str(row["phase"])
                    state = json.loads(row["state_json"] or "{}")
                    if not isinstance(state, dict):
                        state = {}
                    state["runtime_recovery"] = {
                        "reason": reason,
                        "previous_phase": previous_phase,
                        "recovered_at": now,
                    }
                    sequence = int(row["last_event_sequence"]) + 1
                    event_id = uuid.uuid4().hex
                    updated = conn.execute(
                        "UPDATE runtime_operations SET phase='recovery_required', version=version+1, "
                        "last_event_sequence=?, state_json=?, updated_at=? "
                        "WHERE operation_id=? AND version=? AND phase=?",
                        (
                            sequence, json.dumps(state, ensure_ascii=False, default=str), now,
                            row["operation_id"], row["version"], previous_phase,
                        ),
                    )
                    if updated.rowcount != 1:
                        raise OperationConflictError("runtime recovery compare-and-swap failed")
                    conn.execute(
                        "INSERT INTO runtime_events "
                        "(event_id, owner_id, operation_id, session_id, sequence, event_type, timestamp, payload_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            event_id, row["owner_id"], row["operation_id"], row["session_id"],
                            sequence, RuntimeEventType.RECOVERY_REQUIRED.value, now,
                            json.dumps({"reason": reason, "previous_phase": previous_phase}, ensure_ascii=False),
                        ),
                    )
                    recovered_ids.append((row["owner_id"], row["operation_id"]))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return [
            operation
            for owner_id, operation_id in recovered_ids
            if (operation := self.load(owner_id, operation_id)) is not None
        ]

    def list_operations(
        self,
        owner_id: str,
        session_id: str | None = None,
        phase: str | None = None,
        limit: int = 50,
    ) -> list[OperationState]:
        clauses = ["owner_id=?"]
        params: list[Any] = [owner_id]
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        if phase:
            clauses.append("phase=?")
            params.append(phase)
        params.append(max(1, min(limit, 200)))
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT * FROM runtime_operations WHERE "
                + " AND ".join(clauses)
                + " ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                params,
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

    def find_waiting_approval(self, owner_id: str, session_id: str) -> OperationState | None:
        """Find any durable approval wait, including DAG step operations."""
        values = self.list_waiting_approvals(owner_id, session_id, limit=1)
        return values[0] if values else None

    def list_waiting_approvals(
        self, owner_id: str, session_id: str, limit: int = 50,
    ) -> list[OperationState]:
        """List every durable approval wait for one session, including DAG steps."""
        with self._lock:
            rows = self._get_conn().execute(
                """SELECT o.* FROM runtime_operations o
                   JOIN runtime_approvals a ON a.operation_id=o.operation_id
                   WHERE o.owner_id=? AND o.session_id=?
                     AND o.phase=?
                     AND a.status=?
                   GROUP BY o.operation_id
                   ORDER BY o.updated_at DESC, o.created_at DESC
                   LIMIT ?""",
                (
                    owner_id, session_id, OperationPhase.WAITING_APPROVAL.value,
                    ApprovalStatus.PENDING.value,
                    max(1, min(limit, 200)),
                ),
            ).fetchall()
        seen: set[str] = set()
        values: list[OperationState] = []
        for row in rows:
            operation_id = str(row["operation_id"])
            if operation_id not in seen:
                seen.add(operation_id)
                values.append(_operation_from_row(row))
        return values

    def ensure_orchestration_run(
        self,
        run_id: str,
        owner_id: str,
        session_id: str,
        graph_thread_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not run_id:
            return
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO orchestration_runs
                   (run_id, owner_id, session_id, graph_thread_id, status,
                    metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'running', ?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     graph_thread_id=excluded.graph_thread_id,
                     metadata_json=excluded.metadata_json,
                     updated_at=excluded.updated_at""",
                (
                    run_id, owner_id, session_id, graph_thread_id,
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                    now, now,
                ),
            )
            conn.commit()

    def upsert_orchestration_step(
        self,
        run_id: str,
        step_id: str,
        operation_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not run_id or not step_id:
            return
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO orchestration_run_steps
                   (run_id, step_id, operation_id, status, version,
                    metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?)
                   ON CONFLICT(run_id, step_id) DO UPDATE SET
                     operation_id=excluded.operation_id,
                     status=excluded.status,
                     version=orchestration_run_steps.version+1,
                     metadata_json=excluded.metadata_json,
                     updated_at=excluded.updated_at""",
                (
                    run_id, step_id, operation_id, status,
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                    now, now,
                ),
            )
            conn.commit()

    def events(self, owner_id: str, operation_id: str) -> list[RuntimeEvent]:
        return self.list_events(owner_id, operation_id)

    def list_events(
        self, owner_id: str, operation_id: str, limit: int = 200,
    ) -> list[RuntimeEvent]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT * FROM runtime_events WHERE owner_id=? AND operation_id=? "
                "ORDER BY sequence LIMIT ?",
                (owner_id, operation_id, max(1, min(limit, 1000))),
            ).fetchall()
        return [_event_from_row(row) for row in rows]

    def create_approval(self, approval: Approval) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO runtime_approvals
                (approval_id, approval_set_id, owner_id, operation_id, tool_call_id, tool_name,
                 sanitized_arguments_json, risk, status, decision, expires_at,
                 version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    approval.approval_id, approval.approval_set_id, approval.owner_id,
                    approval.operation_id,
                    approval.tool_call_id, approval.tool_name,
                    json.dumps(approval.sanitized_arguments, ensure_ascii=False, default=str),
                    approval.risk, approval.status.value,
                    approval.decision.value if approval.decision else None,
                    approval.expires_at, approval.version, time.time(), time.time(),
                ),
            )
            conn.commit()

    def create_approval_set(self, approval_set: ApprovalSet) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT INTO runtime_approval_sets
                (approval_set_id, owner_id, operation_id, orchestration_run_id,
                 status, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    approval_set.approval_set_id,
                    approval_set.owner_id,
                    approval_set.operation_id,
                    approval_set.orchestration_run_id,
                    approval_set.status.value,
                    approval_set.version,
                    approval_set.created_at or time.time(),
                    approval_set.updated_at or time.time(),
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

    def get_approval_set(self, owner_id: str, approval_set_id: str) -> ApprovalSet | None:
        with self._lock:
            row = self._get_conn().execute(
                "SELECT * FROM runtime_approval_sets "
                "WHERE owner_id=? AND approval_set_id=?",
                (owner_id, approval_set_id),
            ).fetchone()
            if row is not None:
                return _approval_set_from_row(row)
            # Compatibility for approvals written before approval sets existed.
            approval = self._get_conn().execute(
                "SELECT * FROM runtime_approvals "
                "WHERE owner_id=? AND approval_id=?",
                (owner_id, approval_set_id),
            ).fetchone()
        if approval is None:
            return None
        item = _approval_from_row(approval)
        return ApprovalSet(
            approval_set_id=item.approval_id,
            owner_id=item.owner_id,
            operation_id=item.operation_id,
            status=_approval_set_status((item,)),
            version=item.version,
            created_at=item.created_at,
            updated_at=item.updated_at,
        )

    def list_approval_set(
        self, owner_id: str, approval_set_id: str,
    ) -> list[Approval]:
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT * FROM runtime_approvals "
                "WHERE owner_id=? AND (approval_set_id=? OR "
                "(approval_set_id='' AND approval_id=?)) "
                "ORDER BY created_at, approval_id",
                (owner_id, approval_set_id, approval_set_id),
            ).fetchall()
        return [_approval_from_row(row) for row in rows]

    def delete_session_runtime(self, owner_id: str, session_id: str) -> None:
        """Delete session-scoped Runtime data in one SQLite transaction."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("BEGIN IMMEDIATE")
                operation_rows = conn.execute(
                    "SELECT operation_id FROM runtime_operations "
                    "WHERE owner_id=? AND session_id=?",
                    (owner_id, session_id),
                ).fetchall()
                operation_ids = [row["operation_id"] for row in operation_rows]
                operation_run_rows = conn.execute(
                    "SELECT orchestration_run_id FROM runtime_operations "
                    "WHERE owner_id=? AND session_id=? AND orchestration_run_id<>''",
                    (owner_id, session_id),
                ).fetchall()
                run_ids = {
                    row["orchestration_run_id"] for row in operation_run_rows
                }
                run_rows = conn.execute(
                    "SELECT run_id FROM orchestration_runs "
                    "WHERE owner_id=? AND session_id=?",
                    (owner_id, session_id),
                ).fetchall()
                run_ids.update(row["run_id"] for row in run_rows)

                if operation_ids:
                    placeholders = ",".join("?" for _ in operation_ids)
                    conn.execute(
                        f"DELETE FROM runtime_tool_effects WHERE operation_id IN ({placeholders})",
                        operation_ids,
                    )
                    conn.execute(
                        f"DELETE FROM runtime_events WHERE operation_id IN ({placeholders})",
                        operation_ids,
                    )
                    conn.execute(
                        f"DELETE FROM runtime_approvals WHERE operation_id IN ({placeholders})",
                        operation_ids,
                    )
                    conn.execute(
                        f"DELETE FROM runtime_approval_sets WHERE operation_id IN ({placeholders})",
                        operation_ids,
                    )
                if run_ids:
                    placeholders = ",".join("?" for _ in run_ids)
                    conn.execute(
                        f"DELETE FROM orchestration_run_steps WHERE run_id IN ({placeholders})",
                        tuple(run_ids),
                    )
                conn.execute(
                    "DELETE FROM runtime_operations WHERE owner_id=? AND session_id=?",
                    (owner_id, session_id),
                )
                conn.execute(
                    "DELETE FROM orchestration_runs WHERE owner_id=? AND session_id=?",
                    (owner_id, session_id),
                )
                conn.execute(
                    "DELETE FROM session_entries WHERE owner_id=? AND session_id=?",
                    (owner_id, session_id),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def list_approvals(
        self,
        owner_id: str,
        session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Approval]:
        clauses = ["a.owner_id=?"]
        params: list[Any] = [owner_id]
        if session_id:
            clauses.append("o.session_id=?")
            params.append(session_id)
        if status:
            clauses.append("a.status=?")
            params.append(status)
        params.append(max(1, min(limit, 200)))
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT a.* FROM runtime_approvals a "
                "JOIN runtime_operations o ON o.operation_id=a.operation_id "
                "WHERE " + " AND ".join(clauses)
                + " ORDER BY a.updated_at DESC, a.created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_approval_from_row(row) for row in rows]

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
                if (
                    existing.status is not ApprovalStatus.PENDING
                    and existing.decision is decision
                ):
                    conn.commit()
                    return existing
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
                    if existing.approval_set_id:
                        _refresh_approval_set(conn, existing.approval_set_id, decided_at)
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
                if existing.approval_set_id:
                    _refresh_approval_set(conn, existing.approval_set_id, decided_at)
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

    def ensure_branch_summary_entry(
        self,
        owner_id: str,
        session_id: str,
        from_message_id: str,
        abandoned_leaf_id: str,
        message_count: int,
    ) -> tuple[str, dict[str, Any]]:
        """Create or find the durable entry for one abandoned branch.

        The branch identity is stable, so repeated requests for the same
        fork reuse one entry instead of scheduling duplicate summaries.
        """
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT entry_id, payload_json FROM session_entries "
                "WHERE owner_id=? AND session_id=? AND entry_type='branch_summary'",
                (owner_id, session_id),
            ).fetchall()
            for row in rows:
                try:
                    existing = json.loads(row[1] or "{}")
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(existing, dict)
                    and existing.get("from_message_id") == from_message_id
                    and existing.get("abandoned_leaf_id") == abandoned_leaf_id
                ):
                    return row[0], existing

            identity = "\x1f".join((owner_id, session_id, from_message_id, abandoned_leaf_id))
            entry_id = "bs_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
            payload = {
                "from_message_id": from_message_id,
                "abandoned_leaf_id": abandoned_leaf_id,
                "message_count": message_count,
                "status": "scheduled",
                "summary": "",
                "key_points": [],
                "unresolved": "",
                "attempts": 0,
            }
            parent = conn.execute(
                "SELECT entry_id FROM session_entries WHERE owner_id=? AND session_id=? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (owner_id, session_id),
            ).fetchone()
            conn.execute(
                "INSERT INTO session_entries(entry_id, owner_id, session_id, parent_entry_id, "
                "entry_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id, owner_id, session_id, parent[0] if parent else None,
                    "branch_summary", json.dumps(payload, ensure_ascii=False), time.time(),
                ),
            )
            conn.commit()
            return entry_id, payload

    def get_session_entry(self, owner_id: str, entry_id: str) -> dict[str, Any] | None:
        """Read one internal session entry owned by the caller."""
        with self._lock:
            row = self._get_conn().execute(
                "SELECT entry_id, entry_type, payload_json, created_at "
                "FROM session_entries WHERE owner_id=? AND entry_id=?",
                (owner_id, entry_id),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[2] or "{}")
        except json.JSONDecodeError:
            payload = {}
        return {
            "entry_id": row[0],
            "entry_type": row[1],
            "payload": payload if isinstance(payload, dict) else {},
            "created_at": float(row[3]),
        }

    def update_session_entry(
        self, owner_id: str, entry_id: str, payload: dict[str, Any],
    ) -> bool:
        """Replace one internal session entry payload atomically."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "UPDATE session_entries SET payload_json=? "
                "WHERE owner_id=? AND entry_id=?",
                (json.dumps(payload, ensure_ascii=False, default=str), owner_id, entry_id),
            )
            conn.commit()
        return cur.rowcount == 1

    def list_pending_session_entries(
        self, entry_type: str, statuses: tuple[str, ...], limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List internal entries that need process-restart recovery."""
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        bounded_limit = max(1, min(limit, 500))
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT entry_id, owner_id, session_id, payload_json, created_at "
                "FROM session_entries WHERE entry_type=? "
                f"AND json_extract(payload_json, '$.status') IN ({placeholders}) "
                "ORDER BY created_at ASC, rowid ASC LIMIT ?",
                (entry_type, *statuses, bounded_limit),
            ).fetchall()
        entries: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row[3] or "{}")
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or payload.get("status") not in statuses:
                continue
            entries.append({
                "entry_id": row[0],
                "owner_id": row[1],
                "session_id": row[2],
                "payload": payload,
                "created_at": float(row[4]),
            })
        return entries

    def list_session_entries(
        self,
        owner_id: str,
        session_id: str,
        entry_type: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Read user-owned session entries without exposing owner_id."""
        clauses = ["owner_id=?", "session_id=?"]
        params: list[Any] = [owner_id, session_id]
        if entry_type:
            clauses.append("entry_type=?")
            params.append(entry_type)
        params.append(max(1, min(limit, 100)))
        with self._lock:
            rows = self._get_conn().execute(
                "SELECT entry_id, entry_type, payload_json, created_at "
                "FROM session_entries WHERE " + " AND ".join(clauses)
                + " ORDER BY created_at DESC, rowid DESC LIMIT ?",
                params,
            ).fetchall()
        entries: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row[2] or "{}")
            entries.append({
                "entry_id": row[0],
                "entry_type": row[1],
                "payload": payload if isinstance(payload, dict) else {},
                "created_at": float(row[3]),
            })
        return entries

    def delete_session_entries(self, owner_id: str, session_id: str) -> int:
        """Delete all Runtime session entries for an application reset."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "DELETE FROM session_entries WHERE owner_id=? AND session_id=?",
                (owner_id, session_id),
            )
            conn.commit()
        return cur.rowcount

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
    if state.orchestration_run_id and state.step_id:
        now = time.time()
        conn.execute(
            """INSERT INTO orchestration_run_steps
               (run_id, step_id, operation_id, status, version,
                metadata_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, '{}', ?, ?)
               ON CONFLICT(run_id, step_id) DO UPDATE SET
                 operation_id=excluded.operation_id,
                 status=excluded.status,
                 version=orchestration_run_steps.version+1,
                 updated_at=excluded.updated_at""",
            (
                state.orchestration_run_id, state.step_id, state.operation_id,
                state.phase.value, now, now,
            ),
        )
    if state.orchestration_run_id:
        terminal = {
            OperationPhase.COMPLETED.value,
            OperationPhase.FAILED.value,
            OperationPhase.ABORTED.value,
            OperationPhase.RECOVERY_REQUIRED.value,
        }
        if state.operation_scope == "top_level" and state.phase.value in terminal:
            conn.execute(
                "UPDATE orchestration_runs SET status=?, updated_at=? WHERE run_id=?",
                (state.phase.value, time.time(), state.orchestration_run_id),
            )
        elif state.operation_scope == "dag_step":
            step_rows = conn.execute(
                "SELECT status FROM orchestration_run_steps WHERE run_id=?",
                (state.orchestration_run_id,),
            ).fetchall()
            if step_rows and all(row["status"] in terminal for row in step_rows):
                run_status = (
                    OperationPhase.COMPLETED.value
                    if all(row["status"] == OperationPhase.COMPLETED.value for row in step_rows)
                    else OperationPhase.FAILED.value
                )
                conn.execute(
                    "UPDATE orchestration_runs SET status=?, updated_at=? WHERE run_id=?",
                    (run_status, time.time(), state.orchestration_run_id),
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
        created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
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
        created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
        approval_set_id=row["approval_set_id"] or "",
    )


def _approval_set_from_row(row: sqlite3.Row) -> ApprovalSet:
    return ApprovalSet(
        approval_set_id=row["approval_set_id"],
        owner_id=row["owner_id"],
        operation_id=row["operation_id"],
        orchestration_run_id=row["orchestration_run_id"] or "",
        status=ApprovalSetStatus(row["status"]),
        version=int(row["version"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _approval_set_status(approvals: tuple[Approval, ...]) -> ApprovalSetStatus:
    if not approvals:
        return ApprovalSetStatus.PENDING
    statuses = {item.status for item in approvals}
    if ApprovalStatus.PENDING in statuses:
        return (
            ApprovalSetStatus.PARTIALLY_DECIDED
            if len(statuses) > 1 else ApprovalSetStatus.PENDING
        )
    if ApprovalStatus.EXPIRED in statuses:
        return ApprovalSetStatus.EXPIRED
    if ApprovalStatus.CANCELLED in statuses:
        return ApprovalSetStatus.CANCELLED
    if all(item.status is ApprovalStatus.APPROVED for item in approvals):
        return ApprovalSetStatus.APPROVED
    return ApprovalSetStatus.SKIPPED


def _refresh_approval_set(
    conn: sqlite3.Connection, approval_set_id: str, updated_at: float,
) -> None:
    rows = conn.execute(
        "SELECT status, decision FROM runtime_approvals WHERE approval_set_id=?",
        (approval_set_id,),
    ).fetchall()
    if not rows:
        return
    statuses = {ApprovalStatus(row["status"]) for row in rows}
    if ApprovalStatus.PENDING in statuses:
        status = (
            ApprovalSetStatus.PARTIALLY_DECIDED.value
            if len(statuses) > 1 else ApprovalSetStatus.PENDING.value
        )
    elif ApprovalStatus.EXPIRED in statuses:
        status = ApprovalSetStatus.EXPIRED.value
    elif ApprovalStatus.CANCELLED in statuses:
        status = ApprovalSetStatus.CANCELLED.value
    elif all(item is ApprovalStatus.APPROVED for item in statuses):
        status = ApprovalSetStatus.APPROVED.value
    else:
        status = ApprovalSetStatus.SKIPPED.value
    conn.execute(
        "UPDATE runtime_approval_sets SET status=?, version=version+1, updated_at=? "
        "WHERE approval_set_id=?",
        (status, updated_at, approval_set_id),
    )
