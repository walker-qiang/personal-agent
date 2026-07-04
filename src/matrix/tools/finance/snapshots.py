"""snapshot_history and recent_snapshots tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import FinanceToolError, ToolDefinition
from .shared import (
    asset_row,
    clamp_int,
    connect_readonly,
    recent_snapshot_row,
    snapshot_row,
)


def snapshot_history(
    asset_id: str = "",
    since: str = "",
    until: str = "",
    limit: int = 20,
    cache_path: str = "",
) -> dict[str, Any]:
    """Return effective snapshot history for one durable asset id."""
    path = Path(cache_path) if cache_path else Path("var/cache/finance.sqlite")
    if not asset_id.startswith("ast_"):
        raise FinanceToolError("asset_id must be a durable ast_* id")
    limit = clamp_int(limit, 1, 200)
    where = ["a.fact_id = ?"]
    params: list[Any] = [asset_id]
    if since:
        where.append("s.snapshot_date >= ?")
        params.append(since)
    if until:
        where.append("s.snapshot_date <= ?")
        params.append(until)
    params.append(limit)
    conn = connect_readonly(path)
    try:
        asset = conn.execute(
            """SELECT fact_id, code, name, asset_type, bucket, channel, currency,
                      risk_level, holding_cost_pct, expected_yield_pct, status,
                      notes, created_at, updated_at, archived_at
               FROM assets WHERE fact_id = ?""",
            [asset_id],
        ).fetchone()
        if asset is None:
            raise FinanceToolError(f"asset not found: {asset_id}")
        rows = conn.execute(
            f"""SELECT s.fact_id, a.fact_id AS asset_fact_id, a.code, a.name,
                       s.snapshot_date, s.balance_cents, s.expected_yield_pct,
                       s.actual_yield_pct, s.notes, s.created_at, s.correction_of
                FROM snapshots s
                JOIN assets a ON a.id = s.asset_id
                WHERE {" AND ".join(where)}
                ORDER BY s.snapshot_date DESC, s.fact_id DESC
                LIMIT ?""",
            params,
        ).fetchall()
    finally:
        conn.close()
    snapshots = [snapshot_row(row) for row in rows]
    return {
        "asset": asset_row(asset),
        "count": len(snapshots),
        "snapshots": snapshots,
    }


def recent_snapshots(
    query: str = "",
    allocation_bucket: str = "",
    asset_type: str = "",
    limit: int = 20,
    cache_path: str = "",
) -> dict[str, Any]:
    """Return latest effective snapshots across active assets, optionally filtered."""
    path = Path(cache_path) if cache_path else Path("var/cache/finance.sqlite")
    limit = clamp_int(limit, 1, 200)
    where = ["a.archived_at IS NULL"]
    params: list[Any] = []
    if query:
        where.append("(a.fact_id = ? OR a.code LIKE ? OR a.name LIKE ?)")
        like = f"%{query}%"
        params.extend([query, like, like])
    if allocation_bucket:
        where.append("a.bucket = ?")
        params.append(allocation_bucket)
    if asset_type:
        where.append("a.asset_type = ?")
        params.append(asset_type)
    params.append(limit)
    conn = connect_readonly(path)
    try:
        rows = conn.execute(
            f"""SELECT a.fact_id AS asset_fact_id, a.code, a.name, a.asset_type,
                       a.bucket, a.channel, a.currency, s.fact_id AS snapshot_id,
                       s.snapshot_date, s.balance_cents, s.expected_yield_pct,
                       s.actual_yield_pct, s.notes, s.created_at, s.correction_of
                FROM assets a
                LEFT JOIN snapshots s
                  ON s.asset_id = a.id
                 AND s.snapshot_date = (SELECT MAX(snapshot_date) FROM snapshots WHERE asset_id = a.id)
                WHERE {" AND ".join(where)}
                ORDER BY s.snapshot_date DESC, a.bucket, a.asset_type, a.code
                LIMIT ?""",
            params,
        ).fetchall()
    finally:
        conn.close()
    snapshots = [recent_snapshot_row(row) for row in rows]
    return {
        "query": query,
        "allocation_bucket": allocation_bucket,
        "asset_type": asset_type,
        "count": len(snapshots),
        "snapshots": snapshots,
    }


history_tool = ToolDefinition(
    name="finance.snapshot_history",
    description="Return effective snapshot history for one durable asset id.",
    input_schema={
        "type": "object",
        "required": ["asset_id"],
        "properties": {
            "asset_id": {"type": "string"},
            "since": {"type": "string"},
            "until": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "additionalProperties": False,
    },
    handler=snapshot_history,
)

recent_tool = ToolDefinition(
    name="finance.recent_snapshots",
    description="Return latest effective snapshots across active assets, optionally filtered by asset id, code, name, bucket, or asset type.",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "allocation_bucket": {"type": "string"},
            "asset_type": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "additionalProperties": False,
    },
    handler=recent_snapshots,
)