"""asset_lookup tool — find active assets by query."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import FinanceToolError, ToolDefinition
from .shared import asset_row, clamp_int, connect_readonly


def asset_lookup(
    query: str = "",
    include_archived: bool = False,
    limit: int = 20,
    cache_path: str = "",
) -> dict[str, Any]:
    """Find active assets by durable id, code, or name."""
    path = Path(cache_path) if cache_path else Path("var/cache/finance.sqlite")
    limit = clamp_int(limit, 1, 100)
    where = []
    params: list[Any] = []
    if not include_archived:
        where.append("archived_at IS NULL")
    if query:
        where.append("(fact_id = ? OR code LIKE ? OR name LIKE ?)")
        like = f"%{query}%"
        params.extend([query, like, like])
    sql = """SELECT fact_id, code, name, asset_type, bucket, channel, currency,
                    risk_level, holding_cost_pct, expected_yield_pct, status,
                    notes, created_at, updated_at, archived_at
             FROM assets"""
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY bucket, asset_type, code LIMIT ?"
    params.append(limit)
    conn = connect_readonly(path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    assets = [asset_row(row) for row in rows]
    return {
        "query": query,
        "count": len(assets),
        "assets": assets,
    }


tool_definition = ToolDefinition(
    name="finance.asset_lookup",
    description="按 ID、代码或名称查找资产。用于：用户问「某某基金怎么样」「查一下某个股票」「某资产收益如何」。返回资产基本信息、最新快照、收益率等。",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "include_archived": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "additionalProperties": False,
    },
    handler=asset_lookup,
)