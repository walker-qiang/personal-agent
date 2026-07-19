"""holdings_summary tool — aggregate holdings by allocation bucket."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import ToolDefinition
from .shared import build_bucket_summary, cents_to_yuan, connect_readonly, holding_row


def holdings_summary(cache_path: str = "") -> dict[str, Any]:
    """Return current holdings and bucket summary."""
    path = Path(cache_path) if cache_path else Path("var/cache/finance.sqlite")
    conn = connect_readonly(path)
    try:
        holdings_rows = conn.execute(
            """SELECT a.fact_id, a.code, a.name, a.asset_type, a.bucket, a.channel,
                      a.currency, a.risk_level, s.fact_id AS snapshot_id, s.snapshot_date,
                      s.balance_cents, s.expected_yield_pct, s.actual_yield_pct
               FROM assets a
               LEFT JOIN snapshots s
                 ON s.asset_id = a.id
                AND s.snapshot_date = (SELECT MAX(snapshot_date) FROM snapshots WHERE asset_id = a.id)
               WHERE a.archived_at IS NULL
               ORDER BY a.bucket, a.asset_type, a.code"""
        ).fetchall()
        targets = {
            row["bucket"]: {
                "target_pct": row["target_pct"],
                "target_notes": row["notes"],
                "target_updated_at": row["updated_at"],
            }
            for row in conn.execute(
                "SELECT bucket, target_pct, notes, updated_at FROM bucket_targets ORDER BY bucket"
            ).fetchall()
        }
    finally:
        conn.close()
    holdings = [holding_row(row) for row in holdings_rows]
    buckets = build_bucket_summary(holdings, targets)
    total_cents = sum(row["balance_cents"] or 0 for row in holdings if row["currency"] == "CNY")
    return {
        "currency": "CNY",
        "total_balance_cents": total_cents,
        "total_balance_yuan": cents_to_yuan(total_cents),
        "bucket_count": len(buckets),
        "holding_count": len(holdings),
        "buckets": buckets,
        "holdings": holdings,
    }


tool_definition = ToolDefinition(
    name="finance.holdings_summary",
    description="查询当前持仓汇总（按投资桶分类）。用于：用户问「我的持仓」「资产分布」「各个桶有多少钱」「当前收益怎么样」。每个桶显示总金额、预期收益率、实际收益率。",
    input_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    handler=holdings_summary,
)