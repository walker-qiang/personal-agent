"""Shared utilities for finance tools."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

from ..base import FinanceToolError


def connect_readonly(cache_path: Path) -> sqlite3.Connection:
    """Open a read-only SQLite connection to the finance cache."""
    if not cache_path.exists():
        raise FinanceToolError(f"finance cache does not exist: {cache_path}")
    uri = f"file:{quote(str(cache_path.resolve()), safe='/:')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def cents_to_yuan(value: int) -> float:
    return value / 100


def clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise FinanceToolError("limit must be an integer") from None
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


# ---- Row mapping helpers ----

def holding_row(row: sqlite3.Row) -> dict[str, Any]:
    balance_cents = row["balance_cents"]
    return {
        "asset_id": row["fact_id"],
        "asset_code": row["code"],
        "asset_name": row["name"],
        "asset_type": row["asset_type"],
        "allocation_bucket": row["bucket"],
        "channel": row["channel"],
        "currency": row["currency"],
        "risk_level": row["risk_level"],
        "snapshot_id": row["snapshot_id"],
        "as_of": row["snapshot_date"],
        "balance_cents": balance_cents,
        "balance_yuan": cents_to_yuan(balance_cents) if balance_cents is not None else None,
        "expected_annual_yield_pct": row["expected_yield_pct"],
        "actual_yield_pct": row["actual_yield_pct"],
    }


def asset_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["fact_id"],
        "code": row["code"],
        "name": row["name"],
        "asset_type": row["asset_type"],
        "allocation_bucket": row["bucket"],
        "channel": row["channel"],
        "currency": row["currency"],
        "risk_level": row["risk_level"],
        "holding_cost_pct": row["holding_cost_pct"],
        "expected_annual_yield_pct": row["expected_yield_pct"],
        "status": row["status"],
        "notes": row["notes"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "archived_at": row["archived_at"],
    }


def snapshot_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["fact_id"],
        "asset_id": row["asset_fact_id"],
        "asset_code": row["code"],
        "asset_name": row["name"],
        "snapshot_date": row["snapshot_date"],
        "balance_cents": row["balance_cents"],
        "balance_yuan": cents_to_yuan(row["balance_cents"]),
        "expected_annual_yield_pct": row["expected_yield_pct"],
        "actual_yield_pct": row["actual_yield_pct"],
        "notes": row["notes"],
        "created_at": row["created_at"],
        "correction_of": row["correction_of"],
    }


def recent_snapshot_row(row: sqlite3.Row) -> dict[str, Any]:
    balance_cents = row["balance_cents"]
    return {
        "asset_id": row["asset_fact_id"],
        "asset_code": row["code"],
        "asset_name": row["name"],
        "asset_type": row["asset_type"],
        "allocation_bucket": row["bucket"],
        "channel": row["channel"],
        "currency": row["currency"],
        "snapshot_id": row["snapshot_id"],
        "snapshot_date": row["snapshot_date"],
        "balance_cents": balance_cents,
        "balance_yuan": cents_to_yuan(balance_cents) if balance_cents is not None else None,
        "expected_annual_yield_pct": row["expected_yield_pct"],
        "actual_yield_pct": row["actual_yield_pct"],
        "notes": row["notes"] if row["notes"] is not None else "",
        "created_at": row["created_at"],
        "correction_of": row["correction_of"],
    }


def build_bucket_summary(
    holdings: list[dict[str, Any]], targets: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    total_cents = sum(row["balance_cents"] or 0 for row in holdings if row["currency"] == "CNY")
    for row in holdings:
        bucket = row["allocation_bucket"]
        item = buckets.setdefault(
            bucket,
            {
                "allocation_bucket": bucket,
                "asset_count": 0,
                "balance_cents": 0,
                "balance_yuan": 0.0,
                "current_pct": None,
                "target_pct": None,
                "delta_pct": None,
                "is_target_set": False,
            },
        )
        item["asset_count"] += 1
        if row["currency"] == "CNY" and row["balance_cents"] is not None:
            item["balance_cents"] += row["balance_cents"]
    for _bucket, item in buckets.items():
        item["balance_yuan"] = cents_to_yuan(item["balance_cents"])
        if total_cents > 0:
            item["current_pct"] = item["balance_cents"] * 100 / total_cents
        target = targets.get(_bucket)
        if target:
            item["target_pct"] = target["target_pct"]
            item["is_target_set"] = True
            if item["current_pct"] is not None:
                item["delta_pct"] = item["current_pct"] - target["target_pct"]
    ordered: list[dict[str, Any]] = []
    for bucket in ("cash", "stable", "growth"):
        if bucket in buckets:
            ordered.append(buckets.pop(bucket))
    for bucket in sorted(buckets):
        ordered.append(buckets[bucket])
    return ordered