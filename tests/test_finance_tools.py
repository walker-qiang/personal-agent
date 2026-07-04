"""Tests for finance tools against real SQLite cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from matrix.chat import result_count
from matrix.tools import FinanceToolError, ToolRegistry
from matrix.tools.finance import register_all


@pytest.fixture
def registry(tmp_cache_path: Path) -> ToolRegistry:
    r = ToolRegistry()
    register_all(r, tmp_cache_path)
    return r


class TestHoldingsSummary:
    def test_returns_bucket_totals(self, registry):
        result = registry.call("finance.holdings_summary")
        assert result["currency"] == "CNY"
        assert result["total_balance_cents"] == 35000
        assert result["total_balance_yuan"] == 350.0
        assert result["holding_count"] == 2
        buckets = {b["allocation_bucket"]: b for b in result["buckets"]}
        assert list(buckets.keys()) == ["cash", "growth"]
        assert buckets["cash"]["balance_cents"] == 15000
        assert buckets["cash"]["target_pct"] == 40.0
        assert buckets["cash"]["current_pct"] == pytest.approx(42.857142857142854)
        assert buckets["cash"]["is_target_set"] is True

    def test_holdings_list_has_expected_fields(self, registry):
        result = registry.call("finance.holdings_summary")
        holding = result["holdings"][0]
        assert "asset_id" in holding
        assert "asset_code" in holding
        assert "asset_name" in holding
        assert "balance_cents" in holding
        assert "balance_yuan" in holding


class TestAssetLookup:
    def test_matches_durable_id(self, registry):
        result = registry.call("finance.asset_lookup", {"query": "ast_sample_cash"})
        assert result["count"] == 1
        assert result["assets"][0]["code"] == "sample-cash"

    def test_matches_name(self, registry):
        result = registry.call("finance.asset_lookup", {"query": "Fund"})
        assert result["count"] == 1
        assert result["assets"][0]["id"] == "ast_sample_fund"

    def test_returns_empty_when_no_match(self, registry):
        result = registry.call("finance.asset_lookup", {"query": "nonexistent"})
        assert result["count"] == 0
        assert result["assets"] == []

    def test_includes_archived_when_requested(self, registry):
        result = registry.call(
            "finance.asset_lookup",
            {"query": "sample", "include_archived": True},
        )
        assert result["count"] == 2


class TestSnapshotHistory:
    def test_filters_effective_rows(self, registry):
        result = registry.call(
            "finance.snapshot_history",
            {"asset_id": "ast_sample_cash", "since": "2026-05-02", "limit": 5},
        )
        assert result["asset"]["code"] == "sample-cash"
        assert result["count"] == 1
        assert result["snapshots"][0]["id"] == "snap_cash_2"
        assert result["snapshots"][0]["balance_cents"] == 15000

    def test_requires_ast_prefix(self, registry):
        with pytest.raises(FinanceToolError, match="ast_\\*"):
            registry.call("finance.snapshot_history", {"asset_id": "not-an-ast-id"})

    def test_raises_for_unknown_asset(self, registry):
        with pytest.raises(FinanceToolError, match="asset not found"):
            registry.call("finance.snapshot_history", {"asset_id": "ast_nonexistent"})


class TestRecentSnapshots:
    def test_filters_by_query_and_bucket(self, registry):
        result = registry.call(
            "finance.recent_snapshots",
            {"query": "sample-cash", "allocation_bucket": "cash", "limit": 5},
        )
        assert result["count"] == 1
        s = result["snapshots"][0]
        assert s["asset_id"] == "ast_sample_cash"
        assert s["snapshot_id"] == "snap_cash_2"
        assert s["snapshot_date"] == "2026-05-02"
        assert s["balance_cents"] == 15000

    def test_returns_all_active_without_filters(self, registry):
        result = registry.call("finance.recent_snapshots", {"limit": 10})
        assert result["count"] == 2

    def test_filters_by_asset_type(self, registry):
        result = registry.call(
            "finance.recent_snapshots",
            {"asset_type": "fund", "limit": 10},
        )
        assert result["count"] == 1
        assert result["snapshots"][0]["asset_name"] == "Sample Fund"


class TestBucketAllocation:
    def test_excludes_holdings_detail(self, registry):
        result = registry.call("finance.bucket_allocation")
        assert "buckets" in result
        assert "holdings" not in result
        assert result_count(result) == 2

    def test_includes_total_balance(self, registry):
        result = registry.call("finance.bucket_allocation")
        assert result["total_balance_cents"] == 35000
        assert result["total_balance_yuan"] == 350.0


class TestErrorHandling:
    def test_rejects_missing_cache(self, tmp_dir):
        registry = ToolRegistry()
        register_all(registry, tmp_dir / "missing.sqlite")
        with pytest.raises(FinanceToolError, match="finance cache does not exist"):
            registry.call("finance.holdings_summary")