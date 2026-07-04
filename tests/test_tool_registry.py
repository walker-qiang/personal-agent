"""Tests for ToolRegistry."""

from __future__ import annotations

import pytest

from matrix.tools import FinanceToolError, ToolDefinition, ToolRegistry
from matrix.tools.finance import register_all


class TestToolRegistry:
    def test_register_and_list(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        tools = registry.list_tools()
        assert len(tools) == 5
        names = {t["name"] for t in tools}
        assert names == {
            "finance.holdings_summary",
            "finance.asset_lookup",
            "finance.snapshot_history",
            "finance.recent_snapshots",
            "finance.bucket_allocation",
        }

    def test_tool_names_returns_set(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        names = registry.tool_names()
        assert len(names) == 5
        assert "finance.holdings_summary" in names

    def test_rejects_duplicate_registration(self):
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.tool",
            description="test",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: {"ok": True},
        )
        registry.register(td)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(td)

    def test_call_valid_tool(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        result = registry.call("finance.holdings_summary")
        assert result["holding_count"] == 2
        assert result["total_balance_cents"] == 35000

    def test_call_unknown_tool_raises(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        with pytest.raises(FinanceToolError, match="unknown tool"):
            registry.call("finance.unknown")

    def test_call_with_non_dict_arguments_raises(self):
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.tool",
            description="test",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: {"ok": True},
        )
        registry.register(td)
        with pytest.raises(FinanceToolError, match="arguments must be an object"):
            registry.call("test.tool", "not-a-dict")  # type: ignore

    def test_get_returns_tool_or_none(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        assert registry.get("finance.holdings_summary") is not None
        assert registry.get("nonexistent") is None

    def test_to_dict_format(self):
        td = ToolDefinition(
            name="test.tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=lambda x=0: {"result": x},
        )
        d = td.to_dict()
        assert d["name"] == "test.tool"
        assert d["description"] == "A test tool"
        assert d["input_schema"]["properties"]["x"]["type"] == "integer"
        assert "handler" not in d


class TestToolDefinition:
    def test_handler_not_in_repr(self):
        td = ToolDefinition(
            name="test.tool",
            description="test",
            input_schema={},
            handler=lambda: {},
        )
        r = repr(td)
        assert "lambda" not in r
        assert "handler" not in r