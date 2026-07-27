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

    def test_call_unknown_tool_returns_error(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        result = registry.call("finance.unknown")
        assert "error" in result
        assert "不存在" in result["error"]

    def test_call_with_non_dict_arguments_returns_error(self):
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.tool",
            description="test",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: {"ok": True},
        )
        registry.register(td)
        result = registry.call("test.tool", "not-a-dict")  # type: ignore
        assert "error" in result
        assert "object" in result["error"]

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

class TestToolPipeline:
    """Tests for the five-step pipeline (Phase 2)."""

    def test_prepare_arguments_drops_unknown_fields(self):
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.echo",
            description="echo back args",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=lambda x: {"x": x},
        )
        registry.register(td)
        # Extra field "y" should be dropped
        result = registry.call("test.echo", {"x": 42, "y": "extra"})
        assert result == {"x": 42}

    def test_prepare_arguments_restores_stringified_array(self):
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.array",
            description="accept array",
            input_schema={
                "type": "object",
                "properties": {"items": {"type": "array"}},
            },
            handler=lambda items: {"count": len(items)},
        )
        registry.register(td)
        # Some LLMs serialize arrays as strings
        result = registry.call("test.array", {"items": "[1, 2, 3]"})
        assert result == {"count": 3}

    def test_validate_arguments_missing_required(self):
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.required",
            description="requires x",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            handler=lambda x: {"x": x},
        )
        registry.register(td)
        result = registry.call("test.required", {})
        assert "error" in result
        assert "缺少必需参数" in result["error"]

    def test_validate_arguments_wrong_type(self):
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.typed",
            description="typed param",
            input_schema={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
            },
            handler=lambda n: {"doubled": n * 2},
        )
        registry.register(td)
        result = registry.call("test.typed", {"n": "not-a-number"})
        assert "error" in result
        assert "类型错误" in result["error"]

    def test_execute_error_returns_error_dict(self):
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.failing",
            description="always fails",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: (_ for _ in ()).throw(ValueError("boom")),
        )
        registry.register(td)
        result = registry.call("test.failing")
        assert "error" in result
        assert "boom" in result["error"]
        assert "test.failing" in result["error"]

    def test_truncate_applied_to_result(self):
        from matrix.tools.truncate import truncate_result
        # Simulate a tool returning a large list
        registry = ToolRegistry()
        td = ToolDefinition(
            name="test.big",
            description="returns big data",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: {"holdings": [{"id": i} for i in range(100)]},
        )
        registry.register(td)
        result = registry.call("test.big")
        assert len(result["holdings"]) == 50
        assert result["_holdings_original_count"] == 100
