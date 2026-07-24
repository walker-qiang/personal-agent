"""Unit tests for browser automation tools.

Tests cover:
- URL validation (security)
- Text truncation
- High-risk tool classification (HITL)
- MCP config loading
- Agent tool registration
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =====================================================================
# URL Validation Tests
# =====================================================================

class TestURLValidation:
    """Test _validate_url function in browser_tools.py."""

    @pytest.fixture(autouse=True)
    def _load_browser_tools(self):
        """Load browser_tools.py as a standalone module."""
        import importlib.util
        bt_path = PROJECT_ROOT / "var" / "mcp" / "browser_tools.py"
        spec = importlib.util.spec_from_file_location("browser_tools_test", bt_path)
        self.bt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bt)

    def test_valid_https_url(self):
        ok, msg = self.bt._validate_url("https://example.com")
        assert ok is True
        assert msg == ""

    def test_valid_http_url(self):
        ok, msg = self.bt._validate_url("http://example.com/page")
        assert ok is True
        assert msg == ""

    def test_empty_url(self):
        ok, msg = self.bt._validate_url("")
        assert ok is False
        assert "为空" in msg

    def test_file_protocol_blocked(self):
        ok, msg = self.bt._validate_url("file:///etc/passwd")
        assert ok is False
        assert "file" in msg

    def test_javascript_protocol_blocked(self):
        ok, msg = self.bt._validate_url("javascript:alert(1)")
        assert ok is False
        assert "javascript" in msg

    def test_data_protocol_blocked(self):
        ok, msg = self.bt._validate_url("data:text/html,<script>alert(1)</script>")
        assert ok is False
        assert "data" in msg

    def test_localhost_blocked(self):
        ok, msg = self.bt._validate_url("http://localhost:8080")
        assert ok is False
        assert "localhost" in msg

    def test_127_ip_blocked(self):
        ok, msg = self.bt._validate_url("http://127.0.0.1:3000")
        assert ok is False
        assert "127" in msg

    def test_metadata_endpoint_blocked(self):
        ok, msg = self.bt._validate_url("http://169.254.169.254/latest/meta-data")
        assert ok is False

    def test_non_http_protocol_blocked(self):
        ok, msg = self.bt._validate_url("ftp://example.com/file")
        assert ok is False
        assert "http" in msg


# =====================================================================
# Text Truncation Tests
# =====================================================================

class TestTextTruncation:
    """Test _truncate function."""

    @pytest.fixture(autouse=True)
    def _load_browser_tools(self):
        import importlib.util
        bt_path = PROJECT_ROOT / "var" / "mcp" / "browser_tools.py"
        spec = importlib.util.spec_from_file_location("browser_tools_test", bt_path)
        self.bt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bt)

    def test_short_text_not_truncated(self):
        text = "Hello, World!"
        result = self.bt._truncate(text, max_chars=100)
        assert result == text

    def test_long_text_truncated(self):
        text = "A" * 200
        result = self.bt._truncate(text, max_chars=100)
        assert len(result) > 100  # includes truncation message
        assert "截断" in result
        assert result.startswith("A" * 100)

    def test_default_max_chars(self):
        text = "B" * 25000
        result = self.bt._truncate(text)
        assert "截断" in result


# =====================================================================
# HITL High-Risk Classification Tests
# =====================================================================

class TestHighRiskClassification:
    """Test that browser interactive tools are classified as high-risk."""

    @pytest.fixture(autouse=True)
    def _load_helpers(self):
        from matrix.orchestration.nodes._helpers import _is_high_risk, _HIGH_RISK_PATTERNS
        self._is_high_risk = _is_high_risk
        self._patterns = _HIGH_RISK_PATTERNS

    def test_browser_click_is_high_risk(self):
        assert self._is_high_risk("mcp_browser_click") is True

    def test_browser_type_is_high_risk(self):
        assert self._is_high_risk("mcp_browser_type") is True

    def test_browser_select_option_is_high_risk(self):
        assert self._is_high_risk("mcp_browser_select_option") is True

    def test_browser_press_key_is_high_risk(self):
        assert self._is_high_risk("mcp_browser_press_key") is True

    def test_browser_restore_state_is_high_risk(self):
        assert self._is_high_risk("mcp_browser_restore_state") is True

    def test_browser_save_state_is_high_risk(self):
        assert self._is_high_risk("mcp_browser_save_state") is True

    def test_browser_navigate_is_not_high_risk(self):
        assert self._is_high_risk("mcp_browser_navigate") is False

    def test_browser_snapshot_is_not_high_risk(self):
        assert self._is_high_risk("mcp_browser_snapshot") is False

    def test_browser_extract_is_not_high_risk(self):
        assert self._is_high_risk("mcp_browser_extract") is False

    def test_browser_screenshot_is_not_high_risk(self):
        assert self._is_high_risk("mcp_browser_screenshot") is False

    def test_browser_wait_for_is_not_high_risk(self):
        assert self._is_high_risk("mcp_browser_wait_for") is False

    def test_browser_scroll_is_not_high_risk(self):
        assert self._is_high_risk("mcp_browser_scroll") is False

    def test_browser_get_cookies_is_not_high_risk(self):
        assert self._is_high_risk("mcp_browser_get_cookies") is False

    def test_browser_close_is_not_high_risk(self):
        assert self._is_high_risk("mcp_browser_close") is False


# =====================================================================
# MCP Config Tests
# =====================================================================

class TestMCPConfig:
    """Test that browser server config loads correctly."""

    def test_browser_server_in_config(self):
        from matrix.tools.mcp.config import load_mcp_config
        config_path = PROJECT_ROOT / "config" / "mcp_servers.json"
        servers = load_mcp_config(str(config_path))

        server_names = [s.name for s in servers]
        assert "browser" in server_names
        assert "utility" in server_names

    def test_browser_server_config_valid(self):
        from matrix.tools.mcp.config import load_mcp_config
        config_path = PROJECT_ROOT / "config" / "mcp_servers.json"
        servers = load_mcp_config(str(config_path))

        browser_server = next(s for s in servers if s.name == "browser")
        errors = browser_server.validate()
        assert errors == [], f"Browser config invalid: {errors}"
        assert browser_server.transport == "stdio"
        assert browser_server.enabled is True
        assert browser_server.timeout == 120.0
        assert "browser_tools.py" in browser_server.args[0]


# =====================================================================
# Agent Tool Registration Tests
# =====================================================================

class TestAgentToolRegistration:
    """Test that browser tools are registered for investment_analyst."""

    def test_investment_analyst_has_browser_tools(self):
        from matrix.agent.domain_agents.investment_analyst import INVESTMENT_ANALYST
        tools = INVESTMENT_ANALYST.tools

        # Should have read-only browser tools
        assert "mcp_browser_navigate" in tools
        assert "mcp_browser_snapshot" in tools
        assert "mcp_browser_extract" in tools
        assert "mcp_browser_screenshot" in tools

    def test_investment_analyst_no_interactive_browser_tools(self):
        from matrix.agent.domain_agents.investment_analyst import INVESTMENT_ANALYST
        tools = INVESTMENT_ANALYST.tools

        # Should NOT have interactive browser tools (safety)
        assert "mcp_browser_click" not in tools
        assert "mcp_browser_type" not in tools
        assert "mcp_browser_select_option" not in tools
        assert "mcp_browser_press_key" not in tools

    def test_commander_has_all_tools(self):
        """Commander with tools=[] should match all tools including browser."""
        from matrix.agent.base import AgentDefinition
        commander = AgentDefinition(
            id="commander",
            name="Commander",
            description="test",
            domain="commander",
            persona="test",
            tools=[],  # empty = all tools
        )
        assert commander.matches_tool("mcp_browser_navigate") is True
        assert commander.matches_tool("mcp_browser_click") is True

    def test_investment_analyst_matches_browser_tools(self):
        from matrix.agent.domain_agents.investment_analyst import INVESTMENT_ANALYST
        assert INVESTMENT_ANALYST.matches_tool("mcp_browser_navigate") is True
        assert INVESTMENT_ANALYST.matches_tool("mcp_browser_extract") is True
        assert INVESTMENT_ANALYST.matches_tool("mcp_browser_click") is False


# =====================================================================
# Indirect Injection Guard Tests
# =====================================================================

class TestIndirectInjectionGuard:
    """Test that browser tools are covered by injection guard."""

    def test_mcp_browser_tools_are_high_risk_for_injection(self):
        from matrix.guardrails.indirect_injection_guard import _is_high_risk
        # All MCP tools should be checked for injection
        assert _is_high_risk("mcp_browser_navigate") is True
        assert _is_high_risk("mcp_browser_extract") is True
        assert _is_high_risk("mcp_browser_click") is True


# =====================================================================
# Performance Config Tests
# =====================================================================

class TestPerformanceConfig:
    """Test that performance-related env vars are read correctly."""

    @pytest.fixture(autouse=True)
    def _load_browser_tools(self):
        import importlib.util
        bt_path = PROJECT_ROOT / "var" / "mcp" / "browser_tools.py"
        spec = importlib.util.spec_from_file_location("browser_tools_test", bt_path)
        self.bt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bt)

    def test_max_elements_has_default(self):
        """_MAX_ELEMENTS should default to 50."""
        assert self.bt._MAX_ELEMENTS == 50

    def test_block_resources_has_default(self):
        """_BLOCK_RESOURCES should default to True."""
        assert self.bt._BLOCK_RESOURCES is True

    def test_max_elements_is_positive(self):
        """_MAX_ELEMENTS should be a positive integer."""
        assert isinstance(self.bt._MAX_ELEMENTS, int)
        assert self.bt._MAX_ELEMENTS > 0

    def test_inject_refs_accepts_max_elements_param(self):
        """The JS code in _inject_refs should accept a maxElements parameter."""
        import inspect
        source = inspect.getsource(self.bt._inject_refs)
        assert "maxElements" in source
        assert "_MAX_ELEMENTS" in source
