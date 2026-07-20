"""Tests for ToolGuard — tool invocation safety checks."""

from matrix.guardrails import GuardConfig, ToolGuard, ToolGuardError


def _config(block_mode: bool = True, blacklist: list[str] | None = None) -> GuardConfig:
    return GuardConfig(
        tool_block_mode=block_mode,
        tool_blacklist=blacklist or [],
    )


def test_normal_tool_passes():
    guard = ToolGuard(_config())
    ok, reason = guard.check("finance.holdings_summary", {"market": "cn"})
    assert ok is True
    assert reason == ""


def test_blacklisted_tool():
    guard = ToolGuard(_config(blacklist=["dangerous_tool"]))
    ok, reason = guard.check("dangerous_tool", {})
    assert ok is False
    assert "blacklisted" in reason


def test_path_traversal_blocked():
    guard = ToolGuard(_config())
    ok, reason = guard.check("web_fetch", {"url": "../../etc/passwd"})
    assert ok is False
    assert "path_traversal" in reason


def test_etc_passwd_blocked():
    guard = ToolGuard(_config())
    ok, reason = guard.check("web_fetch", {"path": "/etc/passwd"})
    assert ok is False
    assert "path_traversal" in reason


def test_sql_injection_blocked():
    guard = ToolGuard(_config())
    ok, reason = guard.check("some_tool", {"query": "DROP TABLE users"})
    assert ok is False
    assert "sql_injection" in reason


def test_delete_from_blocked():
    guard = ToolGuard(_config())
    ok, reason = guard.check("some_tool", {"sql": "DELETE FROM users WHERE id=1"})
    assert ok is False
    assert "sql_injection" in reason


def test_normal_sql_passes():
    guard = ToolGuard(_config())
    ok, reason = guard.check("some_tool", {"query": "SELECT * FROM users"})
    assert ok is True


def test_empty_arguments():
    guard = ToolGuard(_config())
    ok, reason = guard.check("finance.holdings_summary", {})
    assert ok is True


def test_large_arguments():
    guard = ToolGuard(_config())
    guard._max_args_size = 10
    ok, reason = guard.check("some_tool", {"data": "a" * 20})
    assert ok is False
    assert "too_large" in reason


def test_non_string_values_not_flagged():
    guard = ToolGuard(_config())
    ok, reason = guard.check("some_tool", {"count": 100, "enabled": True})
    assert ok is True