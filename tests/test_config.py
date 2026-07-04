"""Tests for config module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from matrix.config import (
    AgentConfig,
    clamp_float_env,
    clamp_int_env,
    find_root,
    load_config,
    load_bind_addr,
    parse_addr,
)


class TestClampIntEnv:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_CLAMP", raising=False)
        assert clamp_int_env("TEST_CLAMP", 42, 1, 100) == 42

    def test_returns_parsed_value(self, monkeypatch):
        monkeypatch.setenv("TEST_CLAMP", "50")
        assert clamp_int_env("TEST_CLAMP", 42, 1, 100) == 50

    def test_clamps_below_min(self, monkeypatch):
        monkeypatch.setenv("TEST_CLAMP", "0")
        assert clamp_int_env("TEST_CLAMP", 42, 10, 100) == 10

    def test_clamps_above_max(self, monkeypatch):
        monkeypatch.setenv("TEST_CLAMP", "200")
        assert clamp_int_env("TEST_CLAMP", 42, 1, 100) == 100

    def test_returns_default_on_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_CLAMP", "not-a-number")
        assert clamp_int_env("TEST_CLAMP", 42, 1, 100) == 42


class TestClampFloatEnv:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_FLOAT", raising=False)
        assert clamp_float_env("TEST_FLOAT", 3.14, 0.0, 10.0) == 3.14

    def test_clamps_below_min(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "-1.0")
        assert clamp_float_env("TEST_FLOAT", 3.14, 0.0, 10.0) == 0.0

    def test_returns_default_on_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "abc")
        assert clamp_float_env("TEST_FLOAT", 3.14, 0.0, 10.0) == 3.14


class TestParseAddr:
    def test_parses_host_and_port(self):
        host, port = parse_addr("127.0.0.1:8080")
        assert host == "127.0.0.1"
        assert port == 8080

    def test_parses_ipv6(self):
        host, port = parse_addr("[::1]:8080")
        assert host == "[::1]"
        assert port == 8080

    def test_requires_colon(self):
        with pytest.raises(ValueError, match="host:port"):
            parse_addr("no-colon")

    def test_requires_valid_port(self):
        with pytest.raises(ValueError, match="port must be an integer"):
            parse_addr("host:abc")

    def test_rejects_port_out_of_range(self):
        with pytest.raises(ValueError, match="port out of range"):
            parse_addr("host:99999")


class TestFindRoot:
    def test_finds_pyproject_toml(self):
        # Should find the personal-agent root
        root = find_root(Path(__file__))
        assert (root / "pyproject.toml").exists()

    def test_raises_when_no_pyproject(self, tmp_dir):
        with pytest.raises(RuntimeError, match="matrix root not found"):
            find_root(tmp_dir)


class TestLoadBindAddr:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("MATRIX_AGENT_ADDR", raising=False)
        monkeypatch.delenv("PERSONAL_OS_AGENT_ADDR", raising=False)
        monkeypatch.delenv("AGENT_HOST", raising=False)
        monkeypatch.delenv("AGENT_PORT", raising=False)
        host, port = load_bind_addr()
        assert host == "127.0.0.1"
        assert port == 7101

    def test_matrix_addr_env(self, monkeypatch):
        monkeypatch.setenv("MATRIX_AGENT_ADDR", "0.0.0.0:9000")
        monkeypatch.delenv("PERSONAL_OS_AGENT_ADDR", raising=False)
        host, port = load_bind_addr()
        assert host == "0.0.0.0"
        assert port == 9000

    def test_personal_os_addr_takes_priority(self, monkeypatch):
        monkeypatch.setenv("MATRIX_AGENT_ADDR", "0.0.0.0:9000")
        monkeypatch.setenv("PERSONAL_OS_AGENT_ADDR", "10.0.0.1:8000")
        host, port = load_bind_addr()
        assert host == "10.0.0.1"
        assert port == 8000


class TestAgentConfig:
    def test_llm_available_when_key_set(self, agent_config):
        assert agent_config.llm_available
        assert agent_config.llm_unavailable_reason == ""

    def test_llm_unavailable_when_no_key(self, tmp_cache_path):
        config = AgentConfig(
            root_path=tmp_cache_path.parent,
            cache_path=tmp_cache_path,
            trace_path=tmp_cache_path.parent / "trace.jsonl",
            host="127.0.0.1",
            port=0,
        )
        assert not config.llm_available
        assert "missing DEEPSEEK_API_KEY" in config.llm_unavailable_reason

    def test_active_api_key_returns_correct_provider(self, agent_config):
        assert agent_config.active_api_key == "test-key"

    def test_anthropic_config_returns_anthropic_key(self, tmp_cache_path):
        config = AgentConfig(
            root_path=tmp_cache_path.parent,
            cache_path=tmp_cache_path,
            trace_path=tmp_cache_path.parent / "trace.jsonl",
            host="127.0.0.1",
            port=0,
            agent_provider="anthropic",
            anthropic_api_key="claude-key",
        )
        assert config.active_api_key == "claude-key"


class TestLoadConfig:
    def test_loads_defaults(self, monkeypatch):
        """load_config should work with defaults in the personal-agent repo."""
        monkeypatch.delenv("PERSONAL_OS_CACHE_PATH", raising=False)
        monkeypatch.delenv("MATRIX_CACHE_PATH", raising=False)
        # Need to run from inside the repo for find_root
        config = load_config()
        assert config.host == "127.0.0.1"
        assert config.port == 7101
        assert "var/cache/finance.sqlite" in str(config.cache_path)

    def test_personal_os_cache_priority(self, monkeypatch, tmp_dir):
        os_cache = tmp_dir / "os-cache.sqlite"
        os_cache.touch()
        monkeypatch.setenv("PERSONAL_OS_CACHE_PATH", str(os_cache))
        monkeypatch.setenv("MATRIX_CACHE_PATH", str(tmp_dir / "matrix-cache.sqlite"))
        config = load_config()
        assert config.cache_path == os_cache.resolve()