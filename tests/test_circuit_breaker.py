"""Tests for the CircuitBreaker class."""

import time
import pytest

from matrix.orchestration.nodes._helpers import CircuitBreaker


class TestCircuitBreaker:
    def test_init_not_blocked(self):
        cb = CircuitBreaker()
        assert not cb.is_blocked("web_search")
        assert cb.blocked_tools() == set()

    def test_single_failure_does_not_trip(self):
        cb = CircuitBreaker()
        tripped = cb.record_failure("web_search")
        assert not tripped
        assert not cb.is_blocked("web_search")

    def test_two_failures_does_not_trip(self):
        cb = CircuitBreaker()
        cb.record_failure("web_search")
        tripped = cb.record_failure("web_search")
        assert not tripped
        assert not cb.is_blocked("web_search")

    def test_three_failures_trips_breaker(self):
        cb = CircuitBreaker()
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        tripped = cb.record_failure("web_search")
        assert tripped
        assert cb.is_blocked("web_search")
        assert "web_search" in cb.blocked_tools()

    def test_success_resets_breaker(self):
        cb = CircuitBreaker()
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_success("web_search")
        # After success, failure counter should be reset
        assert not cb.is_blocked("web_search")
        assert cb.blocked_tools() == set()

    def test_success_after_trip_resets(self):
        cb = CircuitBreaker()
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_failure("web_search")  # trips
        assert cb.is_blocked("web_search")
        cb.record_success("web_search")
        assert not cb.is_blocked("web_search")

    def test_isolated_per_tool(self):
        cb = CircuitBreaker()
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_failure("web_search")  # trips web_search
        assert cb.is_blocked("web_search")
        assert not cb.is_blocked("news_search")
        assert not cb.is_blocked("web_fetch")

    def test_cooldown_expires(self):
        """Simulate cooldown expiry by directly manipulating the internal state."""
        cb = CircuitBreaker()
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_failure("web_search")  # trips
        assert cb.is_blocked("web_search")

        # Simulate cooldown expiry by setting cooldown to past
        cb._cooldowns["web_search"] = time.time() - 1
        assert not cb.is_blocked("web_search")
        assert cb.blocked_tools() == set()

    def test_reset_clears_all(self):
        cb = CircuitBreaker()
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_failure("web_search")
        cb.record_failure("news_search")
        assert cb.is_blocked("web_search")
        cb.reset()
        assert not cb.is_blocked("web_search")
        assert not cb.is_blocked("news_search")
        assert cb.blocked_tools() == set()

    def test_multiple_tools_can_be_blocked(self):
        cb = CircuitBreaker()
        for _ in range(3):
            cb.record_failure("web_search")
        for _ in range(3):
            cb.record_failure("news_search")
        assert cb.is_blocked("web_search")
        assert cb.is_blocked("news_search")
        assert "web_search" in cb.blocked_tools()
        assert "news_search" in cb.blocked_tools()