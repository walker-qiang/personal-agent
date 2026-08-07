"""Per-tool circuit breaker for orchestration nodes.

Prevents infinite retries on failing tools by tracking consecutive
failures and blocking calls during a cooldown period.
Split from _helpers.py.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("matrix.orchestration")

MAX_CONSECUTIVE_TOOL_FAILURES = 3  # Trip breaker after N consecutive failures of the same tool

CIRCUIT_BREAKER_COOLDOWN_SEC = 30  # Cooldown seconds before a tripped tool can be retried


class CircuitBreaker:
    """Per-tool circuit breaker to prevent infinite retries on failing tools.

    Tracks consecutive failures per tool name. When a tool exceeds
    MAX_CONSECUTIVE_TOOL_FAILURES, it is "tripped" — calls to that tool
    are blocked for CIRCUIT_BREAKER_COOLDOWN_SEC seconds.

    After cooldown expires, the tool is reset and can be tried again.
    """

    def __init__(self) -> None:
        self._failures: dict[str, int] = {}
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()

    def record_failure(self, tool_name: str) -> bool:
        """Record a tool failure. Returns True if the breaker just tripped."""
        with self._lock:
            self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
            if self._failures[tool_name] >= MAX_CONSECUTIVE_TOOL_FAILURES:
                self._cooldowns[tool_name] = time.time() + CIRCUIT_BREAKER_COOLDOWN_SEC
                logger.warning(
                    "circuit_breaker: tripped tool=%s after %d consecutive failures",
                    tool_name, self._failures[tool_name],
                )
                return True
            return False

    def record_success(self, tool_name: str) -> None:
        """Reset the failure counter for a tool after a successful call."""
        with self._lock:
            self._failures.pop(tool_name, None)
            self._cooldowns.pop(tool_name, None)

    def _is_blocked_unlocked(self, tool_name: str) -> bool:
        """Check if a tool is blocked — caller MUST hold self._lock."""
        cooldown_until = self._cooldowns.get(tool_name)
        if cooldown_until is None:
            return False
        if time.time() >= cooldown_until:
            # Cooldown expired — reset the breaker
            self._failures.pop(tool_name, None)
            self._cooldowns.pop(tool_name, None)
            logger.info("circuit_breaker: cooldown expired for tool=%s, resetting", tool_name)
            return False
        return True

    def is_blocked(self, tool_name: str) -> bool:
        """Check if a tool is currently blocked by the circuit breaker."""
        with self._lock:
            return self._is_blocked_unlocked(tool_name)

    def blocked_tools(self) -> set[str]:
        """Return the set of currently blocked tool names."""
        with self._lock:
            return {name for name in list(self._cooldowns) if self._is_blocked_unlocked(name)}

    def reset(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            self._failures.clear()
            self._cooldowns.clear()
