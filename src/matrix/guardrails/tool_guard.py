"""ToolGuard: tool invocation safety checks.

Validates tool calls before execution: blacklist, parameter size,
path traversal, SQL injection, and per-session rate limiting.
Default mode is block (tool_block_mode=True).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import GuardConfig


class ToolGuardError(Exception):
    """Raised when a tool call is blocked by the tool guard."""


class ToolGuard:
    """Checks tool calls for safety before execution."""

    # Patterns for path traversal / dangerous parameters
    PATH_TRAVERSAL = re.compile(r"\.\./|\.\.\\|/etc/passwd|/etc/shadow")
    SQL_INJECTION = re.compile(
        r"\b(DROP|DELETE|TRUNCATE|ALTER|CREATE|INSERT|UPDATE)\s+(TABLE|DATABASE|FROM|INTO)",
        re.IGNORECASE,
    )

    def __init__(self, config: GuardConfig):
        self._block_mode = config.tool_block_mode
        self._blacklist: set[str] = set(config.tool_blacklist)
        self._max_args_size = 10240  # 10KB
        self._call_counts: dict[str, int] = {}  # session_id -> count

    # ---- instance-based call tracking (for per-session rate limiting) ----
    # Note: this is per-instance, not per-session. For proper per-session
    # tracking, use the instance stored on the FastAPI app state.

    def check(self, name: str, arguments: dict, session_id: str = "") -> tuple[bool, str]:
        """Check whether a tool call should be allowed.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.
            session_id: Optional session identifier for rate limiting.

        Returns:
            (allowed, reason) tuple.
        """
        # Check blacklist
        if name in self._blacklist:
            return (False, f"tool blacklisted: {name}")

        # Check argument size
        import json
        try:
            args_json = json.dumps(arguments, ensure_ascii=False)
            if len(args_json.encode("utf-8")) > self._max_args_size:
                return (False, "arguments_too_large")
        except (TypeError, ValueError):
            return (False, "arguments_not_serializable")

        # Check for path traversal in arguments
        for key, value in arguments.items():
            if isinstance(value, str) and self.PATH_TRAVERSAL.search(value):
                return (False, f"path_traversal_detected: param={key}")

        # Check for SQL injection in arguments
        for key, value in arguments.items():
            if isinstance(value, str) and self.SQL_INJECTION.search(value):
                return (False, f"sql_injection_detected: param={key}")

        # Per-session rate limiting (simple counter)
        if session_id:
            count = self._call_counts.get(session_id, 0) + 1
            self._call_counts[session_id] = count
            if count > 100:  # hard limit per session
                return (False, "tool_rate_limit_exceeded")

        return (True, "")

    def reset_session(self, session_id: str) -> None:
        """Reset call count for a session."""
        self._call_counts.pop(session_id, None)