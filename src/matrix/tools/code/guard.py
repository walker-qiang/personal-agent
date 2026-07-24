"""CodeGuard: pre-execution safety checks for code execution tools.

Defense-in-depth layer — the sandbox itself provides process/filesystem/resource
isolation. This guard catches obviously dangerous code before spawning a subprocess.
"""

from __future__ import annotations

import re
from typing import Any


class CodeGuardError(Exception):
    """Raised when code is blocked by the code guard."""


class CodeGuard:
    """Checks Python code for dangerous patterns before execution."""

    # Patterns that are always blocked
    FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\bos\.system\s*\("),
        re.compile(r"\bsubprocess\."),
        re.compile(r"\bshutil\.rmtree\s*\("),
        re.compile(r"\bos\.remove\s*\("),
        re.compile(r"\bos\.unlink\s*\("),
        re.compile(r"\bopen\s*\(\s*['\"]/(?:etc|var|usr|bin|sbin|root|home)"),
        re.compile(r"\b__import__\s*\(\s*['\"](?:ctypes|cffi)"),
        re.compile(r"\bexec\s*\(\s*['\"]"),
        re.compile(r"\beval\s*\(\s*['\"]"),
        re.compile(r"\bos\.popen\s*\("),
        re.compile(r"\bpopen2\."),
        re.compile(r"\bcommands\."),
    ]

    MAX_CODE_SIZE = 10240  # 10KB

    def check(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Check if code execution should be allowed.

        Args:
            tool_name: Name of the tool being called.
            arguments: Tool arguments dict.

        Returns:
            (allowed, reason) tuple. reason is empty string when allowed.
        """
        if not tool_name.startswith("code."):
            return (True, "")

        code = arguments.get("code", "")
        if not isinstance(code, str):
            return (False, "code must be a string")

        # Size check
        if len(code.encode("utf-8")) > self.MAX_CODE_SIZE:
            return (
                False,
                f"code_too_large: {len(code)} bytes (max {self.MAX_CODE_SIZE})",
            )

        # Pattern check
        for pattern in self.FORBIDDEN_PATTERNS:
            match = pattern.search(code)
            if match:
                return (False, f"forbidden_pattern: {match.group()!r}")

        return (True, "")
