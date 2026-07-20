"""InputGuard: prompt injection / data exfiltration / role confusion detection.

Detects malicious user inputs before they reach the agent. Default mode
is warn-only (block_mode=False); switch to block mode after tuning.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import GuardConfig


class InputResult:
    """Result of input guard check."""

    def __init__(self, allowed: bool, reason: str, flags: list[str]):
        self.allowed = allowed
        self.reason = reason
        self.flags = flags


class InputGuard:
    """Checks user messages for prompt injection, data exfiltration,
    role confusion, XSS, and excessive length."""

    # (regex, category, severity)
    # severity: "block" = always block, "warn" = warn-only
    PATTERNS: list[tuple[str, str, str]] = [
        # Prompt override attempts
        (
            r"(ignore|forget|disregard)\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|messages?)",
            "prompt_override",
            "block",
        ),
        # Role confusion / jailbreak
        (
            r"you\s+are\s+(now\s+)?(DAN|jailbroken|unrestricted|unfiltered|without\s+restrictions?)",
            "role_confusion",
            "block",
        ),
        (
            r"(pretend|act|pose)\s+(as\s+if\s+)?(you\s+(are|have)\s+)?(an?\s+)?(no\s+restrictions?|unlimited|unrestricted)",
            "role_confusion",
            "block",
        ),
        # Data exfiltration
        (
            r"(output|print|dump|show|reveal|display)\s+(all\s+)?(system\s+)?(prompts?|messages?|memory|instructions?|history)",
            "data_exfiltration",
            "block",
        ),
        # XSS / script injection
        (
            r"<script[^>]*>",
            "xss_attempt",
            "block",
        ),
        (
            r"javascript\s*:",
            "xss_attempt",
            "block",
        ),
        # Data URI injection
        (
            r"data:text/html",
            "xss_attempt",
            "block",
        ),
        # Markdown image with data URI
        (
            r"!\[.*\]\(data:",
            "xss_attempt",
            "block",
        ),
    ]

    def __init__(self, config: GuardConfig):
        self._block_mode = config.input_block_mode
        self._max_length = config.max_message_len
        self._compiled: list[tuple[re.Pattern, str, str]] = [
            (re.compile(pattern, re.IGNORECASE), category, severity)
            for pattern, category, severity in self.PATTERNS
        ]

    def check(self, message: str, user_id: str = "") -> InputResult:
        """Check a user message for security issues.

        Args:
            message: The user's input message.
            user_id: The user identifier (for logging).

        Returns:
            InputResult with allowed status and reason.
        """
        if not message:
            return InputResult(allowed=True, reason="", flags=[])

        # Check message length
        if len(message.encode("utf-8")) > self._max_length:
            return InputResult(
                allowed=False,
                reason="message_too_long",
                flags=["message_too_long"],
            )

        # Check patterns
        flags: list[str] = []
        for pattern, category, severity in self._compiled:
            if pattern.search(message):
                flags.append(category)
                if severity == "block" and self._block_mode:
                    return InputResult(
                        allowed=False,
                        reason=f"detected: {category}",
                        flags=flags,
                    )

        return InputResult(
            allowed=True,
            reason="",
            flags=flags,
        )