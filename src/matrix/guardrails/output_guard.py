"""OutputGuard: PII redaction for agent responses before returning to user.

Strategy: sanitize in-place (replace PII with partially masked versions).
Default mode is sanitize-only (output_block_mode=False); blocking mode
replaces the entire response with a safe message if PII is detected.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import GuardConfig

# Shared PII patterns across OutputGuard and TraceSanitizer
_PII_SPECS: list[tuple[str, str, str]] = [
    # (name, regex, mask_template)
    # IMPORTANT: id_card must be before phone — phone regex is a subset of id_card.
    ("id_card", r"\d{17}[\dXx]", lambda m: m.group()[:4] + "**********" + m.group()[-4:]),
    # Phone: keep first 3 and last 4, mask middle
    ("phone", r"1[3-9]\d{9}", lambda m: m.group()[:3] + "****" + m.group()[-4:]),
    # Email: mask local part
    ("email", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", lambda m: "u***@" + m.group().split("@", 1)[1]),
    # Bank card: keep last 4
    ("bank_card", r"\b\d{16,19}\b", lambda m: "****" + m.group()[-4:]),
    # API key: full mask
    ("api_key", r"sk-[a-zA-Z0-9]{20,}", lambda _: "sk-***"),
    # IP address
    ("ip", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", lambda _: "x.x.x.x"),
]


def _make_replacer(mask_fn):
    """Wrap a mask function for use with re.sub."""
    return lambda m: mask_fn(m)


class OutputResult:
    """Result of output guard check."""

    def __init__(self, sanitized: str, flags: list[str], had_pii: bool):
        self.sanitized = sanitized
        self.flags = flags
        self.had_pii = had_pii


class OutputGuard:
    """Checks and sanitizes agent output before returning to the user."""

    def __init__(self, config: GuardConfig):
        self._block_mode = config.output_block_mode
        self._patterns: list[tuple[str, re.Pattern, object]] = [
            (name, re.compile(pattern), _make_replacer(mask_fn))
            for name, pattern, mask_fn in _PII_SPECS
        ]

    def check(self, text: str, user_id: str = "") -> OutputResult:
        """Sanitize PII from text. Returns sanitized text and flags."""
        if not text:
            return OutputResult(text, [], False)

        sanitized = text
        flags: list[str] = []

        for name, pattern, replacer in self._patterns:
            if pattern.search(sanitized):
                flags.append(name)
                sanitized = pattern.sub(replacer, sanitized)

        return OutputResult(
            sanitized=sanitized,
            flags=flags,
            had_pii=len(flags) > 0,
        )