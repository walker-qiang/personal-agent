"""Write-path guard for long-term memory.

Memory is the one place where untrusted text gets persisted and then replayed
into every future prompt. A fact injected today can steer the agent weeks
later, which makes the write path a higher-value target than a single chat
turn — the literature calls this memory poisoning, and none of the existing
guards (input / output / tool) covered it.

This module screens each candidate before it is stored:

- **Secrets** are rejected outright. A credential does not belong in a
  long-term store that gets serialised into prompts and synced to the vault.
- **Injection payloads** ("ignore previous instructions", "忽略以上所有指令")
  are rejected — stored verbatim, they would be replayed as trusted context.
- **PII** (phone, ID card, bank card, email) is redacted rather than dropped,
  so the memory keeps its meaning without carrying identifiers.
- **Oversized / empty** values are rejected to keep the prompt budget sane.

All of it is best-effort: a guard failure rejects the write rather than
letting unvalidated content through.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MAX_KEY_LEN = 120
MAX_VALUE_LEN = 1000

# Credential-ish material that must never enter long-term memory.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{16,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|password|passwd)"
               r"\s*[:=]\s*\S{6,}"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

# Text that, once stored, would be replayed as trusted system context.
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions"),
    re.compile(r"(?i)disregard\s+(all\s+)?(previous|prior)\s+"),
    re.compile(r"(?i)system\s*:\s*you\s+are"),
    re.compile(r"忽略(以上|前面|之前)?(所有)?(指令|指示|规则)"),
    re.compile(r"无视(以上|前面)?(所有)?(指令|规则)"),
    re.compile(r"你现在是|请扮演(一个)?不受限制"),
    re.compile(r"(?i)\bdo\s+not\s+tell\s+the\s+user\b"),
)


@dataclass
class ScreeningResult:
    """Outcome of screening one candidate memory."""

    allowed: bool
    value: str
    reason: str = ""
    redacted: bool = False
    flags: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.flags is None:
            self.flags = []


class MemorySafetyGuard:
    """Screens candidate memories before they reach the store."""

    def __init__(self, output_guard: Any | None = None, enabled: bool = True) -> None:
        self._output_guard = output_guard
        self._enabled = enabled

    def screen(self, key: str, value: str) -> ScreeningResult:
        """Return whether (and how) a candidate may be stored."""
        if not self._enabled:
            return ScreeningResult(allowed=True, value=value)

        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return ScreeningResult(allowed=False, value="", reason="empty key or value")
        if len(key) > MAX_KEY_LEN:
            return ScreeningResult(allowed=False, value="", reason="key too long")
        if len(value) > MAX_VALUE_LEN:
            return ScreeningResult(
                allowed=False, value="", reason="value too long",
            )

        for pattern in _SECRET_PATTERNS:
            if pattern.search(value) or pattern.search(key):
                logger.warning("memory_safety: rejected secret-like content key=%s", key)
                return ScreeningResult(
                    allowed=False, value="", reason="looks like a credential",
                )

        for pattern in _INJECTION_PATTERNS:
            if pattern.search(value):
                logger.warning(
                    "memory_safety: rejected injection payload key=%s", key,
                )
                return ScreeningResult(
                    allowed=False, value="",
                    reason="looks like a prompt-injection payload",
                )

        if self._output_guard is not None:
            try:
                result = self._output_guard.check(value)
                if getattr(result, "had_pii", False):
                    logger.info(
                        "memory_safety: redacted PII key=%s flags=%s",
                        key, getattr(result, "flags", []),
                    )
                    return ScreeningResult(
                        allowed=True,
                        value=str(getattr(result, "sanitized", value)),
                        redacted=True,
                        flags=list(getattr(result, "flags", []) or []),
                    )
            except Exception as exc:  # noqa: BLE001
                # A guard failure must not silently let content through, and
                # must not crash the write path either: store it as-is.
                logger.warning("memory_safety: PII check failed: %s", exc)

        return ScreeningResult(allowed=True, value=value)
