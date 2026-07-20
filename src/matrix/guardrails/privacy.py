"""TraceSanitizer: PII redaction before trace persistence.

Strips personally identifiable information from trace arguments and results
before they are written to the SQLite trace store. Uses a more aggressive
strategy than OutputGuard — replaces with [REDACTED] instead of partial masking.
"""

from __future__ import annotations

import re
from typing import Any


class TraceSanitizer:
    """Redacts PII from trace data before persistence."""

    # Patterns: (name, regex, replacement)
    PII_PATTERNS: list[tuple[str, str, str]] = [
        # IMPORTANT: id_card must be before phone — phone regex is a subset of id_card.
        # ID card numbers (Chinese)
        ("id_card", r"\d{17}[\dXx]", "[REDACTED:id_card]"),
        # Phone numbers (Chinese mobile)
        ("phone", r"1[3-9]\d{9}", "[REDACTED:phone]"),
        # API keys
        ("api_key", r"sk-[a-zA-Z0-9]{20,}", "[REDACTED:api_key]"),
        # Email addresses
        ("email", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED:email]"),
        # Bank card numbers (16-19 digits)
        ("bank_card", r"\b\d{16,19}\b", "[REDACTED:bank_card]"),
        # IP addresses
        ("ip", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[REDACTED:ip]"),
        # JWT tokens
        ("jwt", r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}", "[REDACTED:jwt]"),
    ]

    def sanitize(self, value: Any) -> Any:
        """Sanitize a value recursively, redacting PII in strings."""
        if value is None:
            return None
        if isinstance(value, str):
            return self._sanitize_str(value)
        if isinstance(value, dict):
            return {str(k): self.sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.sanitize(item) for item in value]
        return value

    def _sanitize_str(self, text: str) -> str:
        for _name, pattern, replacement in self.PII_PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text