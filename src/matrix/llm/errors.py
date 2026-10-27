"""LLM provider error hierarchy."""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """Raised when the configured model provider cannot return a response."""


class LLMResponseFormatError(LLMError):
    """A completed request whose raw output is retained for application validation."""

    def __init__(
        self, message: str, *, draft: str, usage: dict[str, Any], finish_reason: str,
        recovered_object: dict[str, Any] | None = None,
        normalization: str = "",
    ) -> None:
        super().__init__(message)
        self.draft = draft
        self.usage = usage
        self.finish_reason = finish_reason
        self.recovered_object = recovered_object
        self.normalization = normalization


class LLMTransientError(LLMError):
    """Raised when the model provider failure is likely temporary."""


class LLMAuthError(LLMError):
    """Raised when the configured model provider credentials are rejected."""


class LLMRateLimitError(LLMError):
    """Raised when the client-side rate limiter blocks a request."""
