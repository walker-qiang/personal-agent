"""LLM provider error hierarchy."""

from __future__ import annotations


class LLMError(Exception):
    """Raised when the configured model provider cannot return a response."""


class LLMTransientError(LLMError):
    """Raised when the model provider failure is likely temporary."""


class LLMAuthError(LLMError):
    """Raised when the configured model provider credentials are rejected."""