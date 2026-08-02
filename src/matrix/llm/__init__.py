"""LLM provider abstraction layer.

Provides a unified interface for DeepSeek and Agnes model providers.
"""

from __future__ import annotations

from .deepseek import DeepSeekClient
from .errors import LLMAuthError, LLMError, LLMTransientError, LLMRateLimitError
from .protocol import FunctionCallResult, LLMClient, ToolCall, parse_json_response


def build_llm_client(
    provider: str,
    deepseek_api_key: str = "",
    anthropic_api_key: str = "",  # deprecated, kept for backward compat
    agnes_api_key: str = "",
    model: str = "",
    deepseek_base_url: str = "https://api.deepseek.com",
    agnes_base_url: str = "https://apihub.agnes-ai.com/v1",
    max_tokens: int = 8192,
    timeout_sec: float = 45.0,
    max_message_chars: int = 8000,
) -> LLMClient:
    """Build an LLM client from configuration."""
    if provider == "agnes":
        return DeepSeekClient(
            api_key=agnes_api_key,
            model=model or "agnes-2.0-flash",
            base_url=agnes_base_url,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            max_message_chars=max_message_chars,
        )
    return DeepSeekClient(
        api_key=deepseek_api_key,
        model=model or "deepseek-v4-flash",
        base_url=deepseek_base_url,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_message_chars=max_message_chars,
    )


__all__ = [
    "DeepSeekClient",
    "FunctionCallResult",
    "LLMClient",
    "LLMError",
    "LLMTransientError",
    "LLMAuthError",
    "LLMRateLimitError",
    "ToolCall",
    "parse_json_response",
    "build_llm_client",
]