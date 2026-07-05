"""LLM provider abstraction layer.

Provides a unified interface for DeepSeek and Anthropic model providers.
"""

from __future__ import annotations

from .anthropic import AnthropicClient
from .deepseek import DeepSeekClient
from .errors import LLMAuthError, LLMError, LLMTransientError
from .protocol import LLMClient, ToolCall


def build_llm_client(
    provider: str,
    deepseek_api_key: str = "",
    anthropic_api_key: str = "",
    model: str = "",
    deepseek_base_url: str = "https://api.deepseek.com",
    max_tokens: int = 4096,
    timeout_sec: float = 45.0,
) -> LLMClient:
    """Build an LLM client from configuration."""
    if provider == "anthropic":
        return AnthropicClient(
            api_key=anthropic_api_key,
            model=model or "claude-3-5-sonnet-latest",
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
    return DeepSeekClient(
        api_key=deepseek_api_key,
        model=model or "deepseek-chat",
        base_url=deepseek_base_url,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
    )


__all__ = [
    "AnthropicClient",
    "DeepSeekClient",
    "LLMClient",
    "LLMError",
    "LLMTransientError",
    "LLMAuthError",
    "ToolCall",
    "build_llm_client",
]