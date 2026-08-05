"""LLM provider abstraction layer.

Provides a unified interface for DeepSeek model provider.
Agnes is used only for image/video generation, not as a text LLM.
"""

from __future__ import annotations

from .deepseek import DeepSeekClient
from .errors import LLMAuthError, LLMError, LLMTransientError, LLMRateLimitError
from .protocol import FunctionCallResult, LLMClient, ToolCall, parse_json_response


def build_llm_client(
    provider: str,
    deepseek_api_key: str = "",
    anthropic_api_key: str = "",  # deprecated, kept for backward compat
    agnes_api_key: str = "",  # kept for image/video generation tools
    model: str = "",
    deepseek_base_url: str = "https://api.deepseek.com",
    agnes_base_url: str = "https://apihub.agnes-ai.com/v1",
    max_tokens: int = 8192,
    timeout_sec: float = 45.0,
    max_message_chars: int = 8000,
) -> LLMClient:
    """Build an LLM client from configuration.

    Only DeepSeek is supported as text LLM provider.
    Agnes API key is accepted for image/video generation tools but not used here.
    """
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