"""Anthropic Claude API client."""

from __future__ import annotations

from .errors import LLMError
from .http import post_json_with_retry


class AnthropicClient:
    """LLM client for Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 8192,
        timeout_sec: float = 45.0,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = post_json_with_retry(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers,
            self.timeout_sec,
        )
        try:
            chunks = data["content"]
            return "".join(
                str(chunk.get("text", "")) for chunk in chunks if chunk.get("type") == "text"
            )
        except (KeyError, TypeError) as err:
            raise LLMError("Anthropic response did not include text content") from err