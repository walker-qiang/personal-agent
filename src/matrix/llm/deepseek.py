"""DeepSeek API client."""

from __future__ import annotations

from .errors import LLMError
from .http import post_json_with_retry


class DeepSeekClient:
    """LLM client for DeepSeek chat completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 4096,
        timeout_sec: float = 45.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout_sec = timeout_sec

    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        data = post_json_with_retry(url, payload, headers, self.timeout_sec)
        try:
            return str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as err:
            raise LLMError("DeepSeek response did not include message content") from err