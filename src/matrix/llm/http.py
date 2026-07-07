"""HTTP helpers for LLM provider API calls."""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Iterator
from urllib import error, request

from ..rate_limiter import TokenBucketRateLimiter
from .errors import LLMAuthError, LLMError, LLMTransientError, LLMRateLimitError

MODEL_PROVIDER_MAX_ATTEMPTS = 2
MODEL_PROVIDER_RETRY_DELAY_SEC = 0.5
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
AUTH_HTTP_CODES = {401, 403}
STREAM_READ_CHUNK = 4096

# Module-level rate limiter — can be replaced per provider
_rate_limiter: TokenBucketRateLimiter | None = None


def set_rate_limiter(limiter: TokenBucketRateLimiter | None) -> None:
    """Set the global rate limiter for LLM API calls."""
    global _rate_limiter
    _rate_limiter = limiter


def _acquire_rate_limit() -> None:
    """Acquire a rate limit token before making an API call."""
    if _rate_limiter is not None:
        if not _rate_limiter.acquire(timeout=30.0):
            raise LLMRateLimitError("rate limit timeout — too many concurrent requests")


def post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: float
) -> dict[str, Any]:
    """POST JSON payload to url and return parsed JSON response."""
    _acquire_rate_limit()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except (TimeoutError, socket.timeout) as err:
        raise LLMTransientError(f"model provider timed out after {timeout_sec:g}s") from err
    except error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        message = f"model provider returned {err.code}: {detail[:500]}"
        if err.code in AUTH_HTTP_CODES:
            raise LLMAuthError(
                f"model provider authentication failed ({err.code}); check API key"
            ) from err
        if err.code in TRANSIENT_HTTP_CODES:
            raise LLMTransientError(message) from err
        raise LLMError(message) from err
    except error.URLError as err:
        if isinstance(err.reason, (TimeoutError, socket.timeout)):
            raise LLMTransientError(f"model provider timed out after {timeout_sec:g}s") from err
        raise LLMTransientError(f"model provider unavailable: {err.reason}") from err
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise LLMError(f"model provider returned invalid json: {err}") from err
    if not isinstance(data, dict):
        raise LLMError("model provider returned non-object response")
    return data


def post_json_with_retry(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: float
) -> dict[str, Any]:
    """POST JSON with retry on transient errors."""
    for attempt in range(1, MODEL_PROVIDER_MAX_ATTEMPTS + 1):
        try:
            return post_json(url, payload, headers, timeout_sec)
        except LLMTransientError as err:
            if attempt >= MODEL_PROVIDER_MAX_ATTEMPTS:
                raise LLMTransientError(
                    f"model provider transient failure after {attempt} attempts: {err}"
                ) from err
            time.sleep(MODEL_PROVIDER_RETRY_DELAY_SEC)
    raise LLMTransientError("model provider transient failure")


def _read_sse_lines(resp: Any, timeout_sec: float) -> Iterator[str]:
    """Read SSE lines from a streaming HTTP response.

    Yields each non-empty 'data: ...' line body (the JSON string).
    Handles both 'data: {...}' and 'data: [DONE]'.
    """
    deadline = time.monotonic() + timeout_sec
    buf = b""
    while time.monotonic() < deadline:
        chunk = resp.read(STREAM_READ_CHUNK)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            # SSE format: "data: {...}" or "data: [DONE]"
            if text.startswith("data: "):
                data_str = text[6:]
                if data_str == "[DONE]":
                    return
                yield data_str
    # Drain remaining buffer
    if buf:
        text = buf.decode("utf-8", errors="replace").strip()
        if text.startswith("data: ") and text[6:] != "[DONE]":
            yield text[6:]


def post_json_stream(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: float
) -> Iterator[str]:
    """POST JSON payload with streaming SSE response. Yields raw 'data:' JSON strings."""
    _acquire_rate_limit()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            yield from _read_sse_lines(resp, timeout_sec)
    except (TimeoutError, socket.timeout) as err:
        raise LLMTransientError(f"model provider timed out after {timeout_sec:g}s") from err
    except error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        message = f"model provider returned {err.code}: {detail[:500]}"
        if err.code in AUTH_HTTP_CODES:
            raise LLMAuthError(
                f"model provider authentication failed ({err.code}); check API key"
            ) from err
        if err.code in TRANSIENT_HTTP_CODES:
            raise LLMTransientError(message) from err
        raise LLMError(message) from err
    except error.URLError as err:
        if isinstance(err.reason, (TimeoutError, socket.timeout)):
            raise LLMTransientError(f"model provider timed out after {timeout_sec:g}s") from err
        raise LLMTransientError(f"model provider unavailable: {err.reason}") from err