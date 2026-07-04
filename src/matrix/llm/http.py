"""HTTP helpers for LLM provider API calls."""

from __future__ import annotations

import json
import socket
import time
from typing import Any
from urllib import error, request

from .errors import LLMAuthError, LLMError, LLMTransientError

MODEL_PROVIDER_MAX_ATTEMPTS = 2
MODEL_PROVIDER_RETRY_DELAY_SEC = 0.5
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
AUTH_HTTP_CODES = {401, 403}


def post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_sec: float
) -> dict[str, Any]:
    """POST JSON payload to url and return parsed JSON response."""
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