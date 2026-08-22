"""Controlled durable writes through the personal-os Vault API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class VaultWriteError(RuntimeError):
    """Raised when personal-os rejects or cannot complete a Vault write."""


def sync_memory_profile(user_id: str, profile: dict[str, str]) -> dict[str, Any]:
    encoded = urllib.parse.quote(user_id, safe="")
    return _request("PUT", f"/api/vault/memory/{encoded}", {"profile": profile})


def mutate_skill(
    operation: str,
    *,
    domain: str,
    skill_name: str,
    filename: str = "",
    content: str = "",
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/vault/skills",
        {
            "operation": operation,
            "domain": domain,
            "skill_name": skill_name,
            "filename": filename,
            "content": content,
        },
    )


def _request(method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    base_url = os.environ.get(
        "PERSONAL_OS_API_URL", "http://127.0.0.1:7001",
    ).rstrip("/")
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "personal-agent/vault-client",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=90.0) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            detail = str(payload.get("error") or payload.get("detail") or "")
        except Exception:
            detail = ""
        raise VaultWriteError(
            detail or f"personal-os vault write failed: HTTP {exc.code}",
        ) from exc
    except Exception as exc:
        raise VaultWriteError(f"personal-os vault write unavailable: {exc}") from exc
    if not isinstance(body, dict) or body.get("ok") is not True:
        raise VaultWriteError(str(body.get("error", "personal-os vault write failed")))
    return body
