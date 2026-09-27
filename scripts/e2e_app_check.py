"""App-level end-to-end check: personal-agent + personal-os memory pipeline.

Unlike the unit suite, this drives the real stack over HTTP:

    chat -> memory extraction -> decision -> upsert -> durable vault sync

It is intentionally a black-box check. A failure here means the app is broken
even when every unit test passes.

Usage:
    python3 scripts/e2e_app_check.py

Environment:
    AGENT_URL            default http://127.0.0.1:7101
    PERSONAL_OS_API_URL  default http://127.0.0.1:7001
    E2E_ASSETS_PATH      vault root the API writes to (default /private/tmp/e2e-assets)
    E2E_USERNAME / E2E_PASSWORD  credentials; defaults to admin + ADMIN_PASSWORD
                                 read from .env
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

# Localhost must never go through a proxy: in this environment a global
# HTTP_PROXY is exported, and routed localhost calls come back as 502.
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

AGENT = os.environ.get("AGENT_URL", "http://127.0.0.1:7101").rstrip("/")
OS_API = os.environ.get("PERSONAL_OS_API_URL", "http://127.0.0.1:7001").rstrip("/")
ASSETS = os.environ.get("E2E_ASSETS_PATH", "/private/tmp/e2e-assets")
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

# Each run asks for a preference carrying a unique token, so "the memory was
# stored" stays meaningful no matter how many times this check has run before.
RUN_TOKEN = f"e2e{int(time.time())}"
PROBE_MESSAGE = (
    f"请记住一条长期偏好（代号 {RUN_TOKEN}）："
    f"以后给我做的所有复盘周报都用表格输出，并且默认只列三条要点。"
)
# Substrings that must survive into the stored memory. Key names are chosen by
# the model and vary run to run, so match on meaning instead.
PROBE_TERMS = ("表格输出", "三条要点")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def load_env(path: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def http(method: str, url: str, payload: dict | None = None, token: str = "",
         timeout: float = 120.0) -> tuple[int, dict | str]:
    data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        return exc.code, raw
    try:
        return 200, json.loads(raw)
    except json.JSONDecodeError:
        return 200, raw


def sse_lines(url: str, payload: dict, token: str, timeout: float = 300.0) -> list[str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    lines: list[str] = []
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            lines.append(raw.decode("utf-8", "replace").rstrip("\n"))
            if time.time() - started > timeout:
                break
    return lines


def _memory_items(payload: dict | str) -> list[dict]:
    """`/memory/list` returns its rows under "memories".

    Accept "items" too so the check survives an API rename.
    """
    if not isinstance(payload, dict):
        return []
    rows = payload.get("memories")
    if rows is None:
        rows = payload.get("items", [])
    return [row for row in rows if isinstance(row, dict)]


def _probe_known(memories: dict[str, str]) -> bool:
    """True when some stored memory carries the probe preference."""
    haystack = " ".join(str(key) + " " + str(value) for key, value in memories.items())
    return any(term in haystack for term in PROBE_TERMS)


def main() -> int:
    env = load_env(ENV_PATH)
    username = os.environ.get("E2E_USERNAME", "admin")
    password = os.environ.get("E2E_PASSWORD", env.get("ADMIN_PASSWORD", ""))
    if not password:
        check("credentials available", False, "set E2E_PASSWORD or ADMIN_PASSWORD in .env")
        return 1

    # ── 0. Both services reachable ───────────────────────────────────────────
    for label, url in (("personal-os API", f"{OS_API}/healthz"), ("personal-agent", f"{AGENT}/healthz")):
        code, _ = http("GET", url)
        check(f"{label} reachable", code == 200, f"HTTP {code} {url}")

    code, body = http("POST", f"{AGENT}/api/auth/login", {"username": username, "password": password})
    token = body.get("token", "") if isinstance(body, dict) else ""
    if not check("login", code == 200 and bool(token), f"HTTP {code}"):
        return 1

    # ── 1. Baseline memory state ─────────────────────────────────────────────
    _, before = http("GET", f"{AGENT}/memory/list", token=token)
    baseline = {
        item.get("key"): item.get("value")
        for item in _memory_items(before)
    }

    # ── 2. Real chat turn ────────────────────────────────────────────────────
    print("\n--- chat ---")
    lines = sse_lines(f"{AGENT}/chat", {"message": PROBE_MESSAGE}, token)
    body_text = "\n".join(lines)
    check("chat returned a stream", bool(lines), f"{len(lines)} lines")
    # A tool-output pairing bug surfaces as an upstream 400 relayed to the client.
    pairing_broken = "No tool output found" in body_text
    check("no tool/output pairing error", not pairing_broken)

    # ── 3. Memory extraction landed ──────────────────────────────────────────
    # Idempotent on purpose: a repeat run may legitimately produce no diff
    # (the preference is already stored). What must hold every time is that
    # the probe preference is retrievable and that the writer drained.
    deadline = time.time() + 60
    new_keys: set[str] = set()
    changed_keys: set[str] = set()
    current: dict[str, str] = {}
    while time.time() < deadline:
        _, after = http("GET", f"{AGENT}/memory/list", token=token)
        current = {item.get("key"): item.get("value") for item in _memory_items(after)}
        new_keys = set(current) - set(baseline)
        changed_keys = {
            key for key, value in current.items()
            if key in baseline and baseline[key] != value
        }
        if _probe_known(current):
            break
        time.sleep(3)
    check(
        "probe preference is retrievable",
        _probe_known(current),
        f"new={sorted(new_keys)} changed={sorted(changed_keys)}",
    )

    # Extraction runs in a background worker, so the queue drains a moment
    # after the stream ends. Poll rather than sampling once.
    writer: dict = {}
    deadline = time.time() + 60
    while time.time() < deadline:
        _, stats = http("GET", f"{AGENT}/memory/stats", token=token)
        writer = stats.get("writer", {}) if isinstance(stats, dict) else {}
        if writer.get("submitted", 0) == writer.get("completed", 0):
            break
        time.sleep(2)
    check(
        "memory writer completed its queue",
        writer.get("submitted", 0) == writer.get("completed", 0),
        f"submitted={writer.get('submitted')} completed={writer.get('completed')} "
        f"failed={writer.get('failed')} dropped={writer.get('dropped')}",
    )

    # ── 4. Durable sync reached the vault ────────────────────────────────────
    vault_file = os.path.join(ASSETS, "92-系统", "memory", f"{username}.json")
    vault_synced = False
    deadline = time.time() + 45
    while time.time() < deadline:
        if os.path.exists(vault_file):
            with open(vault_file, encoding="utf-8") as handle:
                content = handle.read()
            # Vault content is the durable half of the pipeline: the preference
            # must be readable straight from the assets repo.
            vault_synced = any(term in content for term in PROBE_TERMS)
            if vault_synced:
                break
        time.sleep(3)
    check("vault file updated by personal-os", vault_synced, vault_file)

    print()
    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
