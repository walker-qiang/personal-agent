#!/usr/bin/env bash
# App-level end-to-end: start personal-os API + personal-agent, then run the
# black-box memory pipeline check (scripts/e2e_app_check.py).
#
# Why a wrapper: the memory pipeline spans two repositories. personal-agent
# extracts and decides, personal-os owns the durable vault write. Unit tests
# only cover the first half.
#
#   bash scripts/e2e-app.sh
#
# Notes:
#   - Localhost traffic must bypass any exported HTTP_PROXY, otherwise the
#     agent -> personal-os call returns 502. Both services are started with
#     NO_PROXY set.
#   - The vault write commits to a git repo. To avoid pushing personal data
#     during a test, the API defaults to a throwaway copy of personal-assets
#     with no remote. Override with E2E_ASSETS_PATH=/path/to/personal-assets
#     only when you want the real vault written.
set -euo pipefail

AGENT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEM_ROOT="$(cd "$AGENT_ROOT/.." && pwd)"
OS_ROOT="$SYSTEM_ROOT/personal-os"

OS_ADDR="${PERSONAL_OS_API_ADDR:-127.0.0.1:7001}"
AGENT_ADDR="${MATRIX_AGENT_ADDR:-127.0.0.1:7101}"
ASSETS_PATH="${E2E_ASSETS_PATH:-/private/tmp/e2e-assets}"
RUNTIME_DIR="${E2E_RUNTIME_DIR:-/private/tmp/e2e-runtime}"
PYTHON_BIN="${MATRIX_PYTHON_BIN:-/opt/homebrew/bin/python3}"

export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

wait_for_health() { # url label
  for _ in $(seq 1 60); do
    if curl -fsS --noproxy '*' "$1" >/dev/null 2>&1; then
      echo "$2 ready"
      return 0
    fi
    sleep 2
  done
  echo "$2 did not become ready: $1" >&2
  return 1
}

# ── personal-os API ─────────────────────────────────────────────────────────
if curl -fsS --noproxy '*' "http://$OS_ADDR/healthz" >/dev/null 2>&1; then
  echo "personal-os API already listening on $OS_ADDR"
else
  command -v go >/dev/null 2>&1 || export PATH="/usr/local/go/bin:$PATH"
  if [[ ! -d "$ASSETS_PATH" ]]; then
    echo "preparing throwaway vault copy at $ASSETS_PATH"
    mkdir -p "$ASSETS_PATH"
    (cd "$SYSTEM_ROOT/personal-assets" && tar -cf - --exclude=.git .) | (cd "$ASSETS_PATH" && tar -xf -)
    (cd "$ASSETS_PATH" && git init -q && git add -A && git commit -q -m "e2e baseline")
  fi
  mkdir -p "$RUNTIME_DIR"
  echo "building personal-os API"
  (cd "$OS_ROOT" && go build -o /private/tmp/personal-os-api ./apps/api)
  (
    export PERSONAL_OS_API_ADDR="$OS_ADDR"
    export PERSONAL_OS_AGENT_ADDR="$AGENT_ADDR"
    export PERSONAL_OS_AGENT_URL="http://$AGENT_ADDR"
    export PERSONAL_OS_AGENT_MODE=external
    export PERSONAL_ASSETS_PATH="$ASSETS_PATH"
    export PERSONAL_OS_RUNTIME_DIR="$RUNTIME_DIR"
    cd "$OS_ROOT" && nohup /private/tmp/personal-os-api >/private/tmp/personal-os-api.log 2>&1 &
  )
  wait_for_health "http://$OS_ADDR/healthz" "personal-os API"
fi

# ── personal-agent ──────────────────────────────────────────────────────────
if curl -fsS --noproxy '*' "http://$AGENT_ADDR/healthz" >/dev/null 2>&1; then
  echo "personal-agent already listening on $AGENT_ADDR"
else
  (
    export MATRIX_PYTHON_BIN="$PYTHON_BIN"
    export MATRIX_AGENT_ADDR="$AGENT_ADDR"
    export PERSONAL_OS_API_URL="http://$OS_ADDR"
    cd "$AGENT_ROOT" && nohup bash scripts/dev.sh >/private/tmp/personal-agent.log 2>&1 &
  )
  wait_for_health "http://$AGENT_ADDR/healthz" "personal-agent"
fi

# ── Check ───────────────────────────────────────────────────────────────────
export PERSONAL_OS_API_URL="http://$OS_ADDR"
export E2E_ASSETS_PATH="$ASSETS_PATH"
export AGENT_URL="http://$AGENT_ADDR"
"$PYTHON_BIN" "$AGENT_ROOT/scripts/e2e_app_check.py"
