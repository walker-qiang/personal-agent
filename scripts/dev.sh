#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Project Matrix Agent ==="
echo "Starting development server..."

# Set default env vars if not set
export MATRIX_AGENT_ADDR="${MATRIX_AGENT_ADDR:-127.0.0.1:7101}"
export MATRIX_CACHE_PATH="${MATRIX_CACHE_PATH:-var/cache/finance.sqlite}"

# Ensure cache and trace directories exist
mkdir -p var/cache var/agent

if [[ -n "${MATRIX_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$MATRIX_PYTHON_BIN"
elif [[ -x "$PWD/.venv/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Python 3.10+ not found; create .venv with 'uv sync'." >&2
    exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10+ is required; selected interpreter: $PYTHON_BIN" >&2
    exit 1
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m matrix
