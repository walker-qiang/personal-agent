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

python3 -m matrix