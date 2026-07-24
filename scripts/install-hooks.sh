#!/bin/bash
# Install git hooks from scripts/hooks/ into .git/hooks/
# Usage: bash scripts/install-hooks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOKS_SRC="$SCRIPT_DIR/hooks"
HOOKS_DST="$SCRIPT_DIR/../.git/hooks"

if [ ! -d "$HOOKS_DST" ]; then
    echo "Error: .git/hooks/ not found at $HOOKS_DST"
    echo "Are you running this from the personal-agent repository?"
    exit 1
fi

for hook in "$HOOKS_SRC"/*; do
    [ -f "$hook" ] || continue
    name=$(basename "$hook")
    target="$HOOKS_DST/$name"
    cp "$hook" "$target"
    chmod +x "$target"
    echo "Installed: $name → $target"
done

echo ""
echo "✓ Git hooks installed:"
echo "  pre-push    — 单元测试 + skill 校验 + 变更检测建议 (Layer 1)"
echo "  post-commit — agent/skill 变更后后台自动评估 (Layer 2/3)"
echo ""
echo "  To bypass pre-push: git push --no-verify"
