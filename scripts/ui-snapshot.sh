#!/usr/bin/env bash
# UI Snapshot Manager — 管理纯 HTML 前端快照，确保出问题时能快速恢复
#
# Usage:
#   ./scripts/ui-snapshot.sh snapshot   保存当前 static/ 为快照
#   ./scripts/ui-snapshot.sh restore    从快照恢复 static/
#   ./scripts/ui-snapshot.sh status     查看快照与当前版本的差异
#
# 快照目录: scripts/snapshots/ui/
# 目标目录: src/matrix/server/static/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SNAPSHOT_DIR="$PROJECT_ROOT/scripts/snapshots/ui"
STATIC_DIR="$PROJECT_ROOT/src/matrix/server/static"

# 受保护的文件（纯 HTML 前端的核心文件，React build 不应覆盖）
PROTECTED_FILES=("index.html" "marked.min.js")

cmd="${1:-status}"

case "$cmd" in
  snapshot)
    echo "📸 保存 UI 快照..."
    mkdir -p "$SNAPSHOT_DIR"
    for f in "${PROTECTED_FILES[@]}"; do
      if [ -f "$STATIC_DIR/$f" ]; then
        cp "$STATIC_DIR/$f" "$SNAPSHOT_DIR/$f"
        echo "  ✓ $f"
      else
        echo "  ⚠ $f 不存在，跳过"
      fi
    done
    echo "快照已保存到 $SNAPSHOT_DIR"
    ;;

  restore)
    if [ ! -f "$SNAPSHOT_DIR/index.html" ]; then
      echo "❌ 没有快照可恢复，请先运行 snapshot"
      exit 1
    fi
    echo "♻️  从快照恢复 UI..."
    mkdir -p "$STATIC_DIR"
    for f in "${PROTECTED_FILES[@]}"; do
      if [ -f "$SNAPSHOT_DIR/$f" ]; then
        cp "$SNAPSHOT_DIR/$f" "$STATIC_DIR/$f"
        echo "  ✓ $f 已恢复"
      fi
    done
    echo "恢复完成。请重启服务使更改生效。"
    ;;

  status)
    echo "🔍 UI 快照状态检查"
    echo "===================="
    if [ ! -f "$SNAPSHOT_DIR/index.html" ]; then
      echo "⚠️  尚未创建快照，运行 './scripts/ui-snapshot.sh snapshot' 创建"
      exit 0
    fi

    all_ok=true
    for f in "${PROTECTED_FILES[@]}"; do
      if [ ! -f "$STATIC_DIR/$f" ]; then
        echo "❌ $f — 当前 static/ 中缺失！"
        all_ok=false
      elif ! diff -q "$SNAPSHOT_DIR/$f" "$STATIC_DIR/$f" > /dev/null 2>&1; then
        echo "❌ $f — 与快照不一致（可能被 React build 覆盖）"
        all_ok=false
      else
        echo "✅ $f — 一致"
      fi
    done

    if [ "$all_ok" = true ]; then
      echo ""
      echo "✅ 所有核心文件与快照一致，前端状态正常。"
    else
      echo ""
      echo "⚠️  存在不一致，运行 './scripts/ui-snapshot.sh restore' 恢复。"
      exit 1
    fi
    ;;

  *)
    echo "Usage: $0 {snapshot|restore|status}"
    exit 1
    ;;
esac