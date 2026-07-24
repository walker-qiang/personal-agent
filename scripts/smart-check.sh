#!/bin/bash
# smart-check.sh — 智能变更检测 + 自动评估
#
# 工作原理:
#   1. 对比当前 HEAD 与上次评估基线记录的 commit
#   2. 检测哪些类型的文件发生了变更
#   3. 根据变更类型自动执行对应层级的评估
#
# 变更类型 → 评估层级映射:
#   Agent 代码 (src/matrix/)        → Layer 2 回归评估
#   Skill/知识库 (技能/)             → Layer 2 回归评估
#   Prompt/LLM 逻辑                  → Layer 2 + Layer 3 质量评估
#   仅测试/文档                      → 跳过 (Layer 1 已覆盖)
#
# 用法:
#   bash scripts/smart-check.sh              # 自动检测并运行
#   bash scripts/smart-check.sh --force      # 强制运行全部评估
#   bash scripts/smart-check.sh --regression # 仅运行回归评估
#   bash scripts/smart-check.sh --quality    # 仅运行质量评估

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TRACKER_FILE=".eval-tracker"
FORCED_MODE=""
RUN_REGRESSION=false
RUN_QUALITY=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --force)      FORCED_MODE="force" ;;
        --regression) FORCED_MODE="regression" ;;
        --quality)    FORCED_MODE="quality" ;;
    esac
done

# Get current commit
CURRENT_COMMIT=$(git rev-parse --short HEAD)

# ---- Determine what changed ----

if [ "$FORCED_MODE" = "force" ]; then
    echo "🔄 强制模式 — 运行全部评估"
    RUN_REGRESSION=true
    RUN_QUALITY=true
elif [ "$FORCED_MODE" = "regression" ]; then
    RUN_REGRESSION=true
elif [ "$FORCED_MODE" = "quality" ]; then
    RUN_QUALITY=true
else
    # Smart detection: compare against last eval commit
    LAST_EVAL_COMMIT=""
    if [ -f "$TRACKER_FILE" ]; then
        LAST_EVAL_COMMIT=$(cat "$TRACKER_FILE" | grep "^commit:" | cut -d: -f2- | tr -d ' ')
    fi

    if [ -z "$LAST_EVAL_COMMIT" ]; then
        echo "📌 首次运行 — 执行全部评估"
        RUN_REGRESSION=true
        RUN_QUALITY=true
    elif [ "$LAST_EVAL_COMMIT" = "$CURRENT_COMMIT" ]; then
        echo "✓ 自上次评估后无新提交，无需检查。"
        echo "  强制运行: bash scripts/smart-check.sh --force"
        exit 0
    else
        # Get changed files since last eval
        CHANGED_FILES=$(git diff --name-only "$LAST_EVAL_COMMIT" "$CURRENT_COMMIT" 2>/dev/null || echo "")

        if [ -z "$CHANGED_FILES" ]; then
            echo "✓ 自上次评估后无文件变更。"
            exit 0
        fi

        AGENT_CHANGED=false
        SKILL_CHANGED=false
        PROMPT_CHANGED=false
        ONLY_INFRA=true

        while IFS= read -r file; do
            case "$file" in
                src/matrix/agent/*|src/matrix/orchestration/*|src/matrix/chat/*|src/matrix/tools/*)
                    AGENT_CHANGED=true; ONLY_INFRA=false ;;
                ../personal-assets/技能/*|skills/*)
                    SKILL_CHANGED=true; ONLY_INFRA=false ;;
                src/matrix/agent/commander.py|src/matrix/agent/domain_agents/*|src/matrix/orchestration/nodes/_helpers.py)
                    PROMPT_CHANGED=true; ONLY_INFRA=false ;;
                tests/*|docs/*|*.md|scripts/*)
                    # Infrastructure/docs — don't trigger eval
                    ;;
                *)
                    ONLY_INFRA=false ;;
            esac
        done <<< "$CHANGED_FILES"

        if [ "$ONLY_INFRA" = true ]; then
            echo "✓ 仅基础设施/文档变更，跳过评估。"
            echo "$CURRENT_COMMIT" > /dev/null
            echo "commit:$CURRENT_COMMIT" > "$TRACKER_FILE"
            exit 0
        fi

        if [ "$AGENT_CHANGED" = true ] || [ "$SKILL_CHANGED" = true ]; then
            RUN_REGRESSION=true
        fi

        if [ "$PROMPT_CHANGED" = true ]; then
            RUN_REGRESSION=true
            RUN_QUALITY=true
        fi
    fi
fi

# ---- Execute evaluations ----

START_TIME=$SECONDS

if [ "$RUN_REGRESSION" = true ]; then
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Layer 2: 回归评估"
    echo "════════════════════════════════════════════════════"
    PYTHONUNBUFFERED=1 python -u -m matrix.evaluation.cli regression 2>&1
    REGRESSION_EXIT=$?
else
    REGRESSION_EXIT=0
fi

if [ "$RUN_QUALITY" = true ]; then
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Layer 3: 质量评估 (LLM-as-Judge)"
    echo "════════════════════════════════════════════════════"
    PYTHONUNBUFFERED=1 python -u -m matrix.evaluation.cli quality 2>&1
    QUALITY_EXIT=$?
else
    QUALITY_EXIT=0
fi

ELAPSED=$((SECONDS - START_TIME))
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ---- Determine overall status ----
OVERALL_EXIT=0
REGRESSION_STATUS="skipped"
QUALITY_STATUS="skipped"

if [ "$RUN_REGRESSION" = true ]; then
    if [ $REGRESSION_EXIT -eq 0 ]; then
        REGRESSION_STATUS="passed"
    else
        REGRESSION_STATUS="failed"
        OVERALL_EXIT=1
    fi
fi

if [ "$RUN_QUALITY" = true ]; then
    if [ $QUALITY_EXIT -eq 0 ]; then
        QUALITY_STATUS="passed"
    else
        QUALITY_STATUS="failed"
        OVERALL_EXIT=1
    fi
fi

# ---- Update tracker ----
echo "commit:$CURRENT_COMMIT" > "$TRACKER_FILE"

# ---- Write status file (for pre-push hook to check) ----
STATUS_FILE="$REPO_ROOT/.eval-status"
if [ $OVERALL_EXIT -eq 0 ]; then
    echo "status:passed" > "$STATUS_FILE"
else
    echo "status:failed" > "$STATUS_FILE"
fi
echo "timestamp:$TIMESTAMP" >> "$STATUS_FILE"
echo "commit:$CURRENT_COMMIT" >> "$STATUS_FILE"
echo "regression:$REGRESSION_STATUS" >> "$STATUS_FILE"
echo "quality:$QUALITY_STATUS" >> "$STATUS_FILE"
echo "elapsed:${ELAPSED}s" >> "$STATUS_FILE"

# ---- Summary (console) ----
echo ""
echo "════════════════════════════════════════════════════"
echo "  评估完成 (耗时 ${ELAPSED}s)"
echo "════════════════════════════════════════════════════"

if [ "$RUN_REGRESSION" = true ]; then
    if [ $REGRESSION_EXIT -eq 0 ]; then
        echo "  Layer 2 回归: ✓ 无回归"
    else
        echo "  Layer 2 回归: ✗ 检测到回归"
    fi
fi

if [ "$RUN_QUALITY" = true ]; then
    if [ $QUALITY_EXIT -eq 0 ]; then
        echo "  Layer 3 质量: ✓ 无质量下降"
    else
        echo "  Layer 3 质量: ✗ 检测到质量下降"
    fi
fi

echo ""
echo "  如需更新基线:"
echo "    python -m matrix.evaluation.cli update-baseline regression"
echo "    python -m matrix.evaluation.cli update-baseline quality"
echo "════════════════════════════════════════════════════"

# ---- macOS notification ----
if [ $OVERALL_EXIT -eq 0 ]; then
    NOTIFY_TITLE="Agent 评估通过"
    NOTIFY_MSG="commit ${CURRENT_COMMIT} · ${ELAPSED}s"
else
    FAILED_PARTS=""
    [ "$REGRESSION_STATUS" = "failed" ] && FAILED_PARTS="回归"
    [ "$QUALITY_STATUS" = "failed" ] && FAILED_PARTS="${FAILED_PARTS} 质量"
    NOTIFY_TITLE="Agent 评估发现问题"
    NOTIFY_MSG="commit ${CURRENT_COMMIT} · ${FAILED_PARTS} · cat .eval-last-run.log"
fi

if command -v osascript &>/dev/null; then
    osascript -e "display notification \"${NOTIFY_MSG}\" with title \"${NOTIFY_TITLE}\"" 2>/dev/null || true
fi

exit $OVERALL_EXIT
