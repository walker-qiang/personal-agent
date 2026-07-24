# 质量保障方案：Eval-Driven Development with Baseline Regression

> 目标：确保每次更新 agent、skill、知识库后，质量至少不下降。
> 约束：个人用户，通过 git 持久化代码，无 CI/CD 服务器，使用免费 LLM 模型（Agnes）。
>
> 业界对标：本方案基于 **Eval-Driven Development（评估驱动开发）** 范式，
> 对标 Hamel Husain 三级评估模型（Unit Tests → Model/Human Eval → A/B Test）
> 和 Anthropic Capability/Regression 二分法。
> "三层质量门禁"是上述业界实践的个人用户适配版本。

## 实现状态

| 层级 | 状态 | 实现内容 |
|------|------|----------|
| Layer 1 | ✅ 已完成 | pre-push hook + check-skills CLI + install-hooks.sh |
| Layer 2 | ✅ 已完成 | 20 条评估数据集 + regression CLI + 基线对比逻辑 + 27 个单元测试 |
| Layer 3 | ✅ 已完成 | quality CLI + LLM-as-Judge + 质量基线对比逻辑 |
| 自动触发 | ✅ 已完成 | post-commit hook + smart-check.sh + 变更类型检测 |

**测试覆盖**：553 passed, 4 skipped, 0 failed

## 现状分析

### 已有基础

| 能力 | 现状 | 评估 |
|------|------|------|
| 单元测试 | 553 个测试，pytest 框架 | 覆盖良好，已集成 pre-push hook |
| 评估框架 | EvalCase → EvalRunner → Evaluator → Metrics → Reporter | 完整可用 |
| 评估数据集 | eval_dataset.json（20 条 case） | 已扩展，覆盖 6 大场景 |
| Skill 测试 | test_skills.py 验证加载和匹配 | 已实现 check-skills 通用校验 |
| Git hooks | pre-push hook 已安装并启用 | Layer 1 已完成 |
| 基线管理 | baseline.py + test_baseline.py（27 个测试） | 已完成回归和质量基线对比 |

### 核心缺口（已解决）

1. ~~**无自动触发**~~：✅ pre-push hook 已安装，每次 push 自动运行
2. ~~**无基线对比**~~：✅ baseline.py 实现回归和质量基线对比
3. ~~**用例不足**~~：✅ 已扩展到 20 条，覆盖 6 大场景
4. ~~**skill 变更无校验**~~：✅ check-skills 命令校验所有 skill 格式

## 方案设计：三层质量门禁

```
代码变更
  │
  ▼
┌─────────────────────────────────────────────┐
│ Layer 1: Fast Gate (pre-push git hook)      │
│ ─ pytest 单元测试 + skill 格式校验           │
│ ─ ~15s, 免费                                 │
│ ─ 自动触发, 阻断 push                        │
└──────────────────────┬──────────────────────┘
                       │ push 通过
                       ▼
┌─────────────────────────────────────────────┐
│ Layer 2: Regression Eval (手动 CLI)          │
│ ─ 确定性评估, 对比基线 pass rate              │
│ ─ ~3min, 少量 LLM token                      │
│ ─ 手动触发, 发布前必跑                        │
└──────────────────────┬──────────────────────┘
                       │ 无回归
                       ▼
┌─────────────────────────────────────────────┐
│ Layer 3: Quality Assessment (手动 CLI)       │
│ ─ LLM-as-Judge, 对比基线质量分数             │
│ ─ ~8min, 较多 LLM token                      │
│ ─ 手动触发, 重大变更后跑                      │
└─────────────────────────────────────────────┘
```

### 设计原则

- **成本递增、频率递减**：Layer 1 每次推送都跑（~15s），Layer 3 只在重大变更后跑
- **基线驱动**：Layer 2/3 的核心不是"绝对分数"，而是"相对基线的变化"
- **个人友好**：不依赖 CI 服务器，git hooks 本地执行；CLI 一条命令搞定
- **渐进采用**：三层相互独立，可先上 Layer 1，后续再加 Layer 2/3
- **零 AI 成本**：所有层使用已配置的免费模型（Agnes），不产生额外费用

---

## Layer 1: Fast Gate（pre-push git hook）

### 触发方式
`git push` 时自动触发，失败则阻断 push。

### 执行内容

```
1. pytest tests/ -x -q                            # 单元测试, fail-fast
2. python -m matrix.evaluation.cli check-skills   # skill 格式校验
```

### 文件结构

```
personal-agent/
├── scripts/
│   ├── hooks/
│   │   └── pre-push                          # git hook 脚本
│   └── install-hooks.sh                      # 一键安装 hooks
```

### pre-push 脚本逻辑

```bash
#!/bin/bash
# pre-push hook: fast quality gate
# 仅检查将要推送的 commit 涉及的文件变更

set -e

# 1. 单元测试 (fail-fast, 30s 超时)
echo "▶ Running unit tests..."
python -m pytest tests/ -x -q --timeout=30 2>&1 | tail -5
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "✗ Unit tests failed. Push blocked."
    exit 1
fi

# 2. Skill 格式校验
echo "▶ Validating skills..."
python -m matrix.evaluation.cli check-skills 2>&1
if [ $? -ne 0 ]; then
    echo "✗ Skill validation failed. Push blocked."
    exit 1
fi

echo "✓ All checks passed."
exit 0
```

### install-hooks.sh

```bash
#!/bin/bash
# 将 scripts/hooks/ 下的 hook 安装到 .git/hooks/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOKS_DIR="$SCRIPT_DIR/../.git/hooks"

for hook in "$SCRIPT_DIR/hooks/"*; do
    name=$(basename "$hook")
    target="$HOOKS_DIR/$name"
    cp "$hook" "$target"
    chmod +x "$target"
    echo "Installed: $name"
done
```

### Skill 校验逻辑

`python -m matrix.evaluation.cli check-skills` 执行：

1. 从 `AgentConfig` 读取 `skills_base_dir`（默认指向 `personal-assets/技能/`，可通过 `MATRIX_SKILLS_BASE_DIR` 环境变量覆盖）
2. 对每个含 `SKILL.md` 的目录，调用 `SkillDefinition.from_dir()` 解析
3. 检查项：
   - YAML frontmatter 能否解析
   - title 和 description 非空
   - workflow 步骤格式合法（tool 名称非空、参数可解析）
   - 引用的 knowledge/scripts 文件存在
4. 任一校验失败 → exit 1，输出错误详情

> **注意**：skill 的 durable source of truth 在 `personal-assets/技能/`，
> `personal-agent/skills/` 目录下的副本已删除，避免同步漂移。

### 退出行为

- **通过**：push 正常进行
- **失败**：push 被阻断，输出失败原因
- **跳过**：`git push --no-verify` 可跳过（紧急情况用，不推荐）

---

## Layer 2: Regression Evaluation（手动 CLI）

### 触发方式
手动执行 `python -m matrix.evaluation.cli regression`，发布前必跑。

### 执行内容

```
1. 加载评估数据集 (tests/baselines/eval_dataset.json, ~20 条 case)
2. 启动 ChatService (需 .env 配置 LLM API key)
3. 逐条运行 EvalRunner + DeterministicEvaluator
4. 加载基线 (tests/baselines/regression_baseline.json)
5. 对比当前结果与基线
6. 输出对比报告 + 退出码 (0=无回归, 1=有回归)
```

### 基线文件格式

`tests/baselines/regression_baseline.json`:

```json
{
  "version": "2026-07-24-v1",
  "created_at": "2026-07-24T10:00:00Z",
  "git_commit": "abc1234",
  "summary": {
    "total": 20,
    "passed": 19,
    "failed": 1,
    "pass_rate": 0.95
  },
  "case_results": {
    "smoke_greeting": { "passed": true },
    "smoke_holdings": { "passed": true },
    "regress_finance_001": { "passed": false, "reason": "known_issue" }
  }
}
```

### 回归判定规则

| 场景 | 判定 |
|------|------|
| 基线 pass → 当前 pass | 正常 |
| 基线 pass → 当前 fail | **回归** (exit 1) |
| 基线 fail → 当前 pass | 改善 (不阻断) |
| 基线 fail → 当前 fail | 持续问题 (警告, 不阻断) |
| 新增 case 无基线 | 仅记录, 不阻断 |

### 基线更新

当确认当前行为正确（修了 bug 或改善了回答）后：

```bash
python -m matrix.evaluation.cli update-baseline regression
```

将当前结果写入基线文件，git commit 后成为新基线。

### 评估数据集设计

已从 5 条扩展到 20 条，按维度覆盖：

| 维度 | Case 数 | 示例 |
|------|---------|------|
| 基础对话 | 3 | 问候、能力询问、感谢 |
| 投资查询 | 6 | 持仓查询、资产查找、快照历史、最近快照、桶配置、组合风险 |
| 搜索工具 | 4 | web搜索、新闻搜索、天气查询、股票行情 |
| 媒体生成 | 1 | 图片生成 |
| 多步骤任务 | 2 | 持仓+新闻、组合分析+建议 |
| 边界场景 | 4 | 无效股票、英文输入、超出范围、乱码输入 |

每条 case 配置：
- `case_id`: 唯一标识
- `user_input`: 用户输入
- `expected.outcome`: 期望结果类型
- `expected.must_include`: 回答中必须包含的关键词
- `expected.must_not_include`: 回答中不能出现的词
- `expected.required_tools`: 必须调用的工具
- `expected.forbidden_tools`: 禁止调用的工具
- `tags`: 标签（用于分维度统计）
- `difficulty`: easy / medium / hard
- `risk`: low / medium / high

---

## Layer 3: Quality Assessment（手动 CLI）

### 触发方式
手动执行 `python -m matrix.evaluation.cli quality`，重大变更后跑。

### 适用场景

- 修改 system prompt
- 更换 LLM 模型
- 调整 ReAct 循环逻辑
- 大规模重构

### 执行内容

```
1. 加载评估数据集 (复用 Layer 2 的 dataset)
2. 启动 ChatService
3. 逐条运行 EvalRunner + DeterministicEvaluator + LLMEvaluator
4. 加载质量基线 (tests/baselines/quality_baseline.json)
5. 对比当前质量分数与基线
6. 输出质量报告 + 退出码
```

### 质量基线格式

`tests/baselines/quality_baseline.json`:

```json
{
  "version": "2026-07-24-v1",
  "created_at": "2026-07-24T10:00:00Z",
  "git_commit": "abc1234",
  "summary": {
    "total": 20,
    "pass_rate": 0.90,
    "avg_quality_score": 0.82,
    "dimensions": {
      "accuracy": 0.85,
      "completeness": 0.80,
      "relevance": 0.88,
      "conciseness": 0.76
    }
  },
  "case_scores": {
    "smoke_greeting": { "overall": 0.95, "dimensions": {...} },
    "smoke_holdings": { "overall": 0.78, "dimensions": {...} }
  }
}
```

### 质量回归判定

| 指标 | 阈值 | 判定 |
|------|------|------|
| 单 case overall 降幅 | > 0.15 | 该 case 质量回归 |
| 平均 overall 降幅 | > 0.05 | 整体质量回归 |
| 任一维度平均分降幅 | > 0.10 | 维度质量回归 |
| 新增 case | 无基线 | 仅记录 |

任一回归条件满足 → exit 1，输出详细对比。

### 基线更新

```bash
python -m matrix.evaluation.cli update-baseline quality
```

---

## CLI 工具设计

### 入口

`python -m matrix.evaluation.cli <command> [options]`

### 命令列表

```
 Commands:
   check-skills          校验所有 skill 定义格式 (Layer 1)
   regression            运行回归评估, 对比基线 (Layer 2)
   quality               运行质量评估, 对比基线 (Layer 3)
   update-baseline <type>  更新基线文件 (regression | quality)
   list-cases            列出当前数据集所有 case
   diff-baseline         对比两次运行结果

 Options:
   --dataset <path>      指定数据集文件 (默认 tests/baselines/eval_dataset.json)
   --baseline <path>     指定基线文件
   --no-baseline         跳过基线对比, 仅运行
   --format <fmt>        输出格式: console (默认) | json
   --cases <ids>         只运行指定 case (逗号分隔)
   --tags <tags>         只运行指定标签的 case
```

### 文件结构

```
personal-agent/
├── src/matrix/evaluation/
│   ├── cli.py                 # CLI 入口 + 命令实现
│   └── baseline.py            # 基线加载、对比、更新逻辑
├── tests/
│   └── baselines/
│       ├── eval_dataset.json  # 评估数据集 (~20 条)
│       ├── regression_baseline.json  # Layer 2 基线
│       └── quality_baseline.json     # Layer 3 基线
├── scripts/
│   ├── hooks/
│   │   └── pre-push
│   └── install-hooks.sh
```

---

## 自动触发机制

> 解决问题：agent 代码或知识库变更后，如何自动执行检查，而不需要手动记住跑评估。

### 架构

```
git commit
  │
  ▼
┌─────────────────────────────────────────────┐
│ post-commit hook (非阻塞)                    │
│ ─ 检测变更文件类型                            │
│ ─ 如果涉及 agent/skill → 后台启动 smart-check │
│ ─ 结果写入 .eval-last-run.log                 │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ smart-check.sh (智能评估调度)                 │
│ ─ 对比 .eval-tracker 记录的上次评估 commit     │
│ ─ 检测变更类型:                               │
│   • agent 代码 → Layer 2 回归评估             │
│   • skill/知识库 → Layer 2 回归评估           │
│   • prompt/LLM → Layer 2 + Layer 3           │
│   • 仅文档/测试 → 跳过                        │
│ ─ 执行对应评估, 更新 .eval-tracker             │
└─────────────────────────────────────────────┘

git push
  │
  ▼
┌─────────────────────────────────────────────┐
│ pre-push hook (阻塞式)                       │
│ ─ pytest 单元测试 (fail-fast)                │
│ ─ check-skills 格式校验                       │
│ ─ 变更检测 + 评估建议 (不阻塞, 仅提示)         │
└─────────────────────────────────────────────┘
```

### 变更类型 → 评估映射

| 变更类型 | 文件路径匹配 | 触发评估 |
|----------|-------------|----------|
| Agent 代码 | `src/matrix/agent/*`, `src/matrix/orchestration/*`, `src/matrix/chat/*`, `src/matrix/tools/*` | Layer 2 回归 |
| Skill/知识库 | `personal-assets/技能/*`, `skills/*` | Layer 2 回归 |
| Prompt/LLM 逻辑 | `commander.py`, `domain_agents/*`, `_helpers.py` | Layer 2 + Layer 3 |
| 仅测试/文档 | `tests/*`, `docs/*`, `*.md`, `scripts/*` | 跳过 |

### 文件结构

```
personal-agent/
├── scripts/
│   ├── hooks/
│   │   ├── pre-push              # Layer 1 门禁 + 变更检测建议
│   │   └── post-commit           # 自动触发后台评估
│   ├── smart-check.sh            # 智能评估调度器
│   └── install-hooks.sh          # 一键安装 hooks
├── .eval-tracker                 # 上次评估的 commit 记录 (gitignored)
├── .eval-last-run.log            # 最近一次评估日志 (gitignored)
```

### smart-check.sh 用法

```bash
# 自动检测变更并运行对应评估
bash scripts/smart-check.sh

# 强制运行全部评估 (Layer 2 + Layer 3)
bash scripts/smart-check.sh --force

# 仅运行回归评估
bash scripts/smart-check.sh --regression

# 仅运行质量评估
bash scripts/smart-check.sh --quality
```

### 工作流说明

1. **git commit** — post-commit hook 检测变更类型，如果涉及 agent/skill，后台启动 smart-check.sh
2. **smart-check.sh** — 对比上次评估的 commit，确定变更范围，执行对应层级评估
3. **评估完成** — 结果写入 `.eval-last-run.log`，tracker 更新为当前 commit
4. **git push** — pre-push hook 运行 Layer 1 快速门禁，并打印评估建议
5. **查看结果** — `cat .eval-last-run.log | tail -30` 或 `bash scripts/smart-check.sh` 查看最近评估

---

## 日常工作流

### 场景 1：日常小改动（修 bug、调参数）

```bash
# 1. 修改代码
vim src/matrix/chat/_service.py

# 2. 提交
git add -A && git commit -m "fix: 修复流式输出截断问题"
# → post-commit hook 自动检测变更
# → 如果涉及 agent 代码, 后台自动启动 Layer 2 回归评估
# → 结果写入 .eval-last-run.log

# 3. 推送 (自动触发 Layer 1)
git push
# → pytest 自动运行
# → skill 校验自动运行
# → 全部通过才允许 push
# → 如果有未评估的变更, 打印评估建议

# 4. (可选) 查看后台评估结果
cat .eval-last-run.log | tail -30
```

### 场景 2：发布前检查

```bash
# 1. 确保代码已推送
git push

# 2. 运行回归评估 (Layer 2)
python -m matrix.evaluation.cli regression
# → 20 条 case 逐条运行
# → 对比基线, 检查有无回归
# → 无回归 → 可以发布

# 3. (可选) 如果改了 prompt 或模型, 跑质量评估
python -m matrix.evaluation.cli quality
```

### 场景 3：确认改善后更新基线

```bash
# 修复了一个 case, 确认回答正确
python -m matrix.evaluation.cli regression --no-baseline
# → 看到该 case 现在 pass

# 更新基线
python -m matrix.evaluation.cli update-baseline regression
git add tests/baselines/regression_baseline.json
git commit -m "chore: update regression baseline after fix"
```

### 场景 4：紧急推送跳过门禁

```bash
git push --no-verify
# 跳过 Layer 1, 事后补跑评估
```

---

## 实施计划

### 阶段 1：Layer 1 快速门禁 ✅ 已完成

| 步骤 | 文件 | 状态 |
|------|------|------|
| 创建 pre-push hook 脚本 | `scripts/hooks/pre-push` | ✅ |
| 创建 install-hooks.sh | `scripts/install-hooks.sh` | ✅ |
| 实现 check-skills 命令 | `src/matrix/evaluation/cli.py` | ✅ |
| 实现 baseline.py 基础结构 | `src/matrix/evaluation/baseline.py` | ✅ |
| 测试 hook 是否正常工作 | — | ✅ |

### 阶段 2：Layer 2 回归评估 ✅ 已完成

| 步骤 | 文件 | 状态 |
|------|------|------|
| 扩展评估数据集到 20 条 | `tests/baselines/eval_dataset.json` | ✅ |
| 实现 regression 命令 | `src/matrix/evaluation/cli.py` | ✅ |
| 实现基线对比逻辑 | `src/matrix/evaluation/baseline.py` | ✅ |
| 添加基线对比单元测试 | `tests/test_baseline.py`（27 个测试） | ✅ |
| 首次运行, 生成初始基线 | `tests/baselines/regression_baseline.json` | 待手动执行 |

### 阶段 3：Layer 3 质量评估 ✅ 已完成

| 步骤 | 文件 | 状态 |
|------|------|------|
| 实现 quality 命令 | `src/matrix/evaluation/cli.py` | ✅ |
| 实现质量基线对比 | `src/matrix/evaluation/baseline.py` | ✅ |
| 添加质量基线单元测试 | `tests/test_baseline.py` | ✅ |
| 首次运行, 生成质量基线 | `tests/baselines/quality_baseline.json` | 待手动执行 |

### 阶段 4：自动触发机制 ✅ 已完成

| 步骤 | 文件 | 状态 |
|------|------|------|
| 创建 smart-check.sh 智能调度器 | `scripts/smart-check.sh` | ✅ |
| 创建 post-commit hook | `scripts/hooks/post-commit` | ✅ |
| 增强 pre-push hook 变更检测 | `scripts/hooks/pre-push` | ✅ |
| 更新 install-hooks.sh | `scripts/install-hooks.sh` | ✅ |
| 添加 .eval-tracker / .eval-last-run.log 到 .gitignore | `.gitignore` | ✅ |
| 安装并验证 hooks | `.git/hooks/` | ✅ |

---

## 成本估算

| 层级 | 频率 | 耗时 | AI 成本 |
|------|------|------|---------|
| Layer 1 | 每次 push (~10次/周) | ~15s | ¥0（无 LLM 调用） |
| Layer 2 | 发布前 (~2次/周) | ~3min | ¥0（使用 Agnes 免费模型） |
| Layer 3 | 重大变更 (~2次/月) | ~8min | ¥0（使用 Agnes 免费模型） |

所有层均使用已配置的免费模型，不产生额外费用。唯一成本是运行时间。

---

## 与现有代码的集成点

| 现有模块 | 集成方式 | 改动量 |
|----------|----------|--------|
| `evaluation/runner.py` | CLI 直接调用 `EvalRunner.run()` | 无改动 |
| `evaluation/evaluators/deterministic.py` | Layer 2 使用 | 无改动 |
| `evaluation/evaluators/llm_judge.py` | Layer 3 使用 | 无改动 |
| `evaluation/metrics.py` | CLI 调用 `compute_metrics()` | 无改动 |
| `evaluation/reporter.py` | CLI 使用 `Reporter` 输出 | 无改动 |
| `evaluation/case.py` | `EvalCase.from_dict()` 加载数据集 | 无改动 |
| `skills/loader.py` | `check-skills` 调用 `load_skills()` | 无改动 |
| `chat/_service.py` | Layer 2/3 创建 `ChatService` 实例 | 无改动 |

**核心设计原则：不修改任何现有业务逻辑，仅新增 CLI 和基线管理模块。**
