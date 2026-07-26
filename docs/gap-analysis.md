# 差距分析与优化设计

> 基于 2026 年 7 月 AI Agent 行业格局，对比 Matrix 与 Claude Code、DeerFlow、Devin 等顶级 Agent 的差距，给出优化方案设计。

---

## 一、P0：规划能力 — Plan-and-Execute

### 1.1 现状

当前 Commander 的规划逻辑（`orchestration/nodes/commander.py → commander_plan_node`）：

```
用户消息 → LLM 生成 delegation_plan → 单步直接 ReAct / 多步 fan-out 委派
```

**问题**：

1. **一次规划，终身不改**：Plan 生成后不再修正。第 2 步返回了意外结果，第 3 步计划已经错了，但系统继续执行。
2. **无显式 DAG**：`delegation_plan` 是扁平列表 `[{step, agent_id, task}]`，没有依赖关系。所有多步计划通过 LangGraph `Send` API 并行 fan-out，但 Send 之间没有顺序保证——如果 step 2 依赖 step 1 的结果，当前架构无法表达。
3. **规划与执行未分离**：Commander 用同一个 LLM 做规划和最终汇总。业界最佳实践是规划器用便宜模型（如 DeepSeek Flash），执行器用强模型。
4. **Token 效率低**：每一步都需要 LLM 重新推理"现在该做什么"，而不是按计划执行。Plan-and-Execute 模式可将 LLM 调用从 ~10 次降到 1-2 次 [$TRAE_REF](https://juejin.cn/post/7663362294992764982)。

### 1.2 业界参考

**DeerFlow 2.0 的 Supervisor-SubAgent 模式**：Supervisor 先分解目标为结构化任务计划，再根据任务类型路由到对应 SubAgent，并行执行无依赖的子任务，最后汇总结果 [$TRAE_REF](https://blog.csdn.net/m0_59235245/article/details/159696326)。

**BabyAGI 的重规划门**：每 K 步或遇到超阈值输出时，触发重规划门，让规划器决定是否修订剩余步骤 [$TRAE_REF](https://juejin.cn/post/7663362294992764982)。

**LangGraph 的 Plan-and-Execute**：`plan_and_execute` agent 将任务分解为 DAG 子任务，执行器按拓扑排序执行，准确率从 ReAct 的 85% 提升到 92%。

### 1.3 优化设计

#### 1.3.1 新增节点：`plan_node`

在 `commander_plan_node` 之后、`delegate` 之前插入独立的规划节点。

**输入**：`user_message`、`messages`（历史）、`working_memory`

**输出**：
```python
execution_plan = {
    "dag": [
        {
            "step_id": 1,
            "agent_id": "investment-analyst",
            "task": "查询当前持仓",
            "depends_on": [],           # 前置步骤 ID 列表
            "skill_name": "",
            "expected_output": "持仓列表及市值",
        },
        {
            "step_id": 2,
            "agent_id": "investment-analyst",
            "task": "分析持仓异动",
            "depends_on": [1],          # 依赖 step 1 的结果
            "skill_name": "",
            "expected_output": "异动标的及原因",
        },
        {
            "step_id": 3,
            "agent_id": "investment-analyst",
            "task": "生成调仓建议",
            "depends_on": [2],
            "skill_name": "",
            "expected_output": "具体调仓方案",
        },
    ],
    "replan_gate": {
        "enabled": True,
        "check_every": 1,               # 每步执行后检查
        "max_total_steps": 10,
    },
    "plan_type": "dag",                 # "linear" | "dag" | "single"
}
```

**规划 LLM 分离**：plan_node 使用 `pipeline_llm`（更便宜的模型），temperature=0.1。

#### 1.3.2 新增节点：`replan_node`

在每步 delegate 执行后检查是否需要重规划。

**触发条件**（任一满足即触发）：
1. 步骤输出与 `expected_output` 严重不匹配（LLM 判断）
2. 工具调用全部失败
3. 步骤返回的 `error` 字段非空

**重规划逻辑**：
```
replan_node:
  1. 收集已完成步骤的输出
  2. 收集当前步骤的失败信息
  3. LLM 评估：剩余步骤是否仍然有效？
  4. 如果计划需要修订 → 生成新的 execution_plan（仅含未完成步骤）
  5. 如果计划仍然有效 → 继续执行
```

#### 1.3.3 图结构调整

```
__start__
    │
    ▼
commander_plan
    │
    ▼
plan_node (NEW)          ← 生成 execution_plan（DAG）
    │
    ├── plan_type == "single" → react_prepare → react_loop → aggregate
    │
    └── plan_type in ("dag", "linear") →
        │
        ▼
    execute_next_step      ← 拓扑排序，取出下一批可执行步骤
        │
        ├── 无依赖步骤 → [Send("delegate") × N 并行]
        │
        └── 有依赖步骤 → Send("delegate", 1) 顺序执行
            │
            ▼
        replan_node (NEW)   ← 每步执行后检查
            │
            ├── 需要重规划 → plan_node（循环）
            │
            └── 继续 → execute_next_step（循环）
                │
                └── 所有步骤完成 → aggregate
```

#### 1.3.4 新增 `AgentState` 字段

```python
# 替换现有 delegation_plan
execution_plan: dict | None           # {dag, replan_gate, plan_type}
completed_steps: Annotated[list, operator.add]  # 已完成步骤的输出
step_outputs: dict[str, Any]          # step_id → 步骤输出
```

---

## 二、P0：记忆系统 — 失败日志 + 分层记忆

### 2.1 现状

Matrix 已有较高水平的记忆基础：

| 记忆层 | 实现 | 状态 |
|--------|------|------|
| 工作记忆 | `state.working_memory` (pinned + insights) | ✅ 已有 |
| 情景记忆 | SQLite `messages` 表 + `get_history()` | ✅ 已有 |
| 语义记忆 | RAG (ChromaDB + BM25) | ✅ 已有 |
| 程序记忆 | Skills (YAML) | ✅ 已有 |
| 记忆演化 | `MemoryEvolution` 4 阶段管线 | ✅ 已有 |

**缺失**：
1. **失败日志**：Agent 不会从错误中学习，重复踩坑。当前工具调用失败后只记录 `tool_results` 中的 `error` 字段，但不会结构化存储失败原因和修复方案。
2. **分层记忆**：所有记忆平坦存储，没有按范围（全局/项目/会话）分层。新会话加载全部记忆，旧会话不会自动加载相关上下文。
3. **执行记忆**：不会记住"上次怎么解决这个问题的"，每次都是从头来。

### 2.2 业界参考

**DeerFlow 双层内存**：工作记忆（当前上下文）+ 归档记忆（完整历史，语义检索取回），使其能处理数小时的长时程任务 [$TRAE_REF](https://blog.csdn.net/m0_59235245/article/details/159696326)。

**失败日志模式**：Agent 用结构化格式记录每次失败（步骤、失败类型、根因、修复方案、任务上下文），下次遇到类似情况时主动检索历史失败教训 [$TRAE_REF](https://juejin.cn/post/7663362294992764982)。

**分层保留**：全局知识（Tier 1）→ 项目上下文（Tier 2）→ 会话状态（Tier 3），新会话逐层加载 [$TRAE_REF](https://juejin.cn/post/7663362294992764982)。

### 2.3 优化设计

#### 2.3.1 失败日志（Failure Journal）

**新增数据表**：
```sql
CREATE TABLE failure_log (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    step_description TEXT NOT NULL,    -- 当时在做什么
    failure_type TEXT NOT NULL,        -- tool_error | llm_error | timeout | hallucination
    tool_name TEXT,                    -- 哪个工具失败
    error_message TEXT,                -- 原始错误信息
    root_cause TEXT,                   -- LLM 分析的根因
    remediation TEXT,                  -- 修复方案
    task_context TEXT,                 -- 任务上下文（用户问题摘要）
    created_at TEXT NOT NULL
);
```

**写入时机**：在 `react_tool_node` 中，每次工具调用返回 `error` 时，异步写入 failure_log。

**检索时机**：在 `react_prepare_node` 中，将用户问题与 `failure_log` 做语义匹配（embedding 相似度 ≥ 0.7），将匹配的教训注入 system prompt：

```
## 历史教训（请避免重复以下错误）
- 上次调用 finance_query 时参数格式错误：应使用 "symbol=HK.00700" 而非 "00700.HK"
- 上次生成报告时 news_search 返回空结果：应先用 web_search 搜索关键词，再用 news_search
```

**实现路径**：
1. `store.py` 新增 `save_failure_log()` / `get_relevant_failures()` 方法
2. `_helpers.py` 新增 `_inject_failure_lessons()` 函数
3. `react_prepare_node` 中调用注入

#### 2.3.2 分层记忆（Hierarchical Retention）

**当前**：`get_profile_formatted()` 一次性返回所有用户记忆。

**改为三层**：
```python
# Tier 1: 全局知识（政策、偏好）
# 存于 user_profile 表，memory_type="policy" 或 "preference"
# 每次会话必加载，永不过滤

# Tier 2: 项目上下文（当前任务相关）
# 存于 user_profile 表，memory_type="project"
# 通过 embedding 匹配用户问题后加载

# Tier 3: 会话状态（当前会话的临时信息）
# 存于 working_memory，会话结束即清
```

**`get_profile_formatted()` 改造**：

```python
def get_profile_formatted(user_id: str, current_query: str = "") -> str:
    # Tier 1: 全局知识（必加载）
    policies = get_policies(user_id)
    preferences = get_preferences_decayed(user_id)
    
    # Tier 2: 项目上下文（按需加载）
    project_memories = []
    if current_query:
        all_memories = get_all_memories(user_id)
        for m in all_memories:
            if m["memory_type"] == "project":
                # 用 embedding 相似度判断是否相关
                if cosine_similarity(embed(current_query), embed(m["key"])) > 0.6:
                    project_memories.append(m)
    
    # 组装输出
    parts = []
    parts.append(format_policies(policies))
    parts.append(format_preferences(preferences))
    if project_memories:
        parts.append(format_project_context(project_memories))
    return "\n\n".join(parts)
```

#### 2.3.3 执行记忆（暂缓）

执行记忆（记住"上次怎么解决的"）需要更复杂的机制：记录成功执行路径 → 冻结为可复用资产 → 下次同类任务直接调用。这与 Skills 系统有重叠，建议在 Skills 系统成熟后再做。

---

## 三、P1：多 Agent 协作 — 扇出-聚合

### 3.1 现状

当前多 Agent 路径：
```
commander_plan → [Send("delegate") × N] → aggregate
```

所有 delegate 通过 LangGraph Send API 并行执行，`agent_results` 通过 `operator.add` 自动合并。

**问题**：
1. **无真正的并行扇出**：Send API 在 LangGraph 中是"逻辑并行"——当所有 Send 的目标节点相同时，框架处理为并行。但当前架构中，每个 Send 传递不同的 `current_step`，实际上是"不同步骤的串行描述"被并行执行，语义不清。
2. **无 Agent 间通信**：delegate 之间完全隔离，无法相互传递中间结果。
3. **无 Supervisor 调度**：缺少中央协调器动态分配任务，只能按预设计划执行。

### 3.2 业界参考

Anthropic 的 Orchestrator-Workers 多 Agent 系统比单 Agent 高出 90.2%，但消耗约 15 倍 token。关键权衡：任务必须真正可分离 [$TRAE_REF](https://juejin.cn/post/7663362294992764982)。

### 3.3 优化设计

#### 3.3.1 明确三种多 Agent 模式

**模式 A：独立扇出（Fan-out）**
```
用户问题：对比腾讯、阿里、美团的估值
  │
  ├── Worker A: 分析腾讯 (send("delegate", {agent_id: "investment-analyst", task: "分析腾讯"}))
  ├── Worker B: 分析阿里 (send("delegate", {agent_id: "investment-analyst", task: "分析阿里"}))
  └── Worker C: 分析美团 (send("delegate", {agent_id: "investment-analyst", task: "分析美团"}))
  │
  └── aggregate: 对比三家公司
```
适用：子任务完全独立，无依赖。

**模式 B：流水线（Pipeline）**
```
用户问题：先查持仓，再分析异动，再给建议
  │
  ├── Step 1: 查询持仓 → 输出持仓列表
  ├── Step 2: 分析异动（依赖 Step 1）→ 输出异动分析
  └── Step 3: 生成建议（依赖 Step 2）→ 输出调仓建议
```
适用：步骤间有强依赖，需顺序执行。

**模式 C：层级委派（Hierarchical）**
```
用户问题：全面分析我的投资组合
  │
  ├── Supervisor: 分解为 3 个子任务
  │   ├── Worker A: 持仓分析 (investment-analyst)
  │   ├── Worker B: 市场分析 (investment-analyst)
  │   └── Worker C: 建议生成 (investment-analyst)
  │
  └── Supervisor: 汇总 → 输出
```
适用：复杂任务，需要 Supervisor 动态协调。

#### 3.3.2 实现方案

**Plan 格式扩展**（与 P0 的 execution_plan 合并）：

```python
execution_plan = {
    "mode": "fan_out",  # "fan_out" | "pipeline" | "hierarchical" | "single"
    "dag": [...],       # 步骤定义
    "replan_gate": {...},
}
```

**图调整**：当 `mode == "fan_out"` 时，所有无依赖步骤并行 Send；当 `mode == "pipeline"` 时，按依赖顺序串行 Execute。

**注意**：真正的层级委派（Hierarchical）需要 Supervisor 在运行时动态分配任务，工作量大，建议作为 Phase 2。

---

## 四、P1：反思回路 — 工具锚定 Critic

### 4.1 现状

已有 Reflexion 模块（`orchestration/nodes/commander.py → reflection_node`）：
- LLM 评估回答质量
- 不合格时生成 self-reflection 并重试
- 但批评来源是**同一 LLM 的自我评价**，容易确认偏差

### 4.2 业界参考

Huang et al. (ICLR 2024) 证明：**LLM 没有外部反馈时无法可靠地自我纠正**。自恋式自我批评往往是确认偏差。工具锚定批评（测试套件、Linter、计算器）才是正道 [$TRAE_REF](https://juejin.cn/post/7663362294992764982)。

### 4.3 优化设计

#### 4.3.1 现有反幻觉验证的增强

当前 `anti_hallucination.py` 已有 `verify_all_claims()` 做事实性验证。将此机制**从"输出后验证"提前到"反思环节的输入"**：

```python
# reflection_node 改造
def reflection_node(state, config):
    # Step 1: 反幻觉验证（已有）
    vr = verify_all_claims(state.final_answer, state.tool_results)
    
    # Step 2: 如果验证结果有 contradicted 或 unverified
    if vr.contradicted > 0 or vr.unverified > 0:
        # 将具体问题注入反思 prompt
        issues = vr.get_issues()  # "声称'腾讯市值5万亿'但工具结果中无此数据"
        reflection_prompt = build_reflection_with_evidence(issues)
        # ... 触发 reflexion retry
    
    # Step 3: 如果全部 verified，跳过反思
    if vr.contradicted == 0 and vr.unverified == 0:
        return {"skip_reflection": True}
```

#### 4.3.2 新增确定性验证器

```python
class DeterministicVerifier:
    """工具锚定验证：不依赖 LLM 的确定性检查"""
    
    def verify_numeric(self, claim_value, tool_results):
        """检查声称的数值是否在工具结果中出现"""
        # 提取所有工具结果中的数值
        # 检查 claim_value 是否在 ±1% 范围内
    
    def verify_code(self, claim, code_output):
        """检查代码执行结果是否与声称一致"""
        # 如果声称"代码执行成功"，检查 stderr 是否为空
        # 如果声称"代码返回了 X"，检查 stdout 是否包含 X
    
    def verify_search(self, claim, search_results):
        """检查搜索结果的 URL 和标题是否与声称一致"""
```

#### 4.3.3 反思 Prompt 增强

当前：`REFLECTION_PROMPT` 和 `REFLEXION_PROMPT` 是纯 LLM 评估。

改为：注入反幻觉验证的**具体问题**作为"批评锚点"：

```
## 反幻觉验证发现的问题
以下声称在工具结果中找不到证据：
- "腾讯市值5.2万亿港元" → 工具结果中未找到此数值，找到的是 5.1 万亿
- "阿里巴巴营收增长12%" → 工具结果中未找到此数据

请根据以上问题修正回答，确保每项数据都有工具结果支撑。
```

---

## 五、P2：可视化调试 — Trace 回放

### 5.1 现状

- 已有 OTel + JSONL Trace 记录
- 纯 HTML 前端有 Trace 面板（文本列表 + 详情）
- 但无法可视化回放决策过程

### 5.2 业界参考

LangSmith 提供完整的执行追踪能力：可视图回放每次 Agent 决策、工具调用的参数与结果、状态变化时间线。工程师可以精确定位到哪一步的哪个决策出了问题 [$TRAE_REF](https://blog.csdn.net/m0_59235245/article/details/159696326)。

### 5.3 优化设计

**短期**（HTML 前端增强）：
- 在 Trace 详情面板中增加时间线视图（CSS 实现）
- 用颜色区分：绿色=成功、红色=失败、黄色=重试、蓝色=思考
- 点击展开工具调用的完整 JSON 参数和结果

**长期**（独立可视化）：
- 考虑使用 LangSmith 或自建可视化面板
- 工作量较大，建议放在 P2

---

## 六、P2：成本/Token 监控

### 6.1 现状

- 无任何 Token 使用量统计
- 无成本核算
- 预算检查（`budget.py`）只做比例控制，不记录历史

### 6.2 优化设计

**新增数据表**：
```sql
CREATE TABLE token_usage (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    purpose TEXT,  -- "planning" | "execution" | "reflection" | "aggregation"
    created_at TEXT NOT NULL
);
```

**写入时机**：在 `_service.py` 的 `stream_chat()` 中，每次 LLM 调用完成后，异步写入。

**前端展示**：在纯 HTML 前端底部状态栏增加 Token 计数器，在用户菜单中增加"用量统计"页面。

**成本换算**：在 `config.py` 中维护各模型的价格表，自动换算为人民币。

---

## 七、实施路线图

### Phase 1（1-2 周）：P0 项

| 任务 | 文件改动 | 预估 |
|------|----------|------|
| 新增 `plan_node` | `orchestration/nodes/plan.py` (新) | 2 天 |
| 新增 `replan_node` | `orchestration/nodes/replan.py` (新) | 1 天 |
| 图结构调整 | `orchestration/graph.py` | 1 天 |
| 新增 failure_log 表 | `store.py` | 0.5 天 |
| 失败日志写入 | `orchestration/nodes/react.py` | 0.5 天 |
| 失败日志检索注入 | `orchestration/nodes/_helpers.py` | 1 天 |
| 分层记忆 | `store.py` + `_service.py` | 1 天 |

### Phase 2（2-4 周）：P1 项

| 任务 | 说明 |
|------|------|
| 扇出-聚合模式 | 基于 execution_plan 的 mode 字段实现 |
| 工具锚定 Critic | 增强反幻觉验证在反思环节的应用 |
| 确定性验证器 | 数值/代码/搜索的确定性验证 |

### Phase 3（长期）：P2 项

| 任务 | 说明 |
|------|------|
| 可视化 Trace 回放 | 时间线视图 + 决策回放 |
| Token 成本仪表板 | 用量统计 + 成本核算 |
| 层级委派 | Supervisor 动态任务分配 |

---

## 八、风险与注意事项

1. **Plan-and-Execute 的过度规划风险**：简单问题（如"今天天气"）不需要 DAG 分解。`plan_node` 必须能识别简单问题并直接走 single 路径。
2. **重规划可能无限循环**：必须设置 `max_total_steps` 硬上限，防止计划反复修订。
3. **失败日志的噪音**：不是所有失败都值得记录。需要过滤：只记录 "非预期失败"（如 API 参数错误），不记录 "预期失败"（如搜索无结果）。
4. **多 Agent 的成本**：Anthropic 数据显示多 Agent 消耗约 15 倍 token。必须确保任务真正可分离才使用扇出模式。
5. **向后兼容**：所有改动通过 `AgentState` 字段扩展实现，旧字段（如 `delegation_plan`）保留但标记为 deprecated，确保平滑迁移。