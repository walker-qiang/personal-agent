# 系统架构

## 概述

Project Matrix 是一个基于"岗位制"设计的通用 Agent 底座，首个落地场景为投资分析员。后端为 Python FastAPI + LangGraph，前端提供两种界面。

## 整体架构

```
┌──────────────────────────────────────────────────┐
│                    前端层                          │
│  ┌─────────────────────┐  ┌───────────────────┐  │
│  │ 纯 HTML 前端 (主 UI)  │  │ React SPA (开发中)  │  │
│  │ static/index.html    │  │ static/react-app/ │  │
│  │ 服务路径: /           │  │ 服务路径: /react-app│  │
│  └─────────┬───────────┘  └────────┬──────────┘  │
└────────────┼───────────────────────┼──────────────┘
             │  HTTP/SSE              │
┌────────────┴───────────────────────┴──────────────┐
│                   服务层 (FastAPI)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ 路由层    │ │ 中间件    │ │ 生命周期           │   │
│  │ /chat    │ │ Auth     │ │ 工具注册           │   │
│  │ /sessions│ │ CORS     │ │ Guardrails 装配    │   │
│  │ /skills  │ │          │ │ MCP Client 连接    │   │
│  │ /tools   │ │          │ │ RAG 初始化         │   │
│  │ /api/*   │ │          │ │ Code Sandbox 注册  │   │
│  └──────────┘ └──────────┘ └──────────────────┘   │
└────────────────────────────────────────────────────┘
             │
┌────────────┴───────────────────────────────────────┐
│                   核心层                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Agent    │ │ LLM      │ │ 编排              │   │
│  │ 岗位制   │ │ 多模型    │ │ LangGraph ReAct   │   │
│  │ 注册表   │ │ 客户端    │ │ Commander→Agent   │   │
│  └──────────┘ └──────────┘ └──────────────────┘   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ 工具      │ │ 技能      │ │ Guardrails       │   │
│  │ 注册表    │ │ 加载器    │ │ 输入/输出/工具     │   │
│  │ 能力声明  │ │ 执行器    │ │ 隐私/Trace        │   │
│  │ Finance  │ │          │ │ 间接注入          │   │
│  │ Web/Search│ │          │ │                   │   │
│  │ Code/MCP  │ │          │ │                   │   │
│  └──────────┘ └──────────┘ └──────────────────┘   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ 弹性机制  │ │ 进度监控  │ │                   │   │
│  │ 熔断器    │ │ 事件系统  │ │                   │   │
│  │ 优雅降级  │ │          │ │                   │   │
│  └──────────┘ └──────────┘ └──────────────────┘   │
└────────────────────────────────────────────────────┘
             │
┌────────────┴───────────────────────────────────────┐
│                   基础设施层                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Store    │ │ Trace    │ │ Memory            │   │
│  │ 会话持久化│ │ OTel 导出 │ │ 演化引擎          │   │
│  │ SQLite   │ │ JSONL    │ │ 上下文压缩         │   │
│  └──────────┘ └──────────┘ └──────────────────┘   │
└────────────────────────────────────────────────────┘
```

## 关键设计决策

### 两个前端并存

| 维度 | 纯 HTML 前端 | React SPA |
|------|-------------|-----------|
| 文件 | `static/index.html` (单文件) | `static/react-app/` (构建产物) |
| 服务路径 | `/` | `/react-app/` |
| 定位 | 管理员/开发者全功能面板 | 日常对话交互界面（开发中） |
| 依赖 | 零外部框架（仅 marked.min.js） | React 18 + TypeScript + Vite |
| 来源 | 手写维护 | `src/matrix/web/` 源码构建 |
| 构建方式 | 无需构建 | `cd src/matrix/web && npm run build` |

**重要规则**：两个前端各自独立，React 构建产物输出到 `static/react-app/` 子目录，**不得覆盖** `static/index.html` 和 `static/marked.min.js`。

### 岗位制 Agent

Agent 按"岗位"定义，每个岗位有独立的系统提示、工具集、技能集。系统采用"通用底座 + 领域聚焦"架构：Commander 作为通用底座负责规划和路由，4 个 Domain Agent 各聚焦一个领域。

| Agent ID | 名称 | 领域 | 工具 | 通用技能 | 领域技能 |
|----------|------|------|------|----------|----------|
| `commander` | 指挥官 | 通用 | 全部 | decision-mirror | wiki-health-check, karpathy-guidelines, personal-reflection, ingest-source-to-knowledge |
| `coding-assistant` | 编程助手 | coding | code.run_python, web_search, web_fetch, knowledge_search, mcp_browser_* | decision-mirror, karpathy-guidelines, planning-with-files | brainstorming |
| `investment-analyst` | 投资分析员 | investment | finance.*, web_search, web_fetch, news_search, code.run_python, mcp_browser_* | decision-mirror | anomaly-diagnosis, portfolio-review, allocation-check, investment-research, investment-watchlist |
| `knowledge-manager` | 知识管理员 | knowledge | knowledge_search, web_search, web_fetch, news_search, code.run_python, mcp_browser_* | decision-mirror | ingest-source-to-knowledge, wiki-health-check, personal-reflection, brainstorming |
| `media-generator` | 媒体生成器 | media | agnes.* | — | — |

**路由逻辑**：Commander 根据用户意图自动路由到对应 Agent：
- 编程开发、代码分析、重构、调试 → `coding-assistant`
- 投资/金融分析 → `investment-analyst`
- 知识整理、信息检索、学习笔记 → `knowledge-manager`
- 图片/视频生成 → `media-generator`
- 简单问题 Commander 直接回答，跨领域问题制定计划后委派

#### AgentDefinition 数据类

每个 Agent 通过 `AgentDefinition` dataclass 定义，位于 `src/matrix/agent/base.py`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `str` | 唯一标识（如 `coding-assistant`） |
| `name` | `str` | 中文名称（如 `编程助手`） |
| `description` | `str` | 简短描述，供 Commander 决策委派 |
| `domain` | `str` | 领域标签（`coding` / `investment` / `knowledge` / `media` / `commander`） |
| `persona` | `str` | System prompt 核心内容 |
| `expertise` | `list[str]` | 专业领域列表 |
| `tools` | `list[str]` | 工具名模式（支持 `*` 通配，空列表 = 全部可用） |
| `general_skills` | `list[str]` | 通用技能（skills/common/） |
| `domain_skills` | `list[str]` | 领域技能（skills/{domain}/） |
| `output_constraints` | `list[str]` | 输出约束规则 |
| `safety_rules` | `list[str]` | 安全规则 |
| `llm_provider` | `str` | LLM 提供方覆盖（可选） |
| `llm_model` | `str` | 具体模型覆盖（可选） |

`AgentDefinition` 提供以下关键方法：
- `all_skills` — 合并 `general_skills + domain_skills`
- `to_system_prompt()` — 生成完整的中文 system prompt
- `matches_tool(tool_name)` — 判断工具是否在 Agent 可用范围内
- `matches_skill(skill_name)` — 判断技能是否在 Agent 技能集中

### 工具体系

所有工具通过 `ToolRegistry` 统一注册，分类管理：
- `finance/` — 金融数据（持仓、快照、资产配置、实时行情）
- `web/` — 网页搜索、新闻搜索、网页抓取、天气
- `code/` — Python 代码沙箱（需 `MATRIX_CODE_SANDBOX_ENABLED=true`）
- `mcp/` — 外部 MCP 服务器工具（如 Playwright 浏览器自动化）
- `rag/` — 知识库检索
- `agnes/` — 图片/视频生成

#### 工具能力声明（P3）

每个工具通过 `ToolDefinition.capabilities` 字段声明其能力标签，支持 Commander 在规划阶段根据任务需求匹配合适的工具和 Agent：

```python
ToolDefinition(
    name="finance.holdings_summary",
    description="获取持仓摘要",
    capabilities=["market_data", "portfolio_analysis"],
    ...
)
```

**能力标签体系**：

| 能力标签 | 含义 | 示例工具 |
|----------|------|----------|
| `market_data` | 实时/历史行情数据 | `finance.realtime_quote`, `finance.stock_history` |
| `portfolio_analysis` | 持仓分析、资产配置 | `finance.holdings_summary`, `finance.bucket_allocation` |
| `web_search` | 互联网搜索 | `web_search`, `news_search`, `web_fetch` |
| `code_execution` | Python 代码沙箱 | `code.run_python` |
| `media_generation` | 图片/视频生成 | `agnes.generate_image`, `agnes.generate_video` |
| `weather` | 天气查询 | `weather_get_current` |

**能力聚合机制**：
- `ToolRegistry.get_capabilities_summary()` — 返回全局 `{capability: count}` 统计
- `ToolRegistry.get_tool_capabilities()` — 返回 `{tool_name: [capabilities]}` 映射
- `AgentRegistry.agents_for_commander()` — 为每个 Agent 计算其可用工具集的 `capabilities_summary`，注入 Commander 的 system prompt 中

```
Agent 可用能力示例：
- investment_analyst: market_data(2), portfolio_analysis(2), web_search(2), code_execution(1), weather(1)
```

这使 Commander 能够在规划时做出更精准的 Agent 选择，例如："需要用实时行情数据 → 选 investment_analyst（有 market_data 能力）"。

### 编排系统

#### Plan-and-Execute 流程

多步任务通过 Commander + DAG 拓扑排序协调执行：

```
commander_plan → _route_dag_first → [Send("delegate") × N 并行]
    → replan_node → _route_after_replan（循环）
    → aggregate
```

- **Commander** 生成 `delegation_plan`，包含步骤定义、Agent 分配、依赖关系
- **DAG 路由** 按拓扑排序取出无依赖步骤并行执行，有依赖步骤顺序执行
- **重规划** 每批步骤执行后检查，最多 2 次修正机会
- **聚合** 汇总所有步骤结果生成最终回答

#### 执行进度监控（P2）

通过结构化进度事件实时反馈多步执行状态，前端 SSE 流中接收 `type: "progress"` 事件。

**事件类型**：

| 事件 | 触发节点 | 数据结构 |
|------|----------|----------|
| `plan_created` | `commander_plan_node` | `{type, plan_type, total_steps, steps, message}` |
| `step_start` | `delegate_node` | `{type, step, total, agent, task, message}` |
| `step_done` | `delegate_node` | `{type, step, total, result_preview, message}` |
| `step_error` | `delegate_node` | `{type, step, error, message}` |
| `replan` | `replan_node` | `{type, reason, attempt, message}` |

**传输机制**：通过 `configurable.event_queue`（`queue.Queue`）传递，无 event_queue 时静默丢弃，不崩溃。仅在多步计划（`len(delegation_plan) > 1`）时发出事件。

### 弹性机制

#### CircuitBreaker 熔断器

防止工具连续失败浪费 token 和 API 配额。位于 `_helpers.py`。

**状态机**：

```
CLOSED ──(3 次连续失败)──> OPEN ──(30s 冷却)──> HALF_OPEN
  ↑                                                    │
  └────────────(成功)──────────────────────────────────┘
  └────────────(失败)────────> OPEN
```

**关键参数**：
- `failure_threshold`: 3 次连续失败
- `cooldown_seconds`: 30 秒冷却
- 隔离：按 `session_id` 隔离，不同会话互不影响

**集成点**：`react_tool_node` 中每次工具调用前检查熔断状态，被熔断的工具从 LLM 可见工具列表中移除（`_prune_tools()`），避免 LLM 反复调用已失败的工具。

#### 优雅降级（P0）

在 `aggregate_node` 中实现三层回退，确保用户始终收到有意义的回复：

1. **正常聚合**：所有 Agent 结果正常，LLM 汇总生成回答
2. **LLM 友好降级**：所有 Agent 结果均含错误时，用 LLM 生成友好的错误说明
3. **硬编码回退**：LLM 调用也失败时，返回硬编码的兜底消息，包含尝试的操作和失败原因

```python
# aggregate_node 三层降级
if all_errors:
    try:
        final_answer = await llm.complete(degradation_prompt, ...)
    except Exception:
        final_answer = _hardcoded_fallback(attempted_actions, errors)
```

### 安全模型

- 所有工具默认只读，写操作需受控接口
- 代码执行：7 层安全（进程隔离/文件隔离/资源限制/超时/输出截断/HITL/CodeGuard）
- Guardrails 管线：输入守卫 → 间接注入检测 → 工具守卫 → 输出守卫 → 隐私脱敏
- 工具调用记录审计日志（JSONL trace）

### 记忆系统

Matrix 采用多层记忆架构，结合 MemoryEvolution 管线自动维护记忆质量。

#### 记忆层级

| 记忆层 | 实现 | 说明 |
|--------|------|------|
| 工作记忆 | `state.working_memory` (pinned + insights) | 当前会话的临时上下文 |
| 情景记忆 | SQLite `messages` 表 + `get_history()` | 历史对话消息 |
| 语义记忆 | RAG (ChromaDB + BM25) | 长期知识检索 |
| 程序记忆 | Skills (YAML) | 可复用的执行流程 |
| 用户画像 | `user_profile` 表 | 用户偏好、策略 |

#### MemoryEvolution 4 阶段管线

每次对话后以后台任务运行，位于 `src/matrix/memory/evolution.py`：

**Stage 1: Importance Scoring（重要性评分）**
- 公式：`importance = decay_weight × type_weight × (1 + recency_boost)`
- `decay_weight`：`2^(-age / half_life)`，policy 类型固定为 1.0（不衰减）
- `type_weight`：policy = 2.0，preference = 1.0
- `recency_boost`：3 天内更新的记忆乘 1.5

**Stage 2: Conflict Detection（冲突检测）**
- 两条记忆 key 相似度 ≥ 0.75 且 value 矛盾时判定为冲突
- 检测逻辑：布尔对立（yes/no）、否定词模式（"不xxx"、"not xxx"）
- 解决策略：保留更新时间更新的那条

**Stage 3: Consolidation（合并）**
- 条件：key 相似度 ≥ 阈值、values 不冲突、同类型
- 有 LLM 时调用 LLM 语义合并，无 LLM 时用 Jaccard 相似度判断
- 精确重复直接删除旧条目

**Stage 4: Active Forgetting（主动遗忘）**
- 触发条件：记忆总数 > `max_memories`（默认 80）
- 保护规则：policy 类型永不遗忘、1 天内更新的不遗忘
- 按 importance 升序删除最低价值的 preference 记忆

### 反思回路（Reflexion）

位于 `orchestration/nodes/commander.py → reflection_node`，在 Agent 输出后评估回答质量：

- LLM 评估回答是否完整、准确、符合用户要求
- 不合格时生成 self-reflection 并重试（最多 1 次）
- 与 `anti_hallucination.py` 的 `verify_all_claims()` 配合，将事实性验证结果注入反思 prompt
- 验证通过（无 contradicted / unverified 声明）时跳过反思，节省 token

### HITL 确认系统

高风险操作需要人工确认（Human-in-the-Loop），通过 SSE 暂停-恢复机制实现：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/chat/confirm` | POST | 确认或跳过高风险操作，恢复对话 |
| `/chat/confirm` | GET | GET 版本，兼容 EventSource |

**高风险操作类型**：
- `code.run_python` — 代码执行
- `mcp_browser_click`、`mcp_browser_type`、`mcp_browser_select_option`、`mcp_browser_press_key` — 浏览器交互
- `mcp_browser_save_state`、`mcp_browser_restore_state` — 浏览器状态操作

HITL 流程：Agent 遇到高风险操作 → SSE 流暂停，发送 `confirm_request` 事件 → 前端展示确认对话框 → 用户确认/跳过 → 通过 `/chat/confirm` 恢复执行。

### DataBus 上下文管理

`src/matrix/context/` 实现 4 层上下文管理架构，解决长对话中的 token 窗口溢出问题：

**L1: ToolResultRefStore（工具结果引用存储）**
- 阈值：单结果 > 8000 字符 或 > 10 个数组元素 → 触发外化
- 大结果存入 SQLite，上下文中仅保留引用对象 `{__stored, __refId, __summary, __hint}`
- 提供 `get_stored_data(refId)` 工具供 LLM 按需取回完整数据
- 默认 TTL 1 小时，过期自动清理

**L2: SemanticCompressor（语义压缩，预留）**

**L3: Compaction（上下文压缩）**
- 触发条件：token 使用量 ≥ 85% 上下文窗口
- 目标：压缩到约 30% 窗口
- 输出结构化 JSON：`user_goal`、`execution_history`、`abandoned_paths`、`data_references`
- 使用 pipeline LLM 执行，失败时降级为简单截断

**L4: DataBus（数据索引）**
- 扫描 messages 中的 `__refId` 引用，构建 `refId → summary` 紧凑索引
- 索引注入系统提示，让 LLM 知道有哪些数据可用

**Budget 预算控制**（`context/budget.py`）：
- 拒绝阈值：98%（免费模型）
- 警告阈值：90%
- 输出预留：4096 tokens

## 路由表

### 根路由

| 路径 | 方法 | 功能 |
|------|------|------|
| `/` | GET | Web UI（纯 HTML 前端） |
| `/healthz` | GET | 健康检查（服务状态、LLM 可用性、provider/model） |

### 聊天与会话

| 路径 | 方法 | 功能 |
|------|------|------|
| `/chat` | POST | SSE 流式对话 |
| `/chat/stream` | GET | SSE 流式对话（兼容 EventSource） |
| `/chat/confirm` | POST | HITL 确认，恢复暂停的对话 |
| `/chat/confirm` | GET | GET 版 HITL 确认 |
| `/reset` | GET/POST | 重置会话 |

### 会话管理

| 路径 | 方法 | 功能 |
|------|------|------|
| `/sessions` | GET | 列出当前用户的最近会话 |
| `/sessions/{id}` | GET | 获取会话元数据 |
| `/sessions/{id}` | DELETE | 删除会话 |
| `/sessions/{id}/messages` | GET | 获取会话消息历史 |
| `/sessions/batch-delete` | POST | 批量删除多个会话 |
| `/sessions/batch-archive` | POST | 批量归档（隐藏）会话 |
| `/sessions/batch-unarchive` | POST | 批量取消归档 |

### 技能管理

| 路径 | 方法 | 功能 |
|------|------|------|
| `/skills` | GET | 列出所有可用技能 |
| `/skills` | POST | 创建新技能 |
| `/skills/{name}` | PUT | 更新技能 SKILL.md |
| `/skills/{name}` | DELETE | 删除技能目录 |
| `/skills/{name}/knowledge` | GET | 列出技能知识文件 |
| `/skills/{name}/knowledge/{filename}` | GET/PUT/DELETE | 读写删知识文件 |
| `/skills/{name}/scripts/{filename}` | GET/PUT/DELETE | 读写删脚本文件 |

### 工具与提供商

| 路径 | 方法 | 功能 |
|------|------|------|
| `/tools` | GET | 列出已注册工具 |
| `/tools/call` | POST | 直接调用工具 |
| `/api/provider` | GET | 列出可用 LLM/图像/视频提供商 |
| `/api/provider` | POST | 切换会话的 LLM 提供商/模型 |

### 认证

| 路径 | 方法 | 功能 |
|------|------|------|
| `/api/auth/register` | POST | 注册新用户 |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/logout` | POST | 用户登出 |

### 文件上传

| 路径 | 方法 | 功能 |
|------|------|------|
| `/api/upload` | POST | 上传文件（PNG/JPEG/PDF/TXT/MD/CSV/JSON/YAML，最大 10MB） |

### Trace 追踪

| 路径 | 方法 | 功能 |
|------|------|------|
| `/api/trace/sessions` | GET | 列出最近的 trace 会话 |
| `/api/trace/sessions/{id}` | GET | 获取指定会话的 trace 事件 |
| `/api/trace/events` | GET | 查询 trace 事件（支持过滤和分页） |
| `/api/trace/stats` | GET | 整体 trace 统计 |
| `/api/trace/spans` | GET | 查询 OTel 标准化 spans |
| `/api/trace/otlp/export` | GET | 获取缓冲的 OTLP 导出数据（调试） |

### MCP 管理

| 路径 | 方法 | 功能 |
|------|------|------|
| `/mcp/servers` | GET | 列出所有 MCP 服务器及连接状态 |
| `/mcp/servers` | POST | 添加 MCP 服务器 |
| `/mcp/servers/{name}` | PUT | 更新 MCP 服务器配置 |
| `/mcp/servers/{name}` | DELETE | 删除 MCP 服务器 |
| `/mcp/servers/{name}/toggle` | POST | 切换启用/禁用状态

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | Python 3.10+, FastAPI, LangGraph |
| LLM | DeepSeek, Anthropic Claude, Agnes |
| 向量检索 | ChromaDB, sentence-transformers, BM25 |
| 可观测 | OpenTelemetry (OTLP), JSONL Trace |
| 前端 (主) | 纯 HTML/CSS/JS, marked.js |
| 前端 (React) | React 18, TypeScript, Vite 5 |
| 存储 | SQLite (会话), 本地文件系统 (技能) |
| MCP | mcp>=1.27.0, Playwright |