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

Agent 按"岗位"定义，每个岗位有独立的系统提示、工具集、技能集：
- `investment_analyst`（投资分析员）— 主对话 Agent
- 更多岗位按需扩展

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

## 路由表

| 路径 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 纯 HTML 前端 |
| `/healthz` | GET | 健康检查 |
| `/chat` | POST | SSE 流式对话 |
| `/sessions` | GET/POST | 会话列表/创建 |
| `/sessions/{id}` | GET/DELETE | 会话详情/删除 |
| `/sessions/{id}/messages` | GET | 会话消息历史 |
| `/skills` | GET/POST | 技能列表/创建 |
| `/skills/{name}` | PUT/DELETE | 技能更新/删除 |
| `/skills/{name}/knowledge` | GET | 技能知识文件 |
| `/skills/{name}/scripts` | GET | 技能脚本文件 |
| `/tools` | GET | 工具列表 |
| `/tools/call` | POST | 调用工具 |
| `/api/auth/login` | POST | 登录 |
| `/api/auth/register` | POST | 注册 |
| `/api/provider` | GET/POST | 模型列表/切换 |
| `/api/upload` | POST | 文件上传 |
| `/api/trace` | GET | Trace 列表 |
| `/api/trace/{session_id}` | GET | Trace 详情 |
| `/api/mcp` | GET/POST | MCP 服务器管理 |
| `/api/mcp/{name}` | PUT/DELETE | MCP 服务器更新/删除 |
| `/api/mcp/{name}/toggle` | POST | MCP 服务器启用/禁用 |

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