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
│  │ Finance  │ │ 执行器    │ │ 隐私/Trace        │   │
│  │ Web/Search│ │          │ │ 间接注入          │   │
│  │ Code/MCP  │ │          │ │                   │   │
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