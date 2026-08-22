# Project Matrix — 个人智能协作网络

基于“通用底座 + 领域聚焦”架构的 Agent 应用与 Runtime，内置 5 个 Agent（Commander + 4 个 Domain Agent），覆盖编程、投资、知识管理、媒体生成四大领域。顶层执行统一进入独立 Runtime；Runtime 负责可恢复 operation、approval、effect 和 session 状态。

## 架构

```
personal-agent/                     # 独立 Git 仓库
├── src/matrix/                     # 核心框架包
│   ├── agent/                      # Agent 定义与注册
│   │   ├── base.py                 # AgentDefinition 数据类
│   │   ├── commander.py            # Commander Agent 定义
│   │   ├── registry.py             # AgentRegistry 注册中心
│   │   └── domain_agents/          # 领域 Agent 定义
│   │       ├── coding_assistant.py # 编程助手
│   │       ├── investment_analyst.py # 投资分析员
│   │       ├── knowledge_manager.py  # 知识管理员
│   │       └── media_generator.py  # 媒体生成器
│   ├── chat/                       # 对话编排服务
│   ├── context/                    # 上下文管理（DataBus、Compaction、Budget）
│   ├── evaluation/                 # 评估框架（Eval runner、baseline、metrics）
│   ├── guardrails/                 # 安全护栏（输入/输出/间接注入/隐私/Tool）
│   ├── llm/                        # LLM 客户端（Codex / DeepSeek；Agnes 用于媒体生成）
│   ├── memory/                     # 记忆演化（MemoryEvolution 4 阶段管线）
│   ├── observability/              # 可观测性（OTel、JSONL Trace）
│   ├── orchestration/              # LangGraph 编排（图、节点、状态、反幻觉）
│   ├── rag/                        # 向量检索（ChromaDB + BM25）
│   ├── server/                     # FastAPI 服务（路由、中间件、静态文件）
│   ├── skills/                     # 技能加载与执行
│   ├── tools/                      # 工具注册（finance/agnes/code/web/mcp/rag）
│   ├── web/                        # 前端 React 应用（Vite + TypeScript）
│   ├── config.py                   # 统一配置
│   └── store.py                    # 持久化存储（SQLite）
├── docs/                           # 项目文档
│   ├── architecture.md             # 当前系统架构
│   ├── frontend-spec.md            # 当前前端规格
│   ├── quality-gate-plan.md        # 当前质量门禁
│   └── archive/                    # 已完成方案和历史差距记录
├── scripts/                        # 运维脚本
├── tools/mcp/                      # 独立 MCP server 源码（browser/utility）
├── tests/                          # 测试用例
└── var/                            # 运行时数据（不提交）
```

## 快速开始

```bash
# 安装依赖
uv sync

# 配置环境变量
cp .env.example .env
# 至少配置 JWT_SECRET；默认文本 provider 为 Codex。
# 使用 DeepSeek 时再配置 DEEPSEEK_API_KEY。

# 启动
./.venv/bin/python -m matrix
```

## Agent 列表

| Agent | 领域 | 说明 |
|-------|------|------|
| `commander` | 通用 | 指挥官，负责规划、路由和汇总 |
| `coding-assistant` | coding | 编程开发、代码分析、重构、调试 |
| `investment-analyst` | investment | 投资分析、持仓管理、市场研究 |
| `knowledge-manager` | knowledge | 知识整理、信息检索、学习笔记 |
| `media-generator` | media | 图片/视频生成 |

## 端点

### 聊天与会话

| 端点 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | SSE 流式对话 |
| `/chat/stream` | GET | SSE 流式对话（兼容 EventSource） |
| `/chat/confirm` | POST/GET | HITL 确认，恢复暂停的对话 |
| `/reset` | GET/POST | 重置会话 |

### 会话管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/sessions` | GET | 列出最近会话 |
| `/sessions/{id}` | GET/DELETE | 会话详情/删除 |
| `/sessions/{id}/messages` | GET | 会话消息历史 |
| `/sessions/batch-delete` | POST | 批量删除 |
| `/sessions/batch-archive` | POST | 批量归档 |
| `/sessions/batch-unarchive` | POST | 批量取消归档 |
| `/sessions/{id}/branch` | POST | 从指定消息创建会话分支 |
| `/sessions/{id}/branches` | GET | 列出会话的所有分支 |
| `/sessions/{id}/leaf` | GET | 获取会话当前叶子节点 |

### 技能管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/skills` | GET/POST | 技能列表/创建 |
| `/skills/{name}` | PUT/DELETE | 技能更新/删除 |
| `/skills/{name}/knowledge/{file}` | GET/PUT/DELETE | 知识文件读写 |
| `/skills/{name}/scripts/{file}` | GET/PUT/DELETE | 脚本文件读写 |
| `/skills/{name}/knowledge` | GET | 列出技能知识文件 |

### 工具与提供商

| 端点 | 方法 | 说明 |
|------|------|------|
| `/tools` | GET | 列出已注册工具 |
| `/tools/call` | POST | 直接调用工具 |
| `/api/provider` | GET/POST | LLM 提供商列表/切换 |

### Runtime 与记忆

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/runtime/operations` | GET | 查询当前用户的 Runtime operations |
| `/api/runtime/approvals` | GET | 查询待处理审批 |
| `/api/runtime/operations/{id}/events` | GET | 查询 operation 事件 |
| `/api/runtime/operations/{id}/retry-context` | GET | 获取 recovery-required operation 的安全重试上下文 |
| `/memory/list` | GET | 查询用户记忆 |
| `/memory/lessons` | GET | 查询跨会话经验教训 |

### 其他

| 端点 | 方法 | 说明 |
|------|------|------|
| `/healthz` | GET | 健康检查 |
| `/api/auth/register` | POST | 注册 |
| `/api/auth/login` | POST | 登录 |
| `/api/auth/logout` | POST | 登出 |
| `/api/upload` | POST | 文件上传 |
| `/api/trace/sessions` | GET | Trace 会话列表 |
| `/api/trace/sessions/{id}` | GET | Trace 详情 |
| `/api/trace/events` | GET | Trace 事件查询 |
| `/api/trace/stats` | GET | Trace 统计 |
| `/api/trace/spans` | GET | OTEL 标准化 spans |
| `/api/trace/otlp/export` | GET | OTLP 导出数据 |
| `/mcp/servers` | GET/POST | MCP 服务器管理 |
| `/mcp/servers/{name}` | PUT/DELETE | MCP 服务器更新/删除 |
| `/mcp/servers/{name}/toggle` | POST | MCP 服务器启用/禁用 |

## 当前执行边界

- 默认模式为 `read_only`。
- `writeback` 只允许显式 allowlist 操作，并且必须经过 Runtime approval。
- `writeback.execute_plan` 通过 `personal-os` API 执行，Agent 不直接写 `personal-assets`。
- Memory 和 Skill mutation 调用 `personal-os /api/vault/*`，由 AssetStore 提交和同步。
- Runtime SQLite 保存可恢复运行态，不是 finance facts 或 broader knowledge 的事实源。

## 开发

```bash
# 运行测试
./.venv/bin/python -m pytest

# 开发启动
bash scripts/dev.sh
```
