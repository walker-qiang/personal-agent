# Project Matrix — 个人智能协作网络

基于"通用底座 + 领域聚焦"架构的 Agent 系统，内置 5 个 Agent（Commander + 4 个 Domain Agent），覆盖编程、投资、知识管理、媒体生成四大领域。

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
│   ├── llm/                        # LLM 客户端（DeepSeek / Anthropic / Agnes）
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
│   ├── architecture.md             # 系统架构
│   ├── gap-analysis.md             # 差距分析与优化设计
│   ├── frontend-spec.md            # 前端规格
│   ├── quality-gate-plan.md        # 质量门禁
│   ├── code-sandbox-plan.md        # 代码沙箱安全模型
│   └── browser-automation-plan.md  # 浏览器自动化
├── scripts/                        # 运维脚本
├── tests/                          # 测试用例
└── var/                            # 运行时数据（不提交）
```

## 快速开始

```bash
# 安装依赖
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 启动
python -m matrix
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

## 开发

```bash
# 运行测试
python -m pytest

# 开发启动
bash scripts/dev.sh
```