# Project Matrix — 个人智能协作网络

基于"岗位制"设计的通用 Agent 底座，首个落地场景为投资分析员。

## 架构

```
personal-agent/                # 独立 Git 仓库
├── src/matrix/                # 核心框架包
│   ├── config.py              # 统一配置
│   ├── llm/                   # LLM 提供者（DeepSeek / Anthropic）
│   ├── tools/                 # 工具注册系统
│   │   ├── registry.py        # ToolRegistry：注册、发现、调用
│   │   └── finance/           # Finance 只读工具
│   ├── chat.py                # 对话编排引擎
│   ├── observability/         # 可观测性（追踪日志）
│   └── server/                # FastAPI HTTP 服务
├── roles/                     # 岗位角色定义（V2.0+）
├── skills/                    # Skill 定义（V2.0+）
└── var/                       # 运行时数据（不提交）
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

## 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/healthz` | GET | 健康检查 |
| `/tools` | GET | 列出可用工具 |
| `/tools/call` | POST | 直接调用工具 |
| `/chat` | POST | SSE 流式对话 |
| `/reset` | POST | 重置会话 |

## 开发

```bash
# 运行测试
python -m pytest

# 开发启动
bash scripts/dev.sh
```