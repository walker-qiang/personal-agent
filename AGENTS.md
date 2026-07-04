# personal-agent 工作规则

- 本仓库是 Project Matrix 的通用 Agent 底座，独立于 `personal-os`。
- 新增或更新项目文档时默认使用中文；代码标识、API path、env var、文件路径和约定术语可保留英文。
- 不修改 `personal-os` 代码；与 `personal-os` 的集成通过 HTTP 代理（`agent.go`）和共享 SQLite cache 实现。
- 不把运行态提交到 Git：`.env`、`var/`、`__pycache__`、`*.pyc`、`dist`、`build` 都必须忽略。
- `var/` 下的文件只是可重建的本地 cache，不提交。
- 所有工具默认只读；后续写操作需走受控接口。
- 工具调用必须记录审计日志（JSONL trace）。
- 代码风格：类型注解、dataclass、Protocol、无隐式全局状态。