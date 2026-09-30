# Codex 可执行文件发现

## 范围

由 `personal-agent` 加载配置时统一发现 Codex 可执行文件；`personal-os`
App 不再维护重复的候选路径。仅处理路径发现，不修改模型、认证或 Runtime 行为。

## 规则

- `CODEX_BIN=codex`、空值和未设置均表示自动发现。
- 自动发现先查 `PATH`，再查 `~/.local/bin/codex`；macOS 额外检查
  Homebrew 常见位置、`/Applications` 和 `~/Applications` 下的
  `Codex.app` / `ChatGPT.app`，覆盖已知的新旧 CLI 布局。
- 自定义命令或路径严格优先，支持 `~`；不存在或不可执行时不静默回退。
- 只选可执行文件，不递归扫描磁盘、不安装软件、不修改认证配置。
- 解析结果交给现有健康检查和 LLM 客户端。找不到时提示修正 `CODEX_BIN`
  或安装 Codex；下次启动重新发现。
- 本机 `.env` 使用 `CODEX_BIN=codex`，不提交本机配置。

## 验证边界

运行现有配置与 Codex 客户端相关测试、一次性路径解析检查和 macOS App 构建。
不新增单元测试，不执行全量回归、模型评测或完整 UI E2E。

自动发现仅覆盖已知位置，不能保证识别未来任意目录布局；非标准安装仍可通过
`CODEX_BIN` 显式指定。
