# 代码执行沙箱改动计划

> 目标：为 personal-agent 新增 `code.run_python` 工具，让 Agent 具备在隔离环境中执行 Python 代码的能力，从而支持数据计算、格式转换、回测验证等任务。

## 1. 设计目标与约束

### 目标
- Agent 可通过 `code.run_python` 工具执行 Python 代码
- 执行结果（stdout / stderr / exit_code）返回给 LLM，支持**执行错误自纠正**（LLM 看到 stderr 后自动修正代码重试）
- 代码在隔离沙箱中运行：临时工作目录 + 资源限制 + 超时控制

### 约束（来自 AGENTS.md）
- 所有工具默认只读；写操作需走受控接口 → 代码执行默认**关闭**，通过 env var 显式开启
- 工具调用必须记录审计日志（JSONL trace）→ 复用现有 `_trace` 机制
- 代码风格：类型注解、dataclass、Protocol、无隐式全局状态
- 不把运行态提交到 Git

### 安全模型
| 层级 | 机制 | 说明 |
|------|------|------|
| L1 进程隔离 | `subprocess.run()` 独立进程 | 代码在子进程中执行，崩溃不影响主服务 |
| L2 文件隔离 | `tempfile.mkdtemp()` 临时目录 | 代码只能读写临时目录，不访问宿主文件系统 |
| L3 资源限制 | `resource.setrlimit()` via `preexec_fn` | 内存上限 512MB、CPU 时间上限 30s、文件大小上限 10MB |
| L4 超时控制 | `subprocess.run(timeout=...)` | 硬超时 30s（可配置），超时杀进程 |
| L5 输出限制 | stdout/stderr 截断 10000 chars | 防止输出洪水 |
| L6 HITL 确认 | `_is_high_risk("code.run_python")` → True | 工具名含 "run"，自动触发人工确认 |
| L7 代码审查 | `CodeGuard` 预检 | 禁止危险模式（`os.system`、`subprocess`、`shutil.rmtree` 等） |

## 2. 架构设计

```
LLM 调用 code.run_python(code="...")
    │
    ▼
ToolRegistry.call("code.run_python", {"code": "..."})
    │
    ├── ToolGuard.check()          ← 现有：黑名单、参数大小、路径遍历
    ├── CodeGuard.check()          ← 新增：代码危险模式检测
    │
    ▼
SandboxExecutor.execute_python(code)
    │
    ├── 1. 创建临时目录 (tempfile.mkdtemp)
    ├── 2. 写入代码到 /tmp/xxx/script.py
    ├── 3. subprocess.run([sys.executable, script.py],
    │       cwd=temp_dir,
    │       env=restricted_env,        ← 仅保留 PATH/PYTHONPATH
    │       timeout=30,
    │       preexec_fn=_set_limits,    ← 内存/CPU/文件大小限制
    │       capture_output=True)
    ├── 4. 截断 stdout/stderr
    ├── 5. 清理临时目录
    │
    ▼
返回 {stdout, stderr, exit_code, execution_time_ms, error_type?}
    │
    ├── IndirectInjectionGuard.check_and_sanitize()  ← 现有：扫描输出中的注入
    │
    ▼
结果注入 tool_results → LLM 可基于 stderr 自纠正
```

## 3. 文件清单

### 新建文件

| 文件路径 | 职责 | 行数估算 |
|----------|------|----------|
| `src/matrix/tools/code/__init__.py` | `register_all()` 注册入口 | ~30 |
| `src/matrix/tools/code/executor.py` | `SandboxExecutor` 核心执行器 | ~150 |
| `src/matrix/tools/code/python_tool.py` | `code.run_python` 工具定义 + handler | ~60 |
| `src/matrix/tools/code/guard.py` | `CodeGuard` 代码危险模式检测 | ~80 |
| `tests/test_code_sandbox.py` | 单元测试 + 集成测试 | ~200 |

### 修改文件

| 文件路径 | 改动内容 | 改动量 |
|----------|----------|--------|
| `src/matrix/config.py` | 新增 6 个沙箱配置字段 + env var + load_config | ~30 行 |
| `src/matrix/server/app.py` | `lifespan()` 中注册 code tools | ~5 行 |
| `src/matrix/guardrails/tool_guard.py` | `ToolGuard` 增加 `code_guard` 可选字段 | ~10 行 |
| `src/matrix/agent/domain_agents/investment_analyst.py` | tools 列表增加 `"code.run_python"` | ~1 行 |
| `pyproject.toml` | 无新依赖（仅用标准库 subprocess + resource） | 0 |

## 4. 核心组件详细设计

### 4.1 SandboxConfig（config.py 扩展）

```python
# 新增 env var 名称
ENV_CODE_SANDBOX_ENABLED = "MATRIX_CODE_SANDBOX_ENABLED"
ENV_CODE_SANDBOX_TIMEOUT_SEC = "MATRIX_CODE_SANDBOX_TIMEOUT_SEC"
ENV_CODE_SANDBOX_MAX_MEMORY_MB = "MATRIX_CODE_SANDBOX_MAX_MEMORY_MB"
ENV_CODE_SANDBOX_MAX_OUTPUT_CHARS = "MATRIX_CODE_SANDBOX_MAX_OUTPUT_CHARS"
ENV_CODE_SANDBOX_NETWORK = "MATRIX_CODE_SANDBOX_NETWORK"

# AgentConfig 新增字段（frozen dataclass）
code_sandbox_enabled: bool = False          # 默认关闭
code_sandbox_timeout_sec: int = 30           # 超时秒数
code_sandbox_max_memory_mb: int = 512        # 内存上限
code_sandbox_max_output_chars: int = 10000   # 输出截断
code_sandbox_network: bool = False           # 网络访问（MVP 不实现隔离，仅标记）
```

`load_config()` 中新增读取逻辑：
```python
code_sandbox_enabled = os.environ.get(ENV_CODE_SANDBOX_ENABLED, "").strip().lower() in ("1", "true", "yes")
code_sandbox_timeout_sec = clamp_int_env(ENV_CODE_SANDBOX_TIMEOUT_SEC, 30, 5, 120)
code_sandbox_max_memory_mb = clamp_int_env(ENV_CODE_SANDBOX_MAX_MEMORY_MB, 512, 128, 4096)
code_sandbox_max_output_chars = clamp_int_env(ENV_CODE_SANDBOX_MAX_OUTPUT_CHARS, 10000, 1000, 50000)
code_sandbox_network = os.environ.get(ENV_CODE_SANDBOX_NETWORK, "").strip().lower() in ("1", "true", "yes")
```

### 4.2 SandboxExecutor（executor.py）

```python
"""Sandboxed Python code executor using subprocess with resource limits."""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a sandboxed code execution."""
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: int
    error_type: str           # "" if success, else exception class name
    error_message: str        # "" if success, else short error description
    truncated: bool           # True if stdout/stderr was truncated


class SandboxExecutor:
    """Execute Python code in an isolated subprocess with resource limits.

    Security layers:
    - Process isolation: separate subprocess
    - Filesystem isolation: temporary working directory
    - Resource limits: memory, CPU time, file size (via preexec_fn)
    - Timeout: hard kill after timeout_sec
    - Output limits: truncate stdout/stderr to max_output_chars
    """

    def __init__(
        self,
        timeout_sec: int = 30,
        max_memory_mb: int = 512,
        max_output_chars: int = 10000,
        network_enabled: bool = False,
    ) -> None:
        self._timeout = timeout_sec
        self._max_memory_mb = max_memory_mb
        self._max_output_chars = max_output_chars
        self._network_enabled = network_enabled

    def execute_python(self, code: str, timeout: int | None = None) -> dict[str, Any]:
        """Execute Python code and return structured result.

        Args:
            code: Python source code to execute.
            timeout: Override timeout in seconds (max: self._timeout).

        Returns:
            Dict with stdout, stderr, exit_code, execution_time_ms,
            error_type, error_message, truncated.
        """
        effective_timeout = min(timeout or self._timeout, self._timeout)
        tmpdir = tempfile.mkdtemp(prefix="matrix_sandbox_")

        try:
            script_path = Path(tmpdir) / "script.py"
            script_path.write_text(code, encoding="utf-8")

            # Restricted environment: only essential vars
            restricted_env = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": tmpdir,           # Redirect HOME to temp
                "TMPDIR": tmpdir,
                "PYTHONPATH": "",         # Clear PYTHONPATH
                "LC_ALL": "en_US.UTF-8",
                "LANG": "en_US.UTF-8",
            }
            if self._network_enabled:
                # Pass through proxy settings if network is enabled
                for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                    if key in os.environ:
                        restricted_env[key] = os.environ[key]

            started = time.perf_counter()
            truncated = False

            try:
                proc = subprocess.run(
                    [sys.executable, "-S", str(script_path)],  # -S: no site-packages init
                    cwd=tmpdir,
                    env=restricted_env,
                    timeout=effective_timeout,
                    capture_output=True,
                    text=True,
                    preexec_fn=self._set_resource_limits,
                )
                stdout = proc.stdout
                stderr = proc.stderr
                exit_code = proc.returncode
                error_type = ""
                error_message = ""

            except subprocess.TimeoutExpired as e:
                stdout = e.stdout or "" if isinstance(e.stdout, str) else ""
                stderr = (e.stderr or "") + f"\n[TIMEOUT: exceeded {effective_timeout}s]"
                exit_code = -1
                error_type = "TimeoutError"
                error_message = f"Execution timed out after {effective_timeout}s"

            except Exception as e:
                stdout = ""
                stderr = str(e)
                exit_code = -1
                error_type = type(e).__name__
                error_message = str(e)

            # Truncate output
            if len(stdout) > self._max_output_chars:
                stdout = stdout[:self._max_output_chars] + "\n... [truncated]"
                truncated = True
            if len(stderr) > self._max_output_chars:
                stderr = stderr[:self._max_output_chars] + "\n... [truncated]"
                truncated = True

            elapsed_ms = round((time.perf_counter() - started) * 1000)

            return {
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
                "execution_time_ms": elapsed_ms,
                "error_type": error_type,
                "error_message": error_message,
                "truncated": truncated,
            }

        finally:
            # Cleanup temp directory
            import shutil
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    def _set_resource_limits(self) -> None:
        """Set resource limits in the child process (via preexec_fn).

        Called after fork() but before exec() in the subprocess.
        """
        # Memory limit (address space)
        mem_bytes = self._max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass  # RLIMIT_AS may not be available on all platforms

        # CPU time limit (seconds)
        cpu_limit = self._timeout + 5  # Grace period beyond wall-clock timeout
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        except (ValueError, OSError):
            pass

        # File size limit (10MB)
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        except (ValueError, OSError):
            pass

        # Prevent core dumps
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass
```

### 4.3 CodeGuard（guard.py）

```python
"""CodeGuard: pre-execution safety checks for code execution tools."""

from __future__ import annotations

import re
from typing import Any


class CodeGuardError(Exception):
    """Raised when code is blocked by the code guard."""


class CodeGuard:
    """Checks Python code for dangerous patterns before execution.

    This is a defense-in-depth layer — the sandbox itself provides
    process/filesystem/resource isolation. This guard catches obviously
    dangerous code before spawning a subprocess.
    """

    # Patterns that are always blocked
    FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
        re.compile(r"\bsubprocess\.", re.IGNORECASE),
        re.compile(r"\bshutil\.rmtree\s*\(", re.IGNORECASE),
        re.compile(r"\bos\.remove\s*\(", re.IGNORECASE),
        re.compile(r"\bos\.unlink\s*\(", re.IGNORECASE),
        re.compile(r"\bopen\s*\(\s*['\"]/(?:etc|var|usr|bin|sbin|root|home)", re.IGNORECASE),
        re.compile(r"\b__import__\s*\(\s*['\"](?:ctypes|cffi)", re.IGNORECASE),
        re.compile(r"\bexec\s*\(\s*['\"]", re.IGNORECASE),  # exec("...") string exec
        re.compile(r"\beval\s*\(\s*['\"]", re.IGNORECASE),  # eval("...") string eval
    ]

    MAX_CODE_SIZE = 10240  # 10KB

    def check(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Check if code execution should be allowed.

        Returns:
            (allowed, reason) tuple.
        """
        if tool_name not in ("code.run_python", "code.run_shell"):
            return (True, "")

        code = arguments.get("code", "")
        if not isinstance(code, str):
            return (False, "code must be a string")

        # Size check
        if len(code.encode("utf-8")) > self.MAX_CODE_SIZE:
            return (False, f"code_too_large: {len(code)} bytes (max {self.MAX_CODE_SIZE})")

        # Pattern check
        for pattern in self.FORBIDDEN_PATTERNS:
            match = pattern.search(code)
            if match:
                return (False, f"forbidden_pattern: {match.group()!r}")

        return (True, "")
```

### 4.4 Python 工具定义（python_tool.py）

```python
"""code.run_python tool definition."""

from __future__ import annotations

from typing import Any

from ..base import ToolDefinition
from .executor import SandboxExecutor


# Module-level executor instance (initialized by register_all)
_executor: SandboxExecutor | None = None


def _run_python(code: str, timeout: int = 0) -> dict[str, Any]:
    """Execute Python code in sandbox.

    Args:
        code: Python source code to execute.
        timeout: Timeout in seconds (0 = use default).
    """
    if _executor is None:
        return {"error": "Code sandbox not initialized", "exit_code": -1}

    result = _executor.execute_python(code, timeout=timeout or None)
    # Add guidance for LLM self-correction
    if result["exit_code"] != 0 and result["stderr"]:
        result["suggestion"] = (
            "代码执行失败。请检查 stderr 中的错误信息，修正代码后重试。"
            "常见问题：语法错误、导入缺失、变量名拼写错误。"
        )
    return result


python_tool = ToolDefinition(
    name="code.run_python",
    description=(
        "在隔离沙箱中执行 Python 代码并返回结果。用于数据处理、数值计算、"
        "格式转换、图表生成等任务。\n\n"
        "执行环境：\n"
        "- 超时限制：30秒\n"
        "- 内存限制：512MB\n"
        "- 文件系统：仅限临时目录\n"
        "- 可用库：Python 标准库（json, csv, math, statistics, datetime 等）\n\n"
        "返回字段：stdout, stderr, exit_code, execution_time_ms\n"
        "如果 exit_code != 0，请根据 stderr 修正代码后重试。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码。不要使用 os.system/subprocess 等系统调用。",
            },
            "timeout": {
                "type": "integer",
                "description": "超时时间（秒），0 表示使用默认值 30 秒",
            },
        },
        "required": ["code"],
    },
    handler=_run_python,
)
```

### 4.5 注册入口（__init__.py）

```python
"""Code execution tools: sandboxed Python execution."""

from __future__ import annotations

from ..registry import ToolRegistry
from .executor import SandboxExecutor
from .python_tool import python_tool


def register_all(
    registry: ToolRegistry,
    timeout_sec: int = 30,
    max_memory_mb: int = 512,
    max_output_chars: int = 10000,
    network_enabled: bool = False,
) -> None:
    """Register all code execution tools.

    Only called when code_sandbox_enabled=True in config.
    """
    from ..base import ToolDefinition
    from .python_tool import _run_python
    import python_tool as _pt  # set module-level executor

    # Initialize executor
    executor = SandboxExecutor(
        timeout_sec=timeout_sec,
        max_memory_mb=max_memory_mb,
        max_output_chars=max_output_chars,
        network_enabled=network_enabled,
    )
    _pt._executor = executor

    registry.register(
        ToolDefinition(
            name=python_tool.name,
            description=python_tool.description,
            input_schema=python_tool.input_schema,
            handler=python_tool.handler,
        )
    )


__all__ = ["register_all", "python_tool", "SandboxExecutor"]
```

### 4.6 app.py 集成

在 `lifespan()` 中，RAG 初始化之后、MCP 之前添加：

```python
# ---- CODE SANDBOX ----
if config.code_sandbox_enabled:
    from ..tools.code import register_all as register_code_tools
    register_code_tools(
        tools_registry,
        timeout_sec=config.code_sandbox_timeout_sec,
        max_memory_mb=config.code_sandbox_max_memory_mb,
        max_output_chars=config.code_sandbox_max_output_chars,
        network_enabled=config.code_sandbox_network,
    )
    # Wire CodeGuard to ToolRegistry (additional pre-execution check)
    from ..tools.code.guard import CodeGuard
    code_guard = CodeGuard()
    # Extension: ToolGuard needs to support chained guards
    # OR: integrate into existing ToolGuard.check()
    logger.info("code: sandbox enabled (timeout=%ds, memory=%dMB)",
                config.code_sandbox_timeout_sec, config.code_sandbox_max_memory_mb)
# ---- END CODE SANDBOX ----
```

### 4.7 ToolGuard 扩展

在 `ToolRegistry.call()` 中，现有 ToolGuard 检查之后增加 CodeGuard 检查：

```python
# registry.py - call() 方法中
# ---- TOOL GUARD (pre-execution) ----
if self._guard:
    ok, reason = self._guard.check(name, args)
    if not ok:
        raise ToolGuardError(f"tool blocked: {reason}")

# ---- CODE GUARD (pre-execution, code tools only) ----
if self._code_guard:
    ok, reason = self._code_guard.check(name, args)
    if not ok:
        raise ToolGuardError(f"code blocked: {reason}")
# ---- END CODE GUARD ----
```

`ToolRegistry` 新增 `set_code_guard()` 方法。

### 4.8 Agent 绑定

**Investment Analyst** — 新增 `code.run_python` 到 tools 列表：

```python
# investment_analyst.py
INVESTMENT_ANALYST = AgentDefinition(
    ...
    tools=[
        "finance.*",
        "finance_query",
        "news_search",
        "web_search",
        "web_fetch",
        "code.run_python",    # ← 新增
    ],
    ...
)
```

**Commander** — tools 为空（全部可用），无需修改。

**Media Generator** — 不需要代码执行，不修改。

### 4.9 System Prompt 更新

在 `DOMAIN_AGENT_REACT_SYSTEM` 的 Tool Usage Rules 中增加：

```
## Code Execution
When you need to perform calculations, data processing, or format conversion:
- Use `code.run_python` to execute Python code
- The code runs in an isolated sandbox (30s timeout, 512MB memory)
- Only Python standard library is available (json, csv, math, statistics, datetime, etc.)
- If execution fails (exit_code != 0), read the stderr, fix the code, and retry
- Do NOT use os.system, subprocess, or file operations outside the temp directory
- Keep code concise — write only what's needed to solve the specific problem
```

## 5. 测试策略

### 单元测试（test_code_sandbox.py）

| 测试用例 | 验证点 |
|----------|--------|
| `test_execute_simple_print` | `print("hello")` → stdout="hello\n", exit_code=0 |
| `test_execute_math_calculation` | `print(2+2)` → stdout="4\n" |
| `test_execute_syntax_error` | 语法错误 → exit_code=1, stderr 含 "SyntaxError" |
| `test_execute_runtime_error` | `1/0` → exit_code=1, stderr 含 "ZeroDivisionError" |
| `test_timeout_infinite_loop` | `while True: pass` → error_type="TimeoutError" |
| `test_memory_limit` | 大列表分配 → 被 RLIMIT_AS 杀死 |
| `test_output_truncation` | 大量 print → truncated=True |
| `test_temp_dir_cleanup` | 执行后临时目录被清理 |
| `test_code_guard_blocks_os_system` | `os.system("ls")` → CodeGuard 拒绝 |
| `test_code_guard_blocks_subprocess` | `import subprocess` → CodeGuard 拒绝 |
| `test_code_guard_allows_pandas` | `import json` → 通过（标准库） |
| `test_code_size_limit` | >10KB 代码 → 拒绝 |
| `test_env_isolation` | 代码中 `os.environ["HOME"]` → 指向临时目录 |
| `test_self_correction_flow` | 模拟 LLM 看到 stderr 后修正代码重试 |

### 集成测试

| 测试用例 | 验证点 |
|----------|--------|
| `test_tool_registered_when_enabled` | config 启用时，`code.run_python` 出现在 tools 列表 |
| `test_tool_not_registered_when_disabled` | config 禁用时，`code.run_python` 不出现 |
| `test_hitl_triggered` | 工具名含 "run"，`_is_high_risk()` 返回 True |
| `test_trace_logged` | 执行后 JSONL trace 中有记录 |

## 6. 实施步骤

### Step 1: 配置扩展（~30 min）
- [ ] `config.py`：新增 6 个字段 + env var + `load_config()` 读取
- [ ] `.env.example`：新增配置项说明

### Step 2: SandboxExecutor 核心（~2h）
- [ ] `tools/code/executor.py`：`SandboxExecutor` 类
- [ ] `tools/code/guard.py`：`CodeGuard` 类
- [ ] 手动验证：`python -c "from matrix.tools.code.executor import SandboxExecutor; ..."`

### Step 3: 工具定义与注册（~1h）
- [ ] `tools/code/python_tool.py`：工具定义 + handler
- [ ] `tools/code/__init__.py`：`register_all()` 入口
- [ ] `server/app.py`：`lifespan()` 中条件注册
- [ ] `tools/registry.py`：新增 `set_code_guard()` + `call()` 中增加 CodeGuard 检查

### Step 4: Agent 集成（~30 min）
- [ ] `agent/domain_agents/investment_analyst.py`：tools 增加 `code.run_python`
- [ ] `orchestration/nodes/_helpers.py`：`DOMAIN_AGENT_REACT_SYSTEM` prompt 增加代码执行指引
- [ ] 验证 `_is_high_risk("code.run_python")` 返回 True

### Step 5: 测试（~2h）
- [ ] `tests/test_code_sandbox.py`：编写全部单元测试
- [ ] 运行测试：`pytest tests/test_code_sandbox.py -v`
- [ ] E2E 验证：启动服务，发送"帮我计算 [1,2,3,4,5] 的平均值"请求

### Step 6: .env 配置与文档（~15 min）
- [ ] `.env` 中添加 `MATRIX_CODE_SANDBOX_ENABLED=true`
- [ ] 重启服务验证

**总预估：~6h**

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `RLIMIT_AS` 在 macOS 上行为不一致 | 中 | 资源限制失效 | `preexec_fn` 中 try/except 静默失败；超时作为兜底 |
| 代码通过 `ctypes` 绕过限制 | 低 | 沙箱逃逸 | CodeGuard 禁止 `__import__('ctypes')`；`-S` 标志减少可导入模块 |
| LLM 生成恶意代码（间接注入） | 低 | 数据泄露 | CodeGuard + HITL 确认 + 临时目录隔离 |
| 临时目录未清理 | 低 | 磁盘泄漏 | `finally` 块 + `shutil.rmtree(ignore_errors=True)` |
| 大量并发执行耗尽资源 | 低 | 服务不可用 | MVP 单进程串行；后续可加并发限制 |
| `subprocess.run` 在某些环境不可用 | 极低 | 功能不可用 | 降级：返回 error，LLM 回退到纯推理 |

## 8. 后续演进路径（不在本次范围）

| 阶段 | 能力 | 说明 |
|------|------|------|
| Phase 2 | `code.run_shell` | Shell 命令执行（更高风险，需更严格 guard） |
| Phase 2 | 网络隔离 | macOS `sandbox-exec` profile 或 Linux `seccomp` |
| Phase 2 | 持久化工作目录 | 同一会话内共享文件系统状态 |
| Phase 3 | Docker 隔离 | 完整容器级隔离，支持任意语言 |
| Phase 3 | 并发执行池 | 限制并发数，队列管理 |
| Phase 3 | 预装包白名单 | 显式管理可导入的第三方库 |
