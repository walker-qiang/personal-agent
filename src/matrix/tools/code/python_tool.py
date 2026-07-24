"""code.run_python tool definition and handler."""

from __future__ import annotations

from typing import Any

from ..base import ToolDefinition
from .executor import SandboxExecutor

# Module-level executor instance (initialized by register_all in __init__.py)
_executor: SandboxExecutor | None = None


def _run_python(code: str, timeout: int = 0) -> dict[str, Any]:
    """Execute Python code in an isolated sandbox.

    Args:
        code: Python source code to execute.
        timeout: Timeout in seconds (0 = use configured default).
    """
    if _executor is None:
        return {"error": "Code sandbox not initialized", "exit_code": -1}

    result = _executor.execute_python(code, timeout=timeout or None)

    # Add guidance for LLM self-correction on failure
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
                "description": (
                    "要执行的 Python 代码。"
                    "不要使用 os.system/subprocess 等系统调用。"
                    "使用 print() 输出结果。"
                ),
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
