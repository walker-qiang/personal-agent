"""Code execution tools: sandboxed Python execution."""

from __future__ import annotations

from ..registry import ToolRegistry
from .executor import SandboxExecutor
from .guard import CodeGuard
from .python_tool import python_tool

__all__ = ["register_all", "python_tool", "SandboxExecutor", "CodeGuard"]


def register_all(
    registry: ToolRegistry,
    timeout_sec: int = 30,
    max_memory_mb: int = 512,
    max_output_chars: int = 10000,
    network_enabled: bool = False,
) -> CodeGuard:
    """Register all code execution tools.

    Only called when code_sandbox_enabled=True in config.

    Returns:
        CodeGuard instance to be wired into the ToolRegistry.
    """
    import sys

    from ..base import ToolDefinition

    # Get the actual module (not the ToolDefinition object that shadows it
    # at the package level due to `from .python_tool import python_tool`)
    _pt = sys.modules[__name__ + ".python_tool"]

    # Initialize executor and inject into the tool module
    executor = SandboxExecutor(
        timeout_sec=timeout_sec,
        max_memory_mb=max_memory_mb,
        max_output_chars=max_output_chars,
        network_enabled=network_enabled,
    )
    _pt._executor = executor

    # Register the tool — use the module's ToolDefinition directly
    _tool = _pt.python_tool
    registry.register(
        ToolDefinition(
            name=_tool.name,
            description=_tool.description,
            input_schema=_tool.input_schema,
            handler=_tool.handler,
        )
    )

    return CodeGuard()
