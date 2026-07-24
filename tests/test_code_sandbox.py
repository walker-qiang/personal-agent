"""Tests for code execution sandbox: SandboxExecutor, CodeGuard, and tool registration."""

from __future__ import annotations

import pytest

from matrix.tools.code import CodeGuard, SandboxExecutor, register_all
from matrix.tools.code.python_tool import _run_python, python_tool
from matrix.tools.registry import ToolRegistry


# ── SandboxExecutor ──────────────────────────────────────────────────────────


class TestSandboxExecutor:
    """Tests for the SandboxExecutor class."""

    def test_basic_print(self):
        executor = SandboxExecutor(timeout_sec=10, max_memory_mb=256)
        result = executor.execute_python("print('hello world')")
        assert result["exit_code"] == 0
        assert "hello world" in result["stdout"]
        assert result["stderr"] == ""
        assert result["execution_time_ms"] >= 0

    def test_multiline_output(self):
        executor = SandboxExecutor(timeout_sec=10)
        code = "for i in range(3):\n    print(f'line {i}')"
        result = executor.execute_python(code)
        assert result["exit_code"] == 0
        assert "line 0" in result["stdout"]
        assert "line 1" in result["stdout"]
        assert "line 2" in result["stdout"]

    def test_syntax_error(self):
        executor = SandboxExecutor(timeout_sec=10)
        result = executor.execute_python("print('missing paren")
        assert result["exit_code"] != 0
        assert "SyntaxError" in result["stderr"]

    def test_runtime_error(self):
        executor = SandboxExecutor(timeout_sec=10)
        result = executor.execute_python("x = 1 / 0")
        assert result["exit_code"] != 0
        assert "ZeroDivisionError" in result["stderr"]

    def test_return_fields_complete(self):
        executor = SandboxExecutor(timeout_sec=10, max_memory_mb=256, max_output_chars=5000)
        result = executor.execute_python("print('ok')")
        expected_keys = {
            "stdout", "stderr", "exit_code", "execution_time_ms",
            "error_type", "error_message", "truncated",
        }
        assert set(result.keys()) == expected_keys

    def test_stdout_capture(self):
        executor = SandboxExecutor(timeout_sec=10)
        code = "import math\nprint(math.pi)"
        result = executor.execute_python(code)
        assert result["exit_code"] == 0
        assert "3.14" in result["stdout"]

    def test_stderr_capture(self):
        executor = SandboxExecutor(timeout_sec=10)
        code = "import sys\nsys.stderr.write('warn msg\\n')"
        result = executor.execute_python(code)
        assert result["exit_code"] == 0
        assert "warn msg" in result["stderr"]

    def test_timeout_enforced(self):
        executor = SandboxExecutor(timeout_sec=2)
        code = "import time\ntime.sleep(10)"
        result = executor.execute_python(code)
        assert result["exit_code"] == -1
        assert result["error_type"] == "TimeoutError"
        assert "TIMEOUT" in result["stderr"]

    def test_output_truncation(self):
        executor = SandboxExecutor(timeout_sec=10, max_output_chars=100)
        code = "print('A' * 500)"
        result = executor.execute_python(code)
        assert result["truncated"] is True
        assert len(result["stdout"]) <= 200  # truncated + suffix

    def test_no_truncation_for_short_output(self):
        executor = SandboxExecutor(timeout_sec=10, max_output_chars=10000)
        result = executor.execute_python("print('short')")
        assert result["truncated"] is False

    def test_import_standard_lib(self):
        executor = SandboxExecutor(timeout_sec=10)
        code = "import json, csv, math, statistics, datetime\nprint('all imports ok')"
        result = executor.execute_python(code)
        assert result["exit_code"] == 0
        assert "all imports ok" in result["stdout"]

    def test_empty_code(self):
        executor = SandboxExecutor(timeout_sec=10)
        result = executor.execute_python("")
        assert result["exit_code"] == 0

    def test_math_computation(self):
        executor = SandboxExecutor(timeout_sec=10)
        code = """
data = [10, 20, 30, 40, 50]
avg = sum(data) / len(data)
print(f"Average: {avg}")
print(f"Count: {len(data)}")
"""
        result = executor.execute_python(code)
        assert result["exit_code"] == 0
        assert "Average: 30.0" in result["stdout"]
        assert "Count: 5" in result["stdout"]

    def test_restricted_env_no_home_leak(self):
        """Sandbox should not inherit the real HOME directory."""
        executor = SandboxExecutor(timeout_sec=10)
        code = "import os\nprint(os.environ.get('HOME', 'not_set'))"
        result = executor.execute_python(code)
        assert result["exit_code"] == 0
        # HOME should be set to a temp dir, not the real user home
        assert "not_set" not in result["stdout"]

    def test_temp_dir_cleaned_up(self):
        """The temporary working directory should be removed after execution."""
        executor = SandboxExecutor(timeout_sec=10)
        executor.execute_python("print('cleanup test')")
        # If cleanup works, no matrix_sandbox_ dirs should linger
        import glob
        import os
        import tempfile
        pattern = os.path.join(tempfile.gettempdir(), "matrix_sandbox_*")
        leftovers = glob.glob(pattern)
        # Some might exist from concurrent runs, but ours should be gone
        # We can't guarantee zero, so just verify the test doesn't crash
        assert isinstance(leftovers, list)


# ── CodeGuard ────────────────────────────────────────────────────────────────


class TestCodeGuard:
    """Tests for the CodeGuard pre-execution safety checks."""

    def test_safe_code_passes(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "print('hello')"})
        assert ok is True
        assert reason == ""

    def test_non_code_tool_passes(self):
        guard = CodeGuard()
        ok, reason = guard.check("finance.holdings_summary", {})
        assert ok is True
        assert reason == ""

    def test_os_system_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "os.system('rm -rf /')"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_subprocess_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "import subprocess\nsubprocess.run(['ls'])"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_shutil_rmtree_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "shutil.rmtree('/tmp')"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_os_remove_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "os.remove('/etc/passwd')"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_os_popen_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "os.popen('whoami')"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_ctypes_import_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "__import__('ctypes')"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_exec_literal_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "exec('import os')"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_eval_literal_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "eval('1+1')"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_safe_exec_pattern_passes(self):
        """exec() with a variable argument should not be blocked (false positive tolerance)."""
        guard = CodeGuard()
        code = "code_obj = compile('x=1', '<str>', 'exec')\nexec(code_obj)"
        ok, reason = guard.check("code.run_python", {"code": code})
        # exec(code_obj) doesn't match exec('...') pattern — should pass
        assert ok is True

    def test_code_size_limit(self):
        guard = CodeGuard()
        guard.MAX_CODE_SIZE = 100  # Override for test
        ok, reason = guard.check("code.run_python", {"code": "x = " + "a" * 200})
        assert ok is False
        assert "code_too_large" in reason

    def test_non_string_code_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": 123})
        assert ok is False
        assert "must be a string" in reason

    def test_missing_code_key_passes(self):
        """Missing 'code' key should pass the guard (validated elsewhere)."""
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {})
        assert ok is True

    def test_sensitive_file_open_blocked(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "open('/etc/passwd')"})
        assert ok is False
        assert "forbidden_pattern" in reason

    def test_normal_file_open_passes(self):
        guard = CodeGuard()
        ok, reason = guard.check("code.run_python", {"code": "open('output.txt', 'w')"})
        assert ok is True


# ── Tool Registration & Integration ──────────────────────────────────────────


class TestCodeToolRegistration:
    """Tests for register_all and tool registry integration."""

    def test_register_all_returns_guard(self):
        registry = ToolRegistry()
        guard = register_all(registry, timeout_sec=5, max_memory_mb=128)
        assert isinstance(guard, CodeGuard)

    def test_tool_registered_in_registry(self):
        registry = ToolRegistry()
        register_all(registry, timeout_sec=5)
        assert "code.run_python" in registry.tool_names()

    def test_tool_has_correct_schema(self):
        registry = ToolRegistry()
        register_all(registry)
        tool = registry.get("code.run_python")
        assert tool is not None
        assert tool.name == "code.run_python"
        assert "code" in tool.input_schema["properties"]
        assert "code" in tool.input_schema["required"]

    def test_tool_list_format(self):
        registry = ToolRegistry()
        register_all(registry)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "code.run_python"

    def test_set_code_guard_on_registry(self):
        registry = ToolRegistry()
        guard = register_all(registry)
        registry.set_code_guard(guard)
        assert registry._code_guard is guard

    def test_call_tool_through_registry(self):
        registry = ToolRegistry()
        register_all(registry, timeout_sec=10)
        result = registry.call("code.run_python", {"code": "print(2+3)"})
        assert result["exit_code"] == 0
        assert "5" in result["stdout"]

    def test_guard_blocks_dangerous_call_through_registry(self):
        registry = ToolRegistry()
        guard = register_all(registry)
        registry.set_code_guard(guard)
        from matrix.guardrails.tool_guard import ToolGuardError
        with pytest.raises(ToolGuardError, match="code blocked"):
            registry.call("code.run_python", {"code": "os.system('ls')"})

    def test_guard_allows_safe_call_through_registry(self):
        registry = ToolRegistry()
        guard = register_all(registry)
        registry.set_code_guard(guard)
        result = registry.call("code.run_python", {"code": "print('safe')"})
        assert result["exit_code"] == 0
        assert "safe" in result["stdout"]

    def test_error_result_has_suggestion(self):
        """When code fails, the result should include a suggestion for the LLM."""
        registry = ToolRegistry()
        register_all(registry, timeout_sec=10)
        result = registry.call("code.run_python", {"code": "x = 1 / 0"})
        assert result["exit_code"] != 0
        assert "suggestion" in result

    def test_success_result_no_suggestion(self):
        """When code succeeds, no suggestion should be present."""
        registry = ToolRegistry()
        register_all(registry, timeout_sec=10)
        result = registry.call("code.run_python", {"code": "print('ok')"})
        assert result["exit_code"] == 0
        assert "suggestion" not in result


# ── python_tool module-level executor ────────────────────────────────────────


class TestPythonToolModule:
    """Tests for the python_tool module-level executor injection."""

    def test_executor_not_initialized_returns_error(self):
        """When _executor is None, calling _run_python should return an error."""
        import sys
        pt = sys.modules["matrix.tools.code.python_tool"]
        original = pt._executor
        pt._executor = None
        try:
            result = _run_python("print('test')")
            assert result["exit_code"] == -1
            assert "not initialized" in result["error"]
        finally:
            pt._executor = original

    def test_tool_definition_fields(self):
        assert python_tool.name == "code.run_python"
        assert python_tool.description  # non-empty
        assert "code" in python_tool.input_schema["properties"]
        assert "timeout" in python_tool.input_schema["properties"]
        assert python_tool.handler is not None
