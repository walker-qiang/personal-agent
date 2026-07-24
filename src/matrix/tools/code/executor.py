"""Sandboxed Python code executor using subprocess with resource limits.

Security layers:
- L1 Process isolation: separate subprocess
- L2 Filesystem isolation: temporary working directory
- L3 Resource limits: memory, CPU time, file size (via preexec_fn)
- L4 Timeout: hard kill after timeout_sec
- L5 Output limits: truncate stdout/stderr to max_output_chars
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class SandboxExecutor:
    """Execute Python code in an isolated subprocess with resource limits."""

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
            timeout: Override timeout in seconds (capped to self._timeout).

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
            restricted_env: dict[str, str] = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": tmpdir,
                "TMPDIR": tmpdir,
                "PYTHONPATH": "",
                "LC_ALL": "en_US.UTF-8",
                "LANG": "en_US.UTF-8",
            }
            if self._network_enabled:
                for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                    val = os.environ.get(key)
                    if val:
                        restricted_env[key] = val

            started = time.perf_counter()
            truncated = False

            try:
                proc = subprocess.run(
                    [sys.executable, "-S", str(script_path)],
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
                stdout = e.stdout if isinstance(e.stdout, str) else ""
                stderr = (e.stderr if isinstance(e.stderr, str) else "") + (
                    f"\n[TIMEOUT: exceeded {effective_timeout}s]"
                )
                exit_code = -1
                error_type = "TimeoutError"
                error_message = f"Execution timed out after {effective_timeout}s"

            except Exception as e:
                stdout = ""
                stderr = str(e)
                exit_code = -1
                error_type = type(e).__name__
                error_message = str(e)

            # Truncate output to prevent flooding the LLM context
            if len(stdout) > self._max_output_chars:
                stdout = stdout[: self._max_output_chars] + "\n... [truncated]"
                truncated = True
            if len(stderr) > self._max_output_chars:
                stderr = stderr[: self._max_output_chars] + "\n... [truncated]"
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
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _set_resource_limits(self) -> None:
        """Set resource limits in the child process (called via preexec_fn).

        Called after fork() but before exec() in the subprocess.
        All setrlimit calls are wrapped in try/except for cross-platform safety.
        """
        # Memory limit (address space)
        mem_bytes = self._max_memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass

        # CPU time limit (seconds) — grace period beyond wall-clock timeout
        cpu_limit = self._timeout + 5
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        except (ValueError, OSError):
            pass

        # File size limit (10MB)
        try:
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024)
            )
        except (ValueError, OSError):
            pass

        # Prevent core dumps
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass
