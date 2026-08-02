"""Tool registry: declarative registration, discovery, and invocation.

Five-step execution pipeline (inspired by Pi-Agent):
  1. prepareArguments — compatibility shim for LLM provider quirks
  2. validateArguments — lightweight schema type checking
  3. beforeToolCall — guards (ToolGuard, CodeGuard)
  4. execute — handler invocation + output truncation
  5. afterToolCall — IndirectInjectionGuard sanitization

All errors are encoded as {"error": "..."} dicts, never raised.
This lets the LLM see the error and decide the next step.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import FinanceToolError, ToolDefinition

logger = logging.getLogger("matrix.tools.registry")


class ToolRegistry:
    """Registry for tool definitions with validation and invocation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._guard: object | None = None  # ToolGuard or None
        self._code_guard: object | None = None  # CodeGuard or None
        self._injection_guard: object | None = None  # IndirectInjectionGuard
        self._circuit_breaker: object | None = None  # CircuitBreaker or None

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition."""
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all tool definitions in LLM-compatible format."""
        return [tool.to_dict() for tool in self._tools.values()]

    def tool_names(self) -> set[str]:
        """Return the set of registered tool names."""
        return set(self._tools.keys())

    def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Invoke a tool through the five-step pipeline.

        Returns a result dict. Errors are encoded as {"error": "..."}.
        Never raises — the caller (ReAct loop) can rely on getting a dict back.
        """
        args = arguments or {}
        if not isinstance(args, dict):
            return {"error": "arguments must be an object"}

        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"工具 {name} 不存在。可用工具: {', '.join(sorted(self._tools.keys())[:10])}"}

        # Step 1: prepareArguments
        args = self._prepare_arguments(tool, args)

        # Step 2: validateArguments
        ok, reason = self._validate_arguments(tool, args)
        if not ok:
            return {"error": f"参数验证失败: {reason}"}

        # Step 2.5: Circuit breaker check (before guards)
        if self._circuit_breaker and self._circuit_breaker.is_blocked(name):
            return {"error": f"工具 {name} 已被熔断（连续失败次数过多），暂时不可用。请稍后重试或使用其他工具。"}

        # Step 3: beforeToolCall — ToolGuard + CodeGuard
        if self._guard:
            ok, reason = self._guard.check(name, args, session_id=session_id)
            if not ok:
                return {
                    "error": (
                        f"工具 {name} 被安全策略拦截: {reason}。"
                        "请调整参数或改用其他工具。"
                    )
                }

        if self._code_guard:
            ok, reason = self._code_guard.check(name, args)
            if not ok:
                return {
                    "error": (
                        f"代码安全策略拦截: 工具 {name} 被拦截: {reason}。"
                        "请移除危险代码后重试。"
                    )
                }

        # Step 4: execute + truncate
        try:
            result = tool.handler(**args)
        except FinanceToolError as err:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(name)
            return {"error": self._format_error(name, args, err)}
        except (TypeError, ValueError) as err:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(name)
            return {"error": self._format_error(name, args, err)}
        except Exception as err:
            if self._circuit_breaker:
                self._circuit_breaker.record_failure(name)
            return {"error": self._format_error(name, args, err)}

        # Circuit breaker: record success
        if self._circuit_breaker:
            self._circuit_breaker.record_success(name)

        # Apply truncation to structured results
        if isinstance(result, dict):
            from .truncate import truncate_result
            result = truncate_result(result)

        # Step 5: afterToolCall — IndirectInjectionGuard
        if self._injection_guard:
            try:
                result = self._injection_guard.check_and_sanitize(name, result)
            except Exception as exc:
                logger.warning(
                    "injection_guard error (tool=%s): %s — returning safe placeholder",
                    name, exc,
                )
                # Fail-closed: return a safe error instead of un-sanitized
                # result that may contain injected malicious content.
                return {"error": f"工具 {name} 的结果安全检查失败，已拦截。请重试或使用其他工具。"}

        return result

    # ---- Pipeline steps ----

    @staticmethod
    def _prepare_arguments(tool: ToolDefinition, args: dict[str, Any]) -> dict[str, Any]:
        """Step 1: compatibility shim for LLM provider parameter quirks.

        - Drops fields not in the tool's input_schema
        - Restores stringified arrays back to real arrays
        """
        props = tool.input_schema.get("properties", {})
        cleaned: dict[str, Any] = {}
        for key, value in args.items():
            if key not in props:
                continue
            expected_type = props[key].get("type", "")
            if expected_type == "array" and isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        cleaned[key] = parsed
                        continue
                except json.JSONDecodeError:
                    pass
            cleaned[key] = value
        return cleaned

    @staticmethod
    def _validate_arguments(tool: ToolDefinition, args: dict[str, Any]) -> tuple[bool, str]:
        """Step 2: lightweight schema validation.

        Checks required fields and basic type matching without a full
        JSON Schema validator.
        """
        schema = tool.input_schema

        # Required fields
        for req in schema.get("required", []):
            if req not in args:
                return False, f"缺少必需参数: {req}"

        # Type checking
        props = schema.get("properties", {})
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        for key, value in args.items():
            expected = props.get(key, {}).get("type", "")
            if expected and expected in type_map:
                # bool is a subclass of int in Python — reject it for any
                # non-boolean type to prevent True/False being accepted as
                # integer or number.
                if expected != "boolean" and isinstance(value, bool):
                    return False, f"参数 {key} 类型错误: 期望 {expected}, 得到 boolean"
                if expected == "boolean" and not isinstance(value, bool):
                    return False, f"参数 {key} 类型错误: 期望 boolean, 得到 {type(value).__name__}"
                if expected != "boolean" and not isinstance(value, type_map[expected]):
                    return False, f"参数 {key} 类型错误: 期望 {expected}, 得到 {type(value).__name__}"
        return True, ""

    @staticmethod
    def _format_error(name: str, args: dict[str, Any], err: Exception) -> str:
        """Format an error message with enough context for the LLM to self-correct."""
        args_preview = json.dumps(args, ensure_ascii=False, default=str)[:200]
        err_type = type(err).__name__
        err_msg = str(err)[:300]
        return f"工具 {name} 执行失败 [{err_type}]: {err_msg}。参数: {args_preview}"

    # ---- Guard setters (unchanged) ----

    def set_guard(self, guard: object) -> None:
        """Attach a ToolGuard instance for pre-execution safety checks."""
        self._guard = guard

    def set_code_guard(self, guard: object) -> None:
        """Attach a CodeGuard instance for code-specific pre-execution checks."""
        self._code_guard = guard

    def set_injection_guard(self, guard: object) -> None:
        """Attach an IndirectInjectionGuard for post-execution result scanning."""
        self._injection_guard = guard

    def set_circuit_breaker(self, breaker: object) -> None:
        """Attach a CircuitBreaker for per-tool failure tracking.

        When set, call() will:
        - Check is_blocked() before execution (returns error dict if blocked)
        - Call record_success() after successful execution
        - Call record_failure() after failed execution
        """
        self._circuit_breaker = breaker

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name, or None."""
        return self._tools.get(name)

    def get_capabilities_summary(self) -> dict[str, list[str]]:
        """Return a summary of tool capabilities grouped by capability tag."""
        summary: dict[str, list[str]] = {}
        for tool in self._tools.values():
            for cap in tool.capabilities:
                if cap not in summary:
                    summary[cap] = []
                summary[cap].append(tool.name)
        return summary

    def get_tool_capabilities(self) -> dict[str, list[str]]:
        """Return per-tool capability mapping: {tool_name: [capability, ...]}."""
        return {tool.name: tool.capabilities for tool in self._tools.values()}
