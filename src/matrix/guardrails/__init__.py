"""Security guardrails for Matrix Agent.

Five independent guard layers:
- InputGuard:              prompt injection / data exfiltration / role confusion detection
- OutputGuard:             PII redaction (phone, ID card, email, bank card, API key)
- ToolGuard:               tool blacklist, parameter validation, path traversal, rate limiting
- IndirectInjectionGuard:  detect & neutralise prompt injection in tool results
- TraceSanitizer:          PII redaction before trace persistence

All guards are fail-open by default — guard exceptions do not block requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .input_guard import InputGuard, InputResult
from .output_guard import OutputGuard, OutputResult
from .privacy import TraceSanitizer
from .tool_guard import ToolGuard, ToolGuardError
from .indirect_injection_guard import (
    IndirectInjectionGuard,
    IndirectInjectionResult,
    InjectionFinding,
)


@dataclass
class GuardConfig:
    """Configuration for all guard layers, loaded from environment variables."""

    input_enabled: bool = True
    input_block_mode: bool = False  # False = warn-only, True = block
    output_enabled: bool = True
    output_block_mode: bool = False  # False = sanitize only, True = block
    tool_enabled: bool = True
    tool_block_mode: bool = True  # False = warn-only, True = block
    tool_blacklist: list[str] = field(default_factory=list)
    trace_privacy: bool = True
    max_message_len: int = 51200  # 50KB
    # Indirect injection guard
    injection_enabled: bool = True
    injection_block_mode: bool = False  # False = sanitise, True = block
    injection_check_all_tools: bool = False  # False = only high-risk tools

    @classmethod
    def from_env(cls) -> GuardConfig:
        """Load guard configuration from environment variables."""
        import os

        def _bool_env(name: str, default: bool) -> bool:
            val = os.environ.get(name, "").strip().lower()
            if val in ("1", "true", "yes", "on"):
                return True
            if val in ("0", "false", "no", "off"):
                return False
            return default

        def _int_env(name: str, default: int) -> int:
            raw = os.environ.get(name, "").strip()
            try:
                return int(raw) if raw else default
            except ValueError:
                return default

        blacklist_raw = os.environ.get("GUARD_TOOL_BLACKLIST", "").strip()
        blacklist = [t.strip() for t in blacklist_raw.split(",") if t.strip()] if blacklist_raw else []

        return cls(
            input_enabled=_bool_env("GUARD_INPUT_ENABLED", True),
            input_block_mode=_bool_env("GUARD_INPUT_BLOCK_MODE", False),
            output_enabled=_bool_env("GUARD_OUTPUT_ENABLED", True),
            output_block_mode=_bool_env("GUARD_OUTPUT_BLOCK_MODE", False),
            tool_enabled=_bool_env("GUARD_TOOL_ENABLED", True),
            tool_block_mode=_bool_env("GUARD_TOOL_BLOCK_MODE", True),
            tool_blacklist=blacklist,
            trace_privacy=_bool_env("GUARD_TRACE_PRIVACY", True),
            max_message_len=_int_env("GUARD_MAX_MESSAGE_LEN", 51200),
            injection_enabled=_bool_env("GUARD_INJECTION_ENABLED", True),
            injection_block_mode=_bool_env("GUARD_INJECTION_BLOCK_MODE", False),
            injection_check_all_tools=_bool_env("GUARD_INJECTION_CHECK_ALL_TOOLS", False),
        )


class GuardrailPipeline:
    """Unified entry point: assembles all guards on demand."""

    def __init__(self, config: GuardConfig | None = None):
        self.config = config or GuardConfig.from_env()
        self.input = InputGuard(self.config) if self.config.input_enabled else None
        self.output = OutputGuard(self.config) if self.config.output_enabled else None
        self.tool = ToolGuard(self.config) if self.config.tool_enabled else None
        self.injection = IndirectInjectionGuard(self.config) if self.config.injection_enabled else None
        self.privacy = TraceSanitizer() if self.config.trace_privacy else None


__all__ = [
    "GuardConfig",
    "GuardrailPipeline",
    "InputGuard",
    "InputResult",
    "OutputGuard",
    "OutputResult",
    "ToolGuard",
    "ToolGuardError",
    "IndirectInjectionGuard",
    "IndirectInjectionResult",
    "InjectionFinding",
    "TraceSanitizer",
]