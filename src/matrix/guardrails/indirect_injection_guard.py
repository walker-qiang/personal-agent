"""IndirectInjectionGuard: detect prompt injection in tool results.

When an agent fetches content from external sources (web search, web fetch,
news search, RAG retrieval), that content is injected into the LLM context as
a tool result.  A malicious web page can embed instructions like "ignore all
previous instructions" inside its body text, tricking the LLM into executing
attacker-controlled commands — this is an *indirect prompt injection* attack.

This guard inspects the textual content of tool results before they enter the
LLM message history and:
1. Flags suspicious patterns (prompt override, role hijack, instruction embedding).
2. In block mode, replaces the entire result with a safe placeholder so the
   LLM never sees the malicious payload.
3. In warn mode (default), sanitises the result by wrapping detected injection
   patterns in ``[FILTERED:category]`` tags so the LLM can still use the
   surrounding text but the attack payload is neutralised.

Default mode is **sanitize** (not block), because aggressively blocking all
tool results that contain a flagged substring would produce false positives on
legitimate financial news articles that happen to quote phrases like
"ignore previous guidance".
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import GuardConfig

logger = logging.getLogger(__name__)

# ---- Data structures ----------------------------------------------------------

@dataclass
class InjectionFinding:
    """A single injection pattern match in a tool result."""
    category: str          # e.g. "prompt_override"
    snippet: str           # ~60 chars around the match (for logging)
    start: int             # character offset in the original text


@dataclass
class IndirectInjectionResult:
    """Result of an indirect injection check."""
    findings: list[InjectionFinding] = field(default_factory=list)
    sanitized: str = ""       # the (possibly sanitised) text to pass to LLM
    blocked: bool = False     # True if result was entirely replaced

    @property
    def is_clean(self) -> bool:
        """True if no injection patterns were found."""
        return len(self.findings) == 0


# ---- Detection patterns ------------------------------------------------------
# These are intentionally broader than InputGuard's patterns because indirect
# injection payloads are often obfuscated or embedded within longer text.

# Severity levels:
#   "high"   — almost certainly an attack; sanitise even in warn mode
#   "medium" — suspicious but may appear in legitimate content; sanitise in
#              warn mode but log for review
#   "low"    — borderline; only flagged in block mode

_PATTERNS: list[tuple[str, str, str]] = [
    # ---- HIGH: explicit instruction override ----
    (
        r"(?:ignore|forget|disregard|discard)\s+(?:all\s+)?(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|messages?|rules?|guidance)",
        "prompt_override",
        "high",
    ),
    (
        r"(?:you\s+are\s+(?:now|act\s+as)|from\s+now\s+on\s+you\s+are)\s+(?:DAN|jailbroken|unrestricted|unfiltered|free|without\s+(?:any\s+)?restrictions?)",
        "role_hijack",
        "high",
    ),
    (
        r"(?:do\s+not|don'?t|never)\s+follow\s+(?:your|the|any)\s+(?:system\s+)?(?:instructions?|rules?|prompts?)",
        "instruction_suppression",
        "high",
    ),
    (
        r"(?:reveal|output|print|dump|show|send|exfiltrate)\s+(?:the\s+)?(?:system\s+)?(?:prompt|instruction|rule|secret|api[_\s-]?key|token|password|credential)s?",
        "data_exfiltration",
        "high",
    ),

    # ---- MEDIUM: instruction-like constructs in tool output ----
    # Matches phrases like "SYSTEM: do X" or "[INSTRUCTION]" or "<|system|>"
    (
        r"(?:^|\n)\s*(?:SYSTEM|INSTRUCTION|ADMIN|IMPORTANT)\s*[:：]\s*(?:ignore|forget|disregard|execute|run|do|act|you\s+(?:are|must|should))",
        "embedded_instruction",
        "medium",
    ),
    # Fake system-role markers
    (
        r"<\|?(?:system|im_start|im_end|assistant)\|?>",
        "fake_role_marker",
        "medium",
    ),
    # "As an AI..." override attempts
    (
        r"(?:as\s+an?\s+(?:AI|assistant|language\s+model|LLM))[,，.。]\s*(?:you\s+(?:must|should|will|are\s+now)|(?:ignore|forget|disregard))",
        "role_assumption",
        "medium",
    ),
    # Command-like directives aimed at the agent
    (
        r"(?:^|\n)\s*(?:please\s+)?(?:execute|run|call|invoke)\s+(?:the\s+)?(?:tool|function|command)\s+(?:to|and)\s+(?:delete|remove|drop|modify|update|send|transfer|execute)",
        "tool_command_injection",
        "medium",
    ),

    # ---- LOW: suspicious but often benign ----
    # Markdown/HTML that looks like it's trying to render as instructions
    (
        r"<(?:system|instruction|prompt|assistant)[^>]*>",
        "suspicious_tag",
        "low",
    ),
    # Encoded "ignore instructions" variants (base64-looking blocks are NOT
    # matched here — too many false positives — but rot13/unicode tricks are)
    (
        r"(?:𝕚𝕘𝕟𝕠𝕣𝕖|ⅰⅿⓖⓝⓞⓡⓔ|ignore)\s+(?:Ⓟⓡⓔⓥⓘⓞⓤⓢ|previous)\s+(?:ⅰ𝕟𝕤𝕥𝕣𝕦𝕔𝕥ⅰⓞ𝕟𝕤|instructions)",
        "obfuscated_override",
        "low",
    ),
]

# Pre-compile for performance
_COMPILED: list[tuple[re.Pattern, str, str]] = [
    (re.compile(p, re.IGNORECASE | re.MULTILINE), cat, sev)
    for p, cat, sev in _PATTERNS
]

# Tools whose results are most likely to carry indirect injection.
# Only these tools' results are checked by default to avoid perf overhead.
_HIGH_RISK_TOOLS = frozenset({
    "web_search", "news_search", "web_fetch",
    "rag_search", "rag_query",
    # MCP tools (external servers) — any tool starting with "mcp_"
})

_SNIPPET_RADIUS = 40  # chars before/after match for logging


def _is_high_risk(tool_name: str) -> bool:
    """Check if a tool is in the high-risk set for indirect injection."""
    if tool_name in _HIGH_RISK_TOOLS:
        return True
    return tool_name.startswith("mcp_")


def _extract_text(result: object) -> str:
    """Extract a plain-text representation from a tool result for scanning.

    Recursively concatenates string values from dicts/lists to preserve
    actual newlines (JSON serialisation would escape \\n and break regex
    line anchors).
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    parts: list[str] = []

    def _walk(obj: object) -> None:
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item)
        elif isinstance(obj, (int, float, bool)):
            parts.append(str(obj))

    _walk(result)
    return "\n".join(parts)


def _build_snippet(text: str, start: int, end: int) -> str:
    """Build a context snippet around a match for logging."""
    s = max(0, start - _SNIPPET_RADIUS)
    e = min(len(text), end + _SNIPPET_RADIUS)
    return text[s:e].replace("\n", "\\n")


# ---- Guard class -------------------------------------------------------------

class IndirectInjectionGuard:
    """Detects and neutralises indirect prompt injection in tool results.

    Usage::

        guard = IndirectInjectionGuard(config)
        result = guard.check("web_fetch", tool_result_dict)
        if result.blocked:
            # result.sanitized is a safe placeholder
        else:
            # result.sanitized has injection patterns neutralised
    """

    def __init__(self, config: GuardConfig):
        # In warn mode (default): sanitise by wrapping patterns in [FILTERED] tags
        # In block mode: replace entire result with a safe placeholder
        self._block_mode = config.injection_block_mode
        self._check_all_tools = config.injection_check_all_tools
        self._max_scan_length = 200_000  # skip scanning very large results

    def check(self, tool_name: str, result: object) -> IndirectInjectionResult:
        """Check a tool result for indirect injection patterns.

        Args:
            tool_name: The name of the tool that produced this result.
            result: The tool's return value (dict, list, str, etc.)

        Returns:
            IndirectInjectionResult with findings and sanitised text.
        """
        # Fast path: skip non-high-risk tools unless check_all is enabled
        if not self._check_all_tools and not _is_high_risk(tool_name):
            return IndirectInjectionResult(sanitized="")

        text = _extract_text(result)
        if not text or len(text) > self._max_scan_length:
            # Too large to scan reliably — let it through (ToolResultRefStore
            # will externalise it anyway, and the summary will be scanned)
            return IndirectInjectionResult(sanitized="")

        findings: list[InjectionFinding] = []
        high_severity_found = False

        for pattern, category, severity in _COMPILED:
            for m in pattern.finditer(text):
                snippet = _build_snippet(text, m.start(), m.end())
                findings.append(InjectionFinding(
                    category=category,
                    snippet=snippet,
                    start=m.start(),
                ))
                if severity == "high":
                    high_severity_found = True

        if not findings:
            return IndirectInjectionResult(sanitized="")

        # Log findings
        categories = list({f.category for f in findings})
        logger.warning(
            "indirect_injection: tool=%s findings=%d categories=%s",
            tool_name, len(findings), categories,
        )

        # In block mode, replace the entire result
        if self._block_mode and high_severity_found:
            return IndirectInjectionResult(
                findings=findings,
                sanitized=(
                    "[BLOCKED: indirect prompt injection detected in tool result. "
                    "The content has been withheld for safety.]"
                ),
                blocked=True,
            )

        # In sanitise mode, neutralise each matched span with [FILTERED:cat]
        sanitized_text = text
        # Sort findings by start position descending so offsets stay valid
        for f in sorted(findings, key=lambda x: x.start, reverse=True):
            # Find the actual matched text at this offset
            # (re-scan to get the exact span)
            for pattern, category, _ in _COMPILED:
                m = pattern.search(sanitized_text, f.start)
                if m and m.start() == f.start:
                    sanitized_text = (
                        sanitized_text[:f.start]
                        + f"[FILTERED:{category}]"
                        + sanitized_text[m.end():]
                    )
                    break

        return IndirectInjectionResult(
            findings=findings,
            sanitized=sanitized_text,
            blocked=False,
        )

    def check_and_sanitize(self, tool_name: str, result: object) -> object:
        """Check and return a sanitised result.

        Recursively sanitises string values within the result structure.
        If blocked, returns a placeholder dict.
        If clean, returns the original result unchanged.

        This is the convenience method to call from the tool execution pipeline.
        """
        res = self.check(tool_name, result)
        if res.is_clean:
            return result

        if res.blocked:
            # Return a blocked result dict
            if isinstance(result, dict):
                return {**result, "result": res.sanitized, "_injection_blocked": True}
            return {"result": res.sanitized, "_injection_blocked": True}

        # Sanitise string values recursively in the result structure
        sanitized_text = res.sanitized
        if not sanitized_text:
            return result

        # Build a mapping from original string values to sanitised versions
        # by finding which original strings contained injection patterns
        def _sanitize_obj(obj: object) -> object:
            if isinstance(obj, str):
                # Check if this specific string had injection patterns
                sub_res = self.check(tool_name, obj)
                if not sub_res.is_clean and not sub_res.blocked:
                    return sub_res.sanitized
                return obj
            elif isinstance(obj, dict):
                return {k: _sanitize_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize_obj(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(_sanitize_obj(item) for item in obj)
            return obj

        return _sanitize_obj(result)
