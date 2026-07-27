"""Unified tool output truncation.

Provides truncate_head / truncate_tail with:
- Dual limits: max_lines AND max_bytes (whichever hits first)
- UTF-8 boundary safety: never splits a multi-byte character
- Truncation metadata for downstream consumers
- Hint line appended when truncated

Usage:
    result = truncate_tail(bash_output)
    if result.truncated:
        print(f"Truncated: {result.total_lines} -> {result.output_lines} lines")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("matrix.tools.truncate")

# ---- Constants ----

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB


@dataclass
class TruncationResult:
    """Result of a truncation operation."""

    content: str
    truncated: bool
    truncated_by: str | None  # "lines" | "bytes" | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int


def truncate_head(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the beginning of content.

    Used for: file reads, search results, structured data —
    the beginning usually has the highest information density.
    """
    if not content:
        return TruncationResult(
            content="", truncated=False, truncated_by=None,
            total_lines=0, total_bytes=0, output_lines=0, output_bytes=0,
        )

    total_bytes = _byte_len(content)
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content, truncated=False, truncated_by=None,
            total_lines=total_lines, total_bytes=total_bytes,
            output_lines=total_lines, output_bytes=total_bytes,
        )

    # Collect from the front
    kept: list[str] = []
    byte_budget = max_bytes
    for line in lines:
        line_bytes = _byte_len(line) + 1  # +1 for newline
        if len(kept) >= max_lines:
            break
        if byte_budget < line_bytes:
            break
        kept.append(line)
        byte_budget -= line_bytes

    output = "\n".join(kept)
    output_bytes = _byte_len(output)

    truncated_by = "lines" if len(kept) >= max_lines else "bytes"
    hint = _build_hint(total_lines, total_bytes, len(kept), output_bytes)
    return TruncationResult(
        content=output + hint,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(kept),
        output_bytes=output_bytes,
    )


def truncate_tail(
    content: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TruncationResult:
    """Keep the end of content.

    Used for: command output, logs, stack traces —
    the end usually has the error/result signal.
    """
    if not content:
        return TruncationResult(
            content="", truncated=False, truncated_by=None,
            total_lines=0, total_bytes=0, output_lines=0, output_bytes=0,
        )

    total_bytes = _byte_len(content)
    lines = content.split("\n")
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content, truncated=False, truncated_by=None,
            total_lines=total_lines, total_bytes=total_bytes,
            output_lines=total_lines, output_bytes=total_bytes,
        )

    # Collect from the back
    kept: list[str] = []
    byte_budget = max_bytes
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        line_bytes = _byte_len(line) + 1
        if len(kept) >= max_lines:
            break
        if byte_budget < line_bytes:
            break
        kept.insert(0, line)
        byte_budget -= line_bytes

    output = "\n".join(kept)
    output_bytes = _byte_len(output)

    truncated_by = "lines" if len(kept) >= max_lines else "bytes"
    hint = _build_hint(total_lines, total_bytes, len(kept), output_bytes)
    return TruncationResult(
        content=hint + output,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(kept),
        output_bytes=output_bytes,
    )


def truncate_string(
    content: str,
    max_chars: int = 5000,
    from_head: bool = True,
) -> str:
    """Simple string truncation with UTF-8 safety.

    Convenience function for tools that just need a char-limited slice.
    Returns the truncated string (no metadata).
    """
    if len(content) <= max_chars:
        return content
    if from_head:
        # Walk forward, don't split a multi-byte char
        cut = max_chars
        # Adjust to avoid splitting a surrogate pair or multi-byte char
        while cut > 0 and _is_continuation_byte(content, cut):
            cut -= 1
        return content[:cut] + "\n\n... (内容已截断)"
    else:
        start = len(content) - max_chars
        while start < len(content) and _is_continuation_byte(content, start):
            start += 1
        return "... (内容已截断) ...\n" + content[start:]


# ---- Internal helpers ----

def _byte_len(s: str) -> int:
    """Return the UTF-8 byte length of a string."""
    return len(s.encode("utf-8"))


def _is_continuation_byte(s: str, pos: int) -> bool:
    """Check if the position falls inside a multi-byte character.

    A Unicode code point encoded as multiple UTF-16 code units (surrogate pair)
    has the second unit in the range U+DC00..U+DFFF. If we're about to cut
    at a position that's a low surrogate, we need to back up.
    """
    if pos <= 0 or pos >= len(s):
        return False
    code = ord(s[pos])
    # Low surrogate: 0xDC00-0xDFFF
    return 0xDC00 <= code <= 0xDFFF


def _build_hint(
    total_lines: int, total_bytes: int,
    output_lines: int, output_bytes: int,
) -> str:
    """Build the truncation hint line."""
    return (
        f"\n[已截断: 原始 {total_lines} 行 / {_fmt_bytes(total_bytes)}, "
        f"保留 {output_lines} 行 / {_fmt_bytes(output_bytes)}]"
    )


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


# ---- Structured result truncation ----

# Fields that may contain long arrays in tool results
_ARRAY_FIELDS = ("holdings", "buckets", "snapshots", "assets", "results", "items", "news", "images")
# Fields that may contain long strings in tool results
_TEXT_FIELDS = ("text", "stdout", "stderr", "content", "output", "notes", "description", "prompt", "url")

DEFAULT_MAX_ARRAY_ITEMS = 50


def truncate_result(
    result: dict[str, Any],
    max_array_items: int = DEFAULT_MAX_ARRAY_ITEMS,
) -> dict[str, Any]:
    """Truncate large fields in a structured tool result dict.

    - Lists in known array fields are capped at max_array_items
    - Long strings in known text fields are truncated with truncate_head
    - Adds _truncated metadata when truncation occurs
    """
    if not isinstance(result, dict):
        return result

    for key in list(result.keys()):
        if key.startswith("_"):
            continue

        # Array truncation
        if key in _ARRAY_FIELDS and isinstance(result[key], list):
            items = result[key]
            if len(items) > max_array_items:
                result[f"_{key}_original_count"] = len(items)
                result[key] = items[:max_array_items]
                if "_truncated" not in result:
                    result["_truncated"] = True

        # Text truncation
        if key in _TEXT_FIELDS and isinstance(result[key], str):
            tr = truncate_head(result[key])
            if tr.truncated:
                result[key] = tr.content
                if "_truncated" not in result:
                    result["_truncated"] = True

    return result
