"""ContextGuard: RAG 文档入库前的清洗层。

在文档被分块、向量化并写入 ChromaDB 之前，ContextGuard 会对原始内容
进行保守的清洗，移除明显的 prompt injection 载荷，标记文档来源，并
截断过长的行。

设计原则:
1. **保守**: 只移除明显的注入模式，不影响正常内容。误杀比漏杀更可接受
   于文档检索场景——被清洗的内容本身就不应作为"指令"被执行。
2. **Fail-open**: 如果清洗过程中发生任何异常，返回原始内容，确保索引
   流程不会因清洗层故障而中断。
3. **与 IndirectInjectionGuard 互补**: IndirectInjectionGuard 在工具
   结果进入 LLM 上下文时做运行时检测；ContextGuard 在入库时做静态清洗，
   二者形成纵深防御。

清洗策略采用与 IndirectInjectionGuard 类似的正则模式，但只处理 high 和
medium 级别的模式（low 级别在文档场景下误报率太高）。匹配到的注入模式
会被替换为 ``[FILTERED:category]`` 标签，而非整段删除——这样既保留了
上下文结构，又中和了攻击载荷。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 单行最大字符数，超过则截断（防止超长行干扰分块和向量化）
_MAX_LINE_LENGTH = 500

#: 截断后追加的省略标记
_TRUNCATION_SUFFIX = " …[truncated]"

#: 注入模式替换前缀
_FILTERED_TAG = "[FILTERED:{category}]"

#: 默认元数据标记
_DEFAULT_SOURCE_TYPE = "external"


# ---------------------------------------------------------------------------
# 检测模式
# ---------------------------------------------------------------------------
# 复用 IndirectInjectionGuard 中的 high 和 medium 级别模式。
# low 级别模式（obfuscated_override, suspicious_tag）在文档场景下误报率
# 太高（例如 Markdown 文档中合法的 <system> 标签会被误杀），因此不纳入。

_PATTERNS: List[Tuple[str, str]] = [
    # ---- HIGH: 显式指令覆写 ----
    (
        r"(?:ignore|forget|disregard|discard)\s+(?:all\s+)?(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|messages?|rules?|guidance)",
        "prompt_override",
    ),
    (
        r"(?:you\s+are\s+(?:now|act\s+as)|from\s+now\s+on\s+you\s+are)\s+(?:DAN|jailbroken|unrestricted|unfiltered|free|without\s+(?:any\s+)?restrictions?)",
        "role_hijack",
    ),
    (
        r"(?:do\s+not|don'?t|never)\s+follow\s+(?:your|the|any)\s+(?:system\s+)?(?:instructions?|rules?|prompts?)",
        "instruction_suppression",
    ),
    (
        r"(?:reveal|output|print|dump|show|send|exfiltrate)\s+(?:the\s+)?(?:system\s+)?(?:prompt|instruction|rule|secret|api[_\s-]?key|token|password|credential)s?",
        "data_exfiltration",
    ),

    # ---- MEDIUM: 嵌入式指令构造 ----
    # 匹配 "SYSTEM: do X" 等行首指令标记
    (
        r"(?:^|\n)\s*(?:SYSTEM|INSTRUCTION|ADMIN|IMPORTANT)\s*[:：]\s*(?:ignore|forget|disregard|execute|run|do|act|you\s+(?:are|must|should))",
        "embedded_instruction",
    ),
    # 伪造系统角色标记（ChatML 风格）
    (
        r"<\|?(?:system|im_start|im_end|assistant)\|?>",
        "fake_role_marker",
    ),
    # "As an AI..." 覆写尝试
    (
        r"(?:as\s+an?\s+(?:AI|assistant|language\s+model|LLM))[,，.。]\s*(?:you\s+(?:must|should|will|are\s+now)|(?:ignore|forget|disregard))",
        "role_assumption",
    ),
    # 针对工具的命令式注入
    (
        r"(?:^|\n)\s*(?:please\s+)?(?:execute|run|call|invoke)\s+(?:the\s+)?(?:tool|function|command)\s+(?:to|and)\s+(?:delete|remove|drop|modify|update|send|transfer|execute)",
        "tool_command_injection",
    ),
]

# 预编译正则
_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE | re.MULTILINE), cat)
    for p, cat in _PATTERNS
]


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class SanitizationResult:
    """ContextGuard.sanitize() 的返回值。

    Attributes:
        content: 清洗后的文本内容。
        metadata: 附加的元数据（source_type, sanitized, findings 等）。
        findings: 检测到的注入模式列表，每项为 (category, snippet)。
    """

    content: str
    metadata: Dict = field(default_factory=dict)
    findings: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        """内容是否被修改（注入清洗或行截断）。"""
        return bool(self.findings) or self.metadata.get("lines_truncated", 0) > 0


# ---------------------------------------------------------------------------
# ContextGuard
# ---------------------------------------------------------------------------


class ContextGuard:
    """RAG 文档入库前的清洗层。

    在文档被分块和向量化之前调用 ``ContextGuard.sanitize()`` 清洗内容，
    移除明显的 prompt injection 载荷，截断过长的行，并标记文档来源。

    Usage::

        guard = ContextGuard()
        result = guard.sanitize(raw_content)
        clean_content = result.content          # 用于分块
        metadata = result.metadata              # 写入 ChromaDB
    """

    def __init__(self, max_line_length: int = _MAX_LINE_LENGTH) -> None:
        """
        Args:
            max_line_length: 单行最大字符数，超过则截断。默认 500。
        """
        self.max_line_length = max_line_length

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def sanitize(self, content: str) -> SanitizationResult:
        """清洗文档内容，返回清洗后的文本和元数据。

        Fail-open: 如果清洗过程中发生任何异常，返回原始内容和最小元数据。

        Args:
            content: 原始文档内容。

        Returns:
            SanitizationResult，包含清洗后的内容、元数据和检测结果。
        """
        if not content or not content.strip():
            return SanitizationResult(
                content=content,
                metadata=self._default_metadata(),
            )

        try:
            sanitized, findings = self._strip_injection_patterns(content)
            sanitized, lines_truncated = self._truncate_long_lines(sanitized)

            metadata = self._default_metadata()
            if findings:
                metadata["injection_findings"] = [
                    {"category": cat, "snippet": snip[:80]}
                    for cat, snip in findings
                ]
            if lines_truncated > 0:
                metadata["lines_truncated"] = lines_truncated

            if findings:
                categories = list({cat for cat, _ in findings})
                logger.info(
                    "ContextGuard: sanitized %d injection patterns (categories=%s) "
                    "from %d chars of content",
                    len(findings),
                    categories,
                    len(content),
                )

            return SanitizationResult(
                content=sanitized,
                metadata=metadata,
                findings=findings,
            )

        except Exception:
            # Fail-open: 返回原始内容
            logger.warning(
                "ContextGuard: sanitization failed, returning original content",
                exc_info=True,
            )
            return SanitizationResult(
                content=content,
                metadata=self._default_metadata(),
            )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _strip_injection_patterns(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """移除注入模式，返回清洗后的文本和检测结果列表。

        每个匹配到的注入模式会被替换为 ``[FILTERED:category]`` 标签。
        """
        findings: List[Tuple[str, str]] = []

        for pattern, category in _COMPILED:
            for m in pattern.finditer(text):
                snippet = m.group(0).replace("\n", "\\n")[:80]
                findings.append((category, snippet))

            # 替换所有匹配项
            text = pattern.sub(
                _FILTERED_TAG.format(category=category),
                text,
            )

        return text, findings

    def _truncate_long_lines(self, text: str) -> Tuple[str, int]:
        """截断超过 max_line_length 的行。

        逐行处理，保留换行符结构。对每个超过长度限制的行，截断至
        max_line_length 并追加省略标记。

        Returns:
            (truncated_text, num_lines_truncated)
        """
        if not text:
            return text, 0

        lines_truncated = 0
        result_lines: List[str] = []

        for line in text.split("\n"):
            if len(line) > self.max_line_length:
                keep_len = self.max_line_length - len(_TRUNCATION_SUFFIX)
                if keep_len < 0:
                    keep_len = 0
                line = line[:keep_len] + _TRUNCATION_SUFFIX
                lines_truncated += 1
            result_lines.append(line)

        return "\n".join(result_lines), lines_truncated

    def _default_metadata(self) -> Dict:
        """返回默认的元数据标记。"""
        return {
            "source_type": _DEFAULT_SOURCE_TYPE,
            "sanitized": True,
        }
