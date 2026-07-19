"""Anti-hallucination: structured claim extraction + two-level verification.

Design:
  Level 1 - String matching (zero-cost): check if claim keywords appear in tool results.
  Level 2 - LLM verification (only for unverified claims): semantic cross-check.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("matrix.anti_hallucination")

# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """Single factual claim extracted from LLM output."""
    text: str
    claim_type: str = "general"  # numeric | date | price | name | event | general
    source_tool: str = ""
    stated_evidence: str = ""
    verified: bool = False
    matched_evidence: str = ""
    confidence: str = "unverified"  # verified | partial | unverified | contradicted


@dataclass
class VerificationResult:
    """Aggregated verification result."""
    claims: list[Claim] = field(default_factory=list)
    total: int = 0
    verified: int = 0
    partial: int = 0
    unverified: int = 0
    contradicted: int = 0
    no_citation: int = 0  # claims without a [来源: tool#N] citation
    summary: str = ""


# ── Step 1: Parse verification block from LLM output ─────────────────────────

_VERIF_BLOCK_RE = re.compile(
    r"\[VERIFICATION\]\s*(.*?)\s*\[/VERIFICATION\]", re.DOTALL | re.IGNORECASE
)

_CLAIM_RE = re.compile(
    r"\[CLAIM\]\s*(.*?)\s*\[/CLAIM\]", re.DOTALL | re.IGNORECASE
)
_EVIDENCE_RE = re.compile(
    r"\[EVIDENCE\]\s*(.*?)\s*\[/EVIDENCE\]", re.DOTALL | re.IGNORECASE
)
_SOURCE_RE = re.compile(
    r"\[SOURCE\]\s*(.*?)\s*\[/SOURCE\]", re.DOTALL | re.IGNORECASE
)


def parse_verification_block(text: str) -> list[Claim]:
    """Parse [VERIFICATION]...[/VERIFICATION] block from LLM output.

    Expected format:
    [CLAIM] claim text [/CLAIM]
    [EVIDENCE] supporting text [/EVIDENCE]
    [SOURCE] tool_name#N [/SOURCE]
    """
    m = _VERIF_BLOCK_RE.search(text)
    if not m:
        return []

    block = m.group(1)
    # Split by claim boundary tags to get interleaved segments
    segments = re.split(r"\[/?CLAIM\]", block, flags=re.IGNORECASE)
    # segments: [0]=before first CLAIM, [1]=claim1, [2]=meta1, [3]=claim2, [4]=meta2, ...
    # Each claim is followed by its evidence+source in the next segment

    claims = []
    for i in range(1, len(segments), 2):  # odd indices = claim text
        claim_text = segments[i].strip()
        if not claim_text:
            continue

        evidence = ""
        source = ""
        # Meta is in the next segment (if exists)
        if i + 1 < len(segments):
            meta = segments[i + 1]
            ev_m = _EVIDENCE_RE.search(meta)
            if ev_m:
                evidence = ev_m.group(1).strip()
            src_m = _SOURCE_RE.search(meta)
            if src_m:
                source = src_m.group(1).strip()

        if claim_text:
            claims.append(Claim(
                text=claim_text,
                claim_type=_classify_claim_type(claim_text),
                source_tool=source,
                stated_evidence=evidence,
            ))

    return claims


# ── Step 2: Heuristic claim extraction (fallback when no verification block) ──

_NUMERIC_PATTERNS = [
    # Chinese
    (re.compile(r"(\d+(?:\.\d+)?)\s*(亿|万|千|百)?\s*(元|美元|港元|日元|欧元|英镑|%|％|倍|个)"), "numeric"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(亿|万)\s*(辆|台|人|次)"), "numeric"),
    # English
    (re.compile(r"(\d+(?:\.\d+)?)\s*(billion|million|thousand|percent|%)", re.IGNORECASE), "numeric"),
    # Price
    (re.compile(r"(\d+(?:\.\d+)?)\s*港[币元]|HK\$\s*(\d+(?:\.\d+)?)", re.IGNORECASE), "price"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*美[元金]|US\$\s*(\d+(?:\.\d+)?)", re.IGNORECASE), "price"),
    # Date
    (re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"), "date"),
    (re.compile(r"(\d{4}-\d{2}-\d{2})"), "date"),
    # Name
    (re.compile(r"(?:[A-Z][a-z]+\s){1,2}[A-Z][a-z]+"), "name"),
]


def _classify_claim_type(text: str) -> str:
    """Classify what kind of factual claim this is."""
    for pat, ctype in _NUMERIC_PATTERNS:
        if pat.search(text):
            return ctype
    return "general"


def extract_claims_heuristic(text: str) -> list[Claim]:
    """Extract factual claims from free text when no verification block exists.

    Uses regex patterns to find numbers, dates, prices, and names.
    """
    # Remove markdown formatting and code blocks
    clean = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    clean = re.sub(r"`[^`]+`", "", clean)

    claims = []
    for pat, ctype in _NUMERIC_PATTERNS:
        for m in pat.finditer(clean):
            claim_text = m.group(0).strip()
            # Dedup
            if not any(c.text == claim_text for c in claims):
                claims.append(Claim(
                    text=claim_text,
                    claim_type=ctype,
                ))

    return claims


# ── Step 3: Level 1 - String matching verification ───────────────────────────

def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace, normalize spaces."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _tool_results_to_text(tool_results: list[dict[str, Any]]) -> str:
    """Flatten all tool results into a single searchable text."""
    parts = []
    for tr in tool_results:
        name = tr.get("name", "")
        if tr.get("result"):
            result_str = json.dumps(tr["result"], ensure_ascii=False) if isinstance(tr["result"], (dict, list)) else str(tr["result"])
        else:
            result_str = tr.get("error", "")
        parts.append(f"[tool:{name}] {result_str}")
    return "\n".join(parts)


def _tool_results_by_index(tool_results: list[dict[str, Any]]) -> dict[str, str]:
    """Build a map of 'tool_name#N' -> result text for the N-th call of each tool.
    N starts at 1. Example: 'news_search#1' -> '{...json...}', 'web_search#1' -> '...html...'
    """
    indexed: dict[str, str] = {}
    tool_counts: dict[str, int] = {}
    for tr in tool_results:
        name = tr.get("name", "")
        tool_counts[name] = tool_counts.get(name, 0) + 1
        key = f"{name}#{tool_counts[name]}"
        if tr.get("result"):
            result_str = json.dumps(tr["result"], ensure_ascii=False) if isinstance(tr["result"], (dict, list)) else str(tr["result"])
        else:
            result_str = tr.get("error", "")
        indexed[key] = result_str
    return indexed


def _extract_key_tokens(text: str) -> list[str]:
    """Extract key tokens from text for fuzzy matching.

    Focus on: numbers, proper nouns, dates, and named entities.
    """
    # Extract numbers (including decimals, percentages)
    numbers = re.findall(r"\d+(?:\.\d+)?%?", text)
    # Extract proper nouns (Chinese: 2+ chars, English: capitalized words)
    proper_nouns = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    proper_nouns += re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)?", text)
    # Extract dates
    dates = re.findall(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}", text)
    return numbers + proper_nouns + dates


_SOURCE_INDEX_RE = re.compile(r"^(.+?)#(\d+)$")


def _parse_source_index(source: str) -> tuple[str, int] | None:
    """Parse 'news_search#2' → ('news_search', 2). Returns None if no index."""
    m = _SOURCE_INDEX_RE.match(source.strip())
    if m:
        return m.group(1), int(m.group(2))
    return None


def _get_target_text_for_claim(claim: Claim, tool_results: list[dict[str, Any]]) -> str | None:
    """Get the specific tool result text for a claim's cited source.

    If the claim has a source_tool with index (e.g. 'news_search#2'), returns
    only the text of the 2nd news_search call. If no index, returns all results
    of that tool. If no source_tool at all, returns None (citation required).
    """
    if not claim.source_tool:
        return None

    indexed = _tool_results_by_index(tool_results)
    # Try exact match first (tool_name#N)
    if claim.source_tool in indexed:
        return indexed[claim.source_tool]

    # Try parsing as index format
    parsed = _parse_source_index(claim.source_tool)
    if parsed:
        tool_name, idx = parsed
        key = f"{tool_name}#{idx}"
        if key in indexed:
            return indexed[key]

    # Fallback: no index → match against all calls of that tool
    parts = []
    for tr in tool_results:
        if tr.get("name", "") == claim.source_tool:
            if tr.get("result"):
                parts.append(json.dumps(tr["result"], ensure_ascii=False) if isinstance(tr["result"], (dict, list)) else str(tr["result"]))
    return "\n".join(parts) if parts else None


def verify_claim_string_match(claim: Claim, tool_results: list[dict[str, Any]]) -> bool:
    """Level 1: Check if claim evidence/keywords appear in tool results.

    Uses citation-based scoping: if the claim cites a specific tool call
    (e.g. news_search#2), verification is limited to that call's result.

    Returns True if match found, False otherwise.
    """
    if not tool_results:
        return False

    # Citation-based scoping: match only against the cited source
    target_text = _get_target_text_for_claim(claim, tool_results)
    if target_text is None:
        # No source cited — cannot verify
        claim.confidence = "unverified"
        return False

    norm_target = _normalize(target_text)

    # Strategy 1: Check stated evidence (if provided)
    if claim.stated_evidence:
        if _normalize(claim.stated_evidence[:50]) in norm_target:
            claim.matched_evidence = claim.stated_evidence[:200]
            claim.verified = True
            claim.confidence = "verified"
            return True

    # Strategy 2: Check key tokens from claim text
    key_tokens = _extract_key_tokens(claim.text)
    if not key_tokens:
        return False

    matched = 0
    matched_evidence_parts = []
    for token in key_tokens:
        if _normalize(token) in norm_target:
            matched += 1
            idx = norm_target.find(_normalize(token))
            if idx >= 0:
                start = max(0, idx - 30)
                end = min(len(norm_target), idx + len(token) + 30)
                matched_evidence_parts.append(norm_target[start:end])

    match_ratio = matched / len(key_tokens) if key_tokens else 0

    if match_ratio >= 0.6:
        claim.matched_evidence = " | ".join(matched_evidence_parts[:3])[:300]
        claim.verified = True
        claim.confidence = "verified"
        return True
    elif match_ratio >= 0.3:
        claim.matched_evidence = " | ".join(matched_evidence_parts[:2])[:200]
        claim.confidence = "partial"
        return False
    else:
        claim.confidence = "unverified"
        return False


# ── Step 4: Level 2 - LLM verification ───────────────────────────────────────

_LLM_VERIFY_PROMPT = """You are a fact-checker. Given a factual claim and tool search results, determine if the claim is supported by the evidence.

Respond with EXACTLY one word:
- SUPPORTED: The claim is directly supported by the tool results
- PARTIALLY: Some parts of the claim are supported, but not all
- CONTRADICTED: The tool results contradict the claim
- NO_EVIDENCE: The tool results do not contain relevant information

Claim: {claim}

Tool Results:
{tool_results}

Your verdict (one word only):"""


def verify_claims_with_llm(
    unverified: list[Claim],
    tool_results: list[dict[str, Any]],
    llm,  # LLMClient
) -> None:
    """Level 2: Use LLM to verify claims that failed string matching.

    Modifies claims in-place (sets confidence and matched_evidence).
    """
    if not unverified or not tool_results:
        return

    results_text = _tool_results_to_text(tool_results)
    # Truncate to avoid token limits
    if len(results_text) > 4000:
        results_text = results_text[:4000] + "\n...[truncated]"

    for claim in unverified:
        try:
            prompt = _LLM_VERIFY_PROMPT.format(
                claim=claim.text,
                tool_results=results_text,
            )
            verdict = llm.complete("", [{"role": "user", "content": prompt}], temperature=0.0).strip().upper()

            if "SUPPORTED" in verdict:
                claim.confidence = "verified"
                claim.verified = True
            elif "PARTIALLY" in verdict:
                claim.confidence = "partial"
            elif "CONTRADICTED" in verdict:
                claim.confidence = "contradicted"
            else:
                claim.confidence = "unverified"
        except Exception as e:
            logger.warning("LLM claim verification failed: %s", e)
            claim.confidence = "unverified"


# ── Step 5: Main verification entry point ────────────────────────────────────

def verify_all_claims(
    text: str,
    tool_results: list[dict[str, Any]],
    llm=None,
) -> VerificationResult:
    """Full verification pipeline.

    1. Parse [VERIFICATION] block from LLM output
    2. Fallback to heuristic extraction if no block
    3. Level 1: string matching
    4. Level 2: LLM verification for unverified claims
    5. Return VerificationResult
    """
    # Parse claims — only from explicit [VERIFICATION] block.
    # No heuristic fallback: regex-based "claim extraction" from free text
    # cannot distinguish factual claims from context references, leading to
    # false positives (e.g., dates, proper nouns in suggestions).
    claims = parse_verification_block(text)

    if not claims:
        return VerificationResult()

    # Level 1: string matching
    for claim in claims:
        verify_claim_string_match(claim, tool_results)

    # Level 2: LLM verification for unverified
    unverified = [c for c in claims if c.confidence == "unverified"]
    if unverified and llm is not None:
        verify_claims_with_llm(unverified, tool_results, llm)

    # Build result
    no_citation_count = sum(1 for c in claims if not c.source_tool)
    result = VerificationResult(
        claims=claims,
        total=len(claims),
        verified=sum(1 for c in claims if c.confidence == "verified"),
        partial=sum(1 for c in claims if c.confidence == "partial"),
        unverified=sum(1 for c in claims if c.confidence == "unverified"),
        contradicted=sum(1 for c in claims if c.confidence == "contradicted"),
        no_citation=no_citation_count,
    )

    if result.verified == result.total:
        result.summary = "全部验证通过"
    elif result.verified + result.partial >= result.total * 0.7:
        result.summary = f"大部分验证通过 ({result.verified}/{result.total})"
    elif result.verified + result.partial >= result.total * 0.3:
        result.summary = f"部分验证通过 ({result.verified}/{result.total})，{result.unverified} 条未验证"
    else:
        result.summary = f"大部分无法验证 ({result.verified}/{result.total})，{result.contradicted} 条矛盾"

    logger.info(
        "anti-hallucination: total=%d verified=%d partial=%d unverified=%d contradicted=%d no_citation=%d",
        result.total, result.verified, result.partial,
        result.unverified, result.contradicted, result.no_citation,
    )
    return result


# ── Step 6: Output building ──────────────────────────────────────────────────

def _mark_unverified_claims(answer: str, result: VerificationResult) -> str:
    """Append [unverified] markers to unverified claims in the answer."""
    modified = answer
    for claim in result.claims:
        if claim.confidence == "unverified" and claim.text in modified:
            modified = modified.replace(
                claim.text,
                f"{claim.text} [未验证]",
                1,
            )
        elif claim.confidence == "contradicted" and claim.text in modified:
            modified = modified.replace(
                claim.text,
                f"{claim.text} [与来源不符]",
                1,
            )
    return modified


def _add_confidence_warning(answer: str, result: VerificationResult) -> str:
    """Prepend a confidence warning to the answer."""
    warning = (
        f"\n\n---\n"
        f"**验证报告**：{result.summary}\n"
    )
    if result.no_citation > 0:
        no_cite_claims = [c for c in result.claims if not c.source_tool]
        warning += "\n**未标注来源的声明**（请补充 [来源: tool#N]）：\n"
        for c in no_cite_claims:
            warning += f"- {c.text}\n"
    if result.contradicted > 0:
        contradicted_claims = [c for c in result.claims if c.confidence == "contradicted"]
        warning += "\n**与来源矛盾的声明**：\n"
        for c in contradicted_claims:
            warning += f"- {c.text}\n"
    if result.unverified > 0:
        unverified_claims = [c for c in result.claims if c.confidence == "unverified"]
        warning += "\n**无法验证的声明**：\n"
        for c in unverified_claims:
            warning += f"- {c.text}\n"
    warning += "\n---\n\n"
    return warning + answer


def _build_safe_response(result: VerificationResult) -> str:
    """Build a safe response when most claims are unverified or contradicted."""
    parts = ["当前搜索结果无法充分验证以下信息："]

    if result.no_citation > 0:
        parts.append("\n**未标注来源的声明**：")
        for c in result.claims:
            if not c.source_tool:
                parts.append(f"- {c.text}")

    if result.contradicted > 0:
        parts.append("\n**与来源矛盾的声明**：")
        for c in result.claims:
            if c.confidence == "contradicted":
                parts.append(f"- {c.text}")

    if result.unverified > 0:
        parts.append("\n**无法验证的声明**：")
        for c in result.claims:
            if c.confidence == "unverified":
                parts.append(f"- {c.text}")

    parts.append("\n建议缩小查询范围或更换搜索词后重试。")
    return "\n".join(parts)


def build_verified_output(
    original_answer: str,
    verification: VerificationResult,
) -> str:
    """Build the final output based on verification results.

    Strategy:
    - 100% verified: return original answer as-is
    - >= 70% verified (incl partial): mark unverified claims
    - >= 30% verified (incl partial): add confidence warning
    - < 30%: safe response (but if there are partial claims, show them with warning)
    """
    if verification.total == 0:
        return original_answer

    # Effective verified: verified + partial at 0.5 weight
    effective_verified = verification.verified + verification.partial * 0.5
    verified_ratio = effective_verified / verification.total

    if verified_ratio >= 1.0:
        # All verified - return as-is (remove verification block for cleanliness)
        clean = _VERIF_BLOCK_RE.sub("", original_answer).strip()
        return clean

    if verified_ratio >= 0.7:
        # Most verified - mark unverified
        clean = _VERIF_BLOCK_RE.sub("", original_answer).strip()
        return _mark_unverified_claims(clean, verification)

    # < 70% — never discard the answer. The LLM's output is the best we have;
    # a post-processing pipeline cannot reliably judge correctness better than
    # the LLM that generated it. Always return the answer with a warning.
    clean = _VERIF_BLOCK_RE.sub("", original_answer).strip()
    return _add_confidence_warning(clean, verification)


def handle_verification_failure(
    original_answer: str,
    verification: VerificationResult,
) -> str:
    """Handle verification failure - decide whether to return original or safe response."""
    if verification.total == 0 or verification.verified > 0:
        return original_answer

    if verification.contradicted > 0:
        return _build_safe_response(verification)

    # All unverified but no contradictions - return original with warning
    return _add_confidence_warning(original_answer, verification)