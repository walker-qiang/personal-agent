"""Tests for verification tag stripping — ensures [VERIFICATION] and related
tags NEVER leak to user-facing output, regardless of which code path produced them.

This test file was created after the bug repeatedly recurred across different
output paths. It tests the _strip_all_verification_tags function directly
and validates the safety-net pattern used in _service.py.
"""

from __future__ import annotations

import pytest

from matrix.orchestration.anti_hallucination import (
    _strip_all_verification_tags,
    _VERIF_BLOCK_RE,
    _UNCLOSED_VERIF_RE,
    _LOOSE_TAG_RE,
)


# ---- All tag formats that should be stripped --------------------------------

class TestStripAllVerificationTags:
    """Verify that EVERY possible tag format is stripped."""

    def test_paired_verification_block(self):
        text = "Answer here.\n[VERIFICATION]\n[CLAIM]test[/CLAIM]\n[/VERIFICATION]"
        result = _strip_all_verification_tags(text)
        assert "[VERIFICATION]" not in result
        assert "[CLAIM]" not in result
        assert "Answer here" in result

    def test_unclosed_verification_block(self):
        """The bug that caused the latest leak: unclosed [VERIFICATION]."""
        text = "Answer here.\n[VERIFICATION]\n[CLAIM]test[/CLAIM]\n[EVIDENCE]quote[/EVIDENCE]"
        result = _strip_all_verification_tags(text)
        assert "[VERIFICATION]" not in result
        assert "[CLAIM]" not in result
        assert "[EVIDENCE]" not in result
        assert "Answer here" in result

    def test_loose_claim_tags(self):
        text = "Answer.\n[CLAIM]something[/CLAIM]\nMore text."
        result = _strip_all_verification_tags(text)
        assert "[CLAIM]" not in result
        assert "[/CLAIM]" not in result
        assert "Answer" in result
        assert "More text" in result

    def test_loose_evidence_tags(self):
        text = "Answer.\n[EVIDENCE]evidence text[/EVIDENCE]"
        result = _strip_all_verification_tags(text)
        assert "[EVIDENCE]" not in result
        assert "[/EVIDENCE]" not in result

    def test_loose_source_tags(self):
        text = "Answer.\n[SOURCE]web_fetch[/SOURCE]"
        result = _strip_all_verification_tags(text)
        assert "[SOURCE]" not in result
        assert "[/SOURCE]" not in result

    def test_confidence_tags(self):
        text = "Answer.\n[CONFIDENCE]0.85[/CONFIDENCE]"
        result = _strip_all_verification_tags(text)
        assert "[CONFIDENCE]" not in result
        assert "[/CONFIDENCE]" not in result

    def test_all_tags_together(self):
        """The exact pattern from the user's screenshot."""
        text = """截至2026年7月22日，詹姆斯的情况如下：

[VERIFICATION]
[CLAIM]詹姆斯已确认离开湖人[/CLAIM]
[EVIDENCE]LeBron James has opted out of his contract[/EVIDENCE]
[SOURCE]web_fetch[/SOURCE]
[CLAIM]尚未做出最终决定[/CLAIM]
[EVIDENCE]Rich Paul said no timeline[/EVIDENCE]
[SOURCE]web_fetch[/SOURCE]
[/VERIFICATION]"""
        result = _strip_all_verification_tags(text)
        assert "[VERIFICATION]" not in result
        assert "[CLAIM]" not in result
        assert "[EVIDENCE]" not in result
        assert "[SOURCE]" not in result
        assert "截至2026年7月22日" in result

    def test_unclosed_tags_from_screenshot(self):
        """Unclosed tags matching the user's screenshot exactly."""
        text = """截至2026年7月22日，詹姆斯的情况如下：

[VERIFICATION]
[CLAIM]詹姆斯已确认离开湖人[/CLAIM]
[EVIDENCE]LeBron James has opted out[/EVIDENCE]
[SOURCE]web_fetch[/SOURCE]
[CLAIM]尚未做出最终决定[/CLAIM]
[EVIDENCE]no timeline yet[/EVIDENCE]
[SOURCE]web_fetch[/SOURCE]"""
        result = _strip_all_verification_tags(text)
        assert "[VERIFICATION]" not in result
        assert "[CLAIM]" not in result
        assert "[EVIDENCE]" not in result
        assert "[SOURCE]" not in result

    def test_case_insensitive(self):
        text = "Answer.\n[verification]\n[claim]test[/claim]\n[/verification]"
        result = _strip_all_verification_tags(text)
        assert "[verification]" not in result.lower()
        assert "[claim]" not in result.lower()

    def test_clean_text_unchanged(self):
        text = "This is a clean answer with no tags."
        result = _strip_all_verification_tags(text)
        assert result == text

    def test_empty_string(self):
        assert _strip_all_verification_tags("") == ""

    def test_only_tags(self):
        text = "[VERIFICATION][CLAIM]x[/CLAIM][/VERIFICATION]"
        result = _strip_all_verification_tags(text)
        assert result == ""

    def test_excessive_blank_lines_cleaned(self):
        text = "Answer.\n\n\n\n\n[VERIFICATION]x[/VERIFICATION]\n\n\n\nMore."
        result = _strip_all_verification_tags(text)
        assert "\n\n\n" not in result  # max 2 consecutive newlines

    def test_nested_brackets_preserved(self):
        """Normal markdown brackets should NOT be stripped."""
        text = "Use [code] for formatting. See [1] for reference."
        result = _strip_all_verification_tags(text)
        assert "[code]" in result
        assert "[1]" in result


# ---- Safety net pattern (as used in _service.py) ---------------------------

class TestSafetyNetPattern:
    """Test the exact safety net pattern used in _service.py.

    This validates that calling _strip_all_verification_tags on ANY
    final_answer before streaming to user will catch all leaks.
    """

    @pytest.mark.parametrize("tag", [
        "[VERIFICATION]",
        "[/VERIFICATION]",
        "[CLAIM]",
        "[/CLAIM]",
        "[EVIDENCE]",
        "[/EVIDENCE]",
        "[SOURCE]",
        "[/SOURCE]",
        "[CONFIDENCE]",
        "[/CONFIDENCE]",
    ])
    def test_single_tag_stripped(self, tag):
        text = f"Answer. {tag} More text."
        result = _strip_all_verification_tags(text)
        assert tag not in result

    @pytest.mark.parametrize("text", [
        # Paired block
        "OK.\n[VERIFICATION]\nstuff\n[/VERIFICATION]",
        # Unclosed block
        "OK.\n[VERIFICATION]\nstuff without closing",
        # Loose tags
        "OK.\n[CLAIM]x[/CLAIM]\n[EVIDENCE]y[/EVIDENCE]",
        # Mixed
        "OK.\n[VERIFICATION]\n[CLAIM]x[/CLAIM]\nno closing verif",
    ])
    def test_all_patterns_caught(self, text):
        result = _strip_all_verification_tags(text)
        # No verification-related tag should survive
        for tag in ["VERIFICATION", "CLAIM", "EVIDENCE", "SOURCE", "CONFIDENCE"]:
            assert f"[{tag}]" not in result
            assert f"[/{tag}]" not in result
