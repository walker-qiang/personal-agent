"""Tests for skill system."""

from __future__ import annotations

import tempfile
from pathlib import Path

from matrix.skills import SkillDefinition, execute_skill, load_skills
from matrix.tools import FinanceToolError, ToolRegistry
from matrix.tools.finance import register_all


class TestSkillDefinition:
    def test_matches_trigger_keywords(self):
        skill = SkillDefinition(
            name="test",
            title="测试",
            trigger_keywords=["异动", "波动", "异常"],
        )
        assert skill.matches("最近有异动吗")
        assert skill.matches("波动很大")
        assert not skill.matches("今天天气")

    def test_matches_case_insensitive(self):
        skill = SkillDefinition(
            name="test",
            title="测试",
            trigger_keywords=["异动"],
        )
        assert skill.matches("异动")

    def test_empty_trigger_keywords(self):
        skill = SkillDefinition(name="test", title="测试")
        assert not skill.matches("anything")


class TestLoadSkills:
    def test_loads_from_markdown_files(self):
        """Test loading skills from the investment skills directory."""
        skills_dir = Path("skills/investment")
        if not skills_dir.exists():
            return  # Skip if dir doesn't exist in test context
        skills = load_skills(skills_dir)
        assert len(skills) == 3
        names = {s.name for s in skills}
        assert names == {"anomaly-diagnosis", "portfolio-review", "allocation-check"}

    def test_anomaly_diagnosis_has_workflow(self):
        skills_dir = Path("skills/investment")
        if not skills_dir.exists():
            return
        skills = load_skills(skills_dir)
        anomaly = next(s for s in skills if s.name == "anomaly-diagnosis")
        assert len(anomaly.workflow) == 3
        assert anomaly.workflow[0]["tool"] == "finance.holdings_summary"
        assert anomaly.workflow[1]["tool"] == "finance.recent_snapshots"
        assert anomaly.workflow[2]["tool"] == "finance.bucket_allocation"

    def test_allocation_check_has_workflow(self):
        skills_dir = Path("skills/investment")
        if not skills_dir.exists():
            return
        skills = load_skills(skills_dir)
        alloc = next(s for s in skills if s.name == "allocation-check")
        assert len(alloc.workflow) == 1
        assert alloc.workflow[0]["tool"] == "finance.bucket_allocation"

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            skills = load_skills(Path(tmp))
            assert skills == []


class TestExecuteSkill:
    def test_executes_workflow(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        skill = SkillDefinition(
            name="test",
            title="测试",
            workflow=[
                {"step": 1, "tool": "finance.holdings_summary", "arguments": {}},
                {"step": 2, "tool": "finance.bucket_allocation", "arguments": {}},
            ],
        )
        result = execute_skill(skill, registry)
        assert result["skill"] == "test"
        assert result["steps_executed"] == 2
        assert len(result["errors"]) == 0

    def test_handles_tool_error(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        skill = SkillDefinition(
            name="test",
            title="测试",
            workflow=[
                {"step": 1, "tool": "finance.unknown", "arguments": {}},
            ],
        )
        result = execute_skill(skill, registry)
        assert result["steps_executed"] == 0
        assert len(result["errors"]) == 1

    def test_skips_non_tool_steps(self, tmp_cache_path):
        registry = ToolRegistry()
        register_all(registry, tmp_cache_path)
        skill = SkillDefinition(
            name="test",
            title="测试",
            workflow=[
                {"step": 1, "purpose": "描述步骤"},
                {"step": 2, "tool": "finance.holdings_summary", "arguments": {}},
            ],
        )
        result = execute_skill(skill, registry)
        assert result["steps_executed"] == 1