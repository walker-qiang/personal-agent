"""Tests for skill system."""

from __future__ import annotations

import tempfile
from pathlib import Path

from matrix.skills import SkillDefinition, execute_skill, load_skills
from matrix.tools import FinanceToolError, ToolRegistry
from matrix.tools.finance import register_all


class TestSkillDefinition:
    def test_matches_description(self):
        skill = SkillDefinition(
            name="test",
            title="异动诊断",
            description="对持仓数据进行异动诊断，识别异常变化并进行归因分析。",
        )
        assert skill.matches("最近有异动吗")  # "异动诊断" in "最近有异动吗" → True
        assert skill.matches("异动诊断")  # exact match
        assert not skill.matches("今天天气")

    def test_matches_title(self):
        skill = SkillDefinition(
            name="test",
            title="配置偏离检查",
            description="检查配置偏离度。",
        )
        assert skill.matches("配置偏离检查")  # exact title
        assert skill.matches("偏离度")  # "偏离度" in "配置偏离检查 检查配置偏离度。" → True

    def test_empty_no_match(self):
        skill = SkillDefinition(name="test", title="测试")
        assert not skill.matches("anything")

    def test_negation_rejects_match(self):
        """Negation words should prevent matching."""
        skill = SkillDefinition(
            name="anomaly-diagnosis",
            title="异动诊断",
            description="识别持仓异动并进行归因分析。",
        )
        assert not skill.matches("今天没有异动")
        assert not skill.matches("不是异动诊断")
        assert not skill.matches("不需要异动诊断")

    def test_negation_does_not_block_positive(self):
        """Positive queries without negation should still match."""
        skill = SkillDefinition(
            name="anomaly-diagnosis",
            title="异动诊断",
            description="识别持仓异动并进行归因分析。",
        )
        assert skill.matches("帮我做一下异动诊断")
        assert skill.matches("最近有异动")
        assert skill.matches("异动诊断")

    def test_negation_with_other_keywords(self):
        """Negation of one word should not block matching of another."""
        skill = SkillDefinition(
            name="portfolio-review",
            title="组合复盘",
            description="定期组合复盘，检查配置偏离。",
        )
        assert skill.matches("没有异动，但帮我做组合复盘")
        assert not skill.matches("不需要组合复盘")


class TestLoadSkills:
    def test_loads_from_skill_dirs(self):
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