"""Tests for role system."""

from __future__ import annotations

from matrix.role import INVESTMENT_ANALYST, RoleDefinition


class TestRoleDefinition:
    def test_fields(self):
        role = RoleDefinition(
            id="test-role",
            name="测试角色",
            persona="测试人格",
            expertise=["领域1"],
            tools=["finance.*"],
            skills=["test-skill"],
            output_constraints=["约束1"],
            safety_rules=["规则1"],
        )
        assert role.id == "test-role"
        assert role.name == "测试角色"
        assert len(role.expertise) == 1
        assert len(role.safety_rules) == 1

    def test_to_system_prompt(self):
        role = RoleDefinition(
            id="test",
            name="测试",
            persona="测试人格描述",
            expertise=["领域A"],
            safety_rules=["只读"],
            output_constraints=["中文回答"],
        )
        prompt = role.to_system_prompt()
        assert "测试" in prompt
        assert "测试人格描述" in prompt
        assert "领域A" in prompt
        assert "只读" in prompt
        assert "中文回答" in prompt

    def test_empty_fields_produce_no_sections(self):
        role = RoleDefinition(id="test", name="Test", persona="P")
        prompt = role.to_system_prompt()
        assert "专业领域" not in prompt
        assert "安全规则" not in prompt
        assert "输出约束" not in prompt


class TestInvestmentAnalyst:
    def test_has_required_fields(self):
        assert INVESTMENT_ANALYST.id == "investment-analyst"
        assert INVESTMENT_ANALYST.name == "投资分析员"
        assert len(INVESTMENT_ANALYST.expertise) == 5
        assert len(INVESTMENT_ANALYST.skills) == 3
        assert "anomaly-diagnosis" in INVESTMENT_ANALYST.skills
        assert "portfolio-review" in INVESTMENT_ANALYST.skills
        assert "allocation-check" in INVESTMENT_ANALYST.skills

    def test_safety_rules_are_read_only(self):
        rules = INVESTMENT_ANALYST.safety_rules
        assert any("只读" in r for r in rules)
        assert any("严禁" in r for r in rules)

    def test_system_prompt_is_chinese(self):
        prompt = INVESTMENT_ANALYST.to_system_prompt()
        assert "投资分析员" in prompt
        assert "只读" in prompt