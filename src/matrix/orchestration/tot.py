"""Tree of Thoughts (ToT) / Language Agent Tree Search (LATS) module.

设计理念:
    传统 ReAct 是单路径搜索: 每一步只选一个动作执行.
    ToT/LATS 引入多路径搜索: 每一步生成 N 个候选动作, 评估后选最优.

核心组件:
    ┌──────────────────────────────────────────────────────┐
    │  ToTNode                                              │
    │  树节点: state + action + value + visits + children   │
    ├──────────────────────────────────────────────────────┤
    │  UCB1Selector                                         │
    │  选择策略: value/visits + c * sqrt(ln(N)/n)           │
    │  平衡探索 (exploration) 与利用 (exploitation)         │
    ├──────────────────────────────────────────────────────┤
    │  TreeSearchEngine                                     │
    │  搜索引擎: 生成候选 → 评估 → 选择 → 执行 → 回溯       │
    ├──────────────────────────────────────────────────────┤
    │  ToTEvaluator                                         │
    │  评估器: LLM 打分 + 启发式规则                         │
    └──────────────────────────────────────────────────────┘

与编排管道的集成:
    1. Commander 规划阶段: 生成多个候选计划, 评估后选最优
    2. ReAct 执行阶段: 每步生成候选工具调用, 评估后执行最优
    3. Reflection 阶段: 如果结果不满意, 回溯尝试替代路径

设计权衡 (个人使用场景):
    - 不实现完整 LATS (需要大量 LLM 调用, 成本过高)
    - 采用 ToT-lite: 仅在关键决策点分支 (不是每步都分支)
    - 分支数限制在 2-3, 总节点数限制在 10 以内
    - LLM 不可用时退化为标准 ReAct (单路径)
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.protocol import LLMClient

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────

_DEFAULT_BRANCH_FACTOR = 3  # 每个节点生成的候选数
_DEFAULT_MAX_DEPTH = 4       # 树的最大深度
_DEFAULT_MAX_NODES = 10      # 树的最大节点数
_UCB_EXPLORATION_C = 1.414   # sqrt(2), UCB1 标准探索常数
_MIN_VALUE_THRESHOLD = 0.3   # 低于此值的候选被剪枝

# ── 评估 Prompt ──────────────────────────────────────────────────────────

_EVALUATE_ACTION_SYSTEM = """你是一个动作评估器。评估给定的动作对于完成用户任务的质量.

评估维度:
1. 相关性: 动作是否直接指向用户目标 (0-1)
2. 可行性: 动作是否可执行, 参数是否合理 (0-1)
3. 效率: 是否是最优路径, 还是绕了弯路 (0-1)
4. 信息增益: 能否带来新的有用信息 (0-1)

返回 JSON:
{{
  "score": 0.0-1.0,  // 综合评分
  "reasoning": "简短理由"
}}

只返回 JSON, 不要其他文字."""

_EVALUATE_PLAN_SYSTEM = """你是一个计划评估器。评估给定计划完成用户任务的质量.

用户任务: {task}
候选计划:
{plan}

评估维度:
1. 完整性: 是否覆盖了完成任务所需的所有步骤 (0-1)
2. 顺序合理性: 步骤之间的依赖关系是否正确 (0-1)
3. 效率: 是否存在冗余步骤 (0-1)
4. 可行性: 每步是否可执行 (0-1)

返回 JSON:
{{
  "score": 0.0-1.0,
  "reasoning": "简短理由",
  "missing_steps": ["缺失的步骤描述"]
}}

只返回 JSON, 不要其他文字."""

_GENERATE_ALTERNATIVES_SYSTEM = """你是一个动作生成器. 为用户任务生成 {n} 个不同的下一步动作候选.

用户任务: {task}
当前状态 (已执行的步骤和结果):
{context}

可用工具:
{tools}

要求:
1. 每个候选必须是不同的策略 (不同的工具或不同的参数)
2. 候选之间不应重复
3. 每个候选附带简短理由

返回 JSON:
{{
  "candidates": [
    {{
      "action": "工具名称",
      "arguments": {{}},
      "reasoning": "为什么选择这个动作"
    }}
  ]
}}

只返回 JSON, 不要其他文字."""


# ── 数据结构 ─────────────────────────────────────────────────────────────


@dataclass
class ToTNode:
    """Tree of Thoughts 的树节点.

    Attributes:
        node_id: 唯一标识.
        action: 导致此节点的动作描述.
        arguments: 动作的参数.
        parent_id: 父节点 ID (根节点为 None).
        children_ids: 子节点 ID 列表.
        value: 此节点的评估值 (0-1).
        visits: 被访问的次数.
        depth: 在树中的深度 (根节点为 0).
        status: 节点状态.
        result: 执行此动作后的结果.
        reasoning: 选择此动作的理由.
    """
    node_id: str
    action: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    value: float = 0.0
    visits: int = 0
    depth: int = 0
    status: str = "pending"  # pending, exploring, evaluated, pruned, completed
    result: Any = None
    reasoning: str = ""


@dataclass
class ToTResult:
    """Tree of Thoughts 搜索结果."""
    best_path: list[ToTNode] = field(default_factory=list)
    total_nodes: int = 0
    total_evaluations: int = 0
    pruned_count: int = 0
    best_value: float = 0.0
    used_tot: bool = False  # 是否实际使用了 ToT (False = 退化为单路径)


# ── UCB1 选择器 ──────────────────────────────────────────────────────────


class UCB1Selector:
    """Upper Confidence Bound 1 选择策略.

    UCB1 公式: UCB = Q(n) + c * sqrt(ln(N) / n)

    其中:
        Q(n) = 节点的平均价值
        N = 父节点的总访问次数
        n = 当前节点的访问次数
        c = 探索常数 (默认 sqrt(2))

    平衡策略:
    - 高 Q 值的节点更容易被选中 (exploitation)
    - 低访问次数的节点获得探索加成 (exploration)
    - 当所有节点访问次数相同时, 退化为贪心选择
    """

    def __init__(self, exploration_c: float = _UCB_EXPLORATION_C) -> None:
        self.c = exploration_c

    def select(
        self,
        candidates: list[ToTNode],
        parent_visits: int = 0,
    ) -> ToTNode | None:
        """从候选节点中选择 UCB1 值最高的节点.

        Args:
            candidates: 待选择的候选节点列表.
            parent_visits: 父节点的总访问次数 (用于计算探索项).

        Returns:
            UCB1 值最高的节点, 或 None (候选为空).
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        best_node = candidates[0]
        best_ucb = -float("inf")

        for node in candidates:
            if node.status == "pruned":
                continue

            if node.visits == 0:
                # 未访问的节点给予无限探索值 (优先探索)
                ucb = float("inf")
            else:
                avg_value = node.value / node.visits
                if parent_visits > 0:
                    exploration = self.c * math.sqrt(
                        math.log(parent_visits) / node.visits
                    )
                else:
                    exploration = self.c
                ucb = avg_value + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_node = node

        return best_node


# ── ToT 评估器 ───────────────────────────────────────────────────────────


class ToTEvaluator:
    """使用 LLM 和启发式规则评估动作/计划的质量.

    LLM 不可用时退化为简单启发式评分.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def evaluate_action(
        self,
        task: str,
        action: str,
        arguments: dict[str, Any],
        context: str = "",
    ) -> tuple[float, str]:
        """评估单个动作的质量.

        Returns:
            (score, reasoning) — 分数 0-1, 理由.
        """
        if self._llm is None:
            return self._heuristic_score(task, action, arguments)

        try:
            action_desc = f"工具: {action}\n参数: {json.dumps(arguments, ensure_ascii=False)}"
            if context:
                action_desc += f"\n上下文: {context[:500]}"

            result = self._llm.complete_json(
                _EVALUATE_ACTION_SYSTEM,
                [{"role": "user", "content": f"用户任务: {task}\n\n待评估动作:\n{action_desc}"}],
                temperature=0.0,
            )

            score = float(result.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reasoning = str(result.get("reasoning", ""))[:200]

            return score, reasoning

        except Exception as exc:
            logger.warning("ToT evaluator: LLM failed, using heuristic: %s", exc)
            return self._heuristic_score(task, action, arguments)

    def evaluate_plan(
        self,
        task: str,
        plan: list[dict[str, Any]],
    ) -> tuple[float, str, list[str]]:
        """评估计划的质量.

        Returns:
            (score, reasoning, missing_steps) — 分数, 理由, 缺失步骤.
        """
        if self._llm is None:
            return self._heuristic_plan_score(task, plan)

        try:
            plan_str = json.dumps(plan, ensure_ascii=False, indent=2)
            result = self._llm.complete_json(
                _EVALUATE_PLAN_SYSTEM.format(task=task, plan=plan_str),
                [{"role": "user", "content": task}],
                temperature=0.0,
            )

            score = float(result.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            reasoning = str(result.get("reasoning", ""))[:200]
            missing_steps = [
                str(s) for s in result.get("missing_steps", [])
                if isinstance(s, str)
            ]

            return score, reasoning, missing_steps

        except Exception as exc:
            logger.warning("ToT plan evaluator: LLM failed, using heuristic: %s", exc)
            return self._heuristic_plan_score(task, plan)

    def _heuristic_score(
        self,
        task: str,
        action: str,
        arguments: dict[str, Any],
    ) -> tuple[float, str]:
        """启发式评分: 基于关键词匹配和参数完整性."""
        score = 0.5  # 默认中等分

        # 如果动作名出现在任务中, 加分
        if action and any(kw in task for kw in action.split(".")):
            score += 0.2

        # 如果有参数, 加分 (说明不是空调用)
        if arguments:
            score += 0.1

        # 参数值与任务相关
        for v in arguments.values():
            if isinstance(v, str) and v.lower() in task.lower():
                score += 0.1
                break

        score = min(1.0, score)
        return score, "heuristic evaluation"

    def _heuristic_plan_score(
        self,
        task: str,
        plan: list[dict[str, Any]],
    ) -> tuple[float, str, list[str]]:
        """启发式计划评分."""
        if not plan:
            return 0.1, "empty plan", ["plan is empty"]

        score = 0.5
        # 有多个步骤说明考虑了复杂性
        if len(plan) >= 2:
            score += 0.2
        # 步骤中有 purpose 说明
        if any(s.get("purpose") for s in plan):
            score += 0.15
        # 步骤中有 task 描述
        if all(s.get("task") for s in plan):
            score += 0.15

        score = min(1.0, score)
        return score, "heuristic plan evaluation", []


# ── Tree Search Engine ────────────────────────────────────────────────────


class TreeSearchEngine:
    """Tree of Thoughts 搜索引擎.

    在关键决策点生成多个候选动作, 评估后使用 UCB1 选择最优.

    用法:
        engine = TreeSearchEngine(llm=pipeline_llm)
        result = engine.explore_plan(task, candidate_plans)
        if result.used_tot:
            best_plan = result.best_path
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        branch_factor: int = _DEFAULT_BRANCH_FACTOR,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        max_nodes: int = _DEFAULT_MAX_NODES,
        min_value: float = _MIN_VALUE_THRESHOLD,
    ) -> None:
        self._llm = llm
        self._evaluator = ToTEvaluator(llm=llm)
        self._selector = UCB1Selector()
        self._branch_factor = branch_factor
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._min_value = min_value

    def select_best_plan(
        self,
        task: str,
        candidate_plans: list[list[dict[str, Any]]],
    ) -> ToTResult:
        """从多个候选计划中选择最优计划.

        Args:
            task: 用户任务.
            candidate_plans: 多个候选计划列表.

        Returns:
            ToTResult 包含最优计划路径.
        """
        if not candidate_plans:
            return ToTResult()

        if len(candidate_plans) == 1:
            # 只有一个候选, 直接评估
            score, reasoning, missing = self._evaluator.evaluate_plan(
                task, candidate_plans[0]
            )
            node = ToTNode(
                node_id="root",
                action="plan",
                value=score,
                visits=1,
                depth=0,
                status="evaluated",
                reasoning=reasoning,
                result=candidate_plans[0],
            )
            return ToTResult(
                best_path=[node],
                total_nodes=1,
                total_evaluations=1,
                best_value=score,
                used_tot=False,
            )

        # 多个候选: 评估并选择
        nodes: list[ToTNode] = []
        evaluations = 0
        pruned = 0

        for i, plan in enumerate(candidate_plans):
            score, reasoning, missing = self._evaluator.evaluate_plan(task, plan)
            evaluations += 1

            node = ToTNode(
                node_id=f"plan_{i}",
                action=f"plan_{i}",
                value=score,
                visits=1,
                depth=0,
                status="evaluated",
                reasoning=reasoning,
                result=plan,
            )

            # 剪枝: 低分候选
            if score < self._min_value:
                node.status = "pruned"
                pruned += 1
            else:
                nodes.append(node)

        if not nodes:
            # 所有候选都被剪枝, 回退到第一个
            logger.warning("ToT: all plans pruned, falling back to first candidate")
            score, reasoning, _ = self._evaluator.evaluate_plan(task, candidate_plans[0])
            return ToTResult(
                best_path=[ToTNode(
                    node_id="fallback",
                    action="plan",
                    value=score,
                    visits=1,
                    depth=0,
                    status="evaluated",
                    reasoning=reasoning,
                    result=candidate_plans[0],
                )],
                total_nodes=len(candidate_plans),
                total_evaluations=evaluations,
                best_value=score,
                used_tot=False,
            )

        # 选择最优
        best = max(nodes, key=lambda n: n.value)
        best.status = "completed"

        return ToTResult(
            best_path=[best],
            total_nodes=len(candidate_plans),
            total_evaluations=evaluations,
            pruned_count=pruned,
            best_value=best.value,
            used_tot=True,
        )

    def evaluate_action_candidates(
        self,
        task: str,
        candidates: list[dict[str, Any]],
        context: str = "",
    ) -> list[tuple[dict[str, Any], float, str]]:
        """评估多个候选动作, 返回按分数排序的列表.

        Args:
            task: 用户任务.
            candidates: 候选动作列表, 每个包含 action 和 arguments.
            context: 当前执行上下文.

        Returns:
            排序后的列表 [(candidate, score, reasoning), ...]
        """
        evaluated: list[tuple[dict[str, Any], float, str]] = []

        for candidate in candidates:
            action = candidate.get("action", candidate.get("name", ""))
            arguments = candidate.get("arguments", {})
            score, reasoning = self._evaluator.evaluate_action(
                task, action, arguments, context
            )
            if score >= self._min_value:
                evaluated.append((candidate, score, reasoning))

        # 按分数降序排列
        evaluated.sort(key=lambda x: x[1], reverse=True)
        return evaluated

    def should_use_tot(
        self,
        task: str,
        plan_length: int,
        has_tools: bool = True,
    ) -> bool:
        """判断是否应该启用 ToT.

        启用条件 (满足任一):
        1. 任务包含多个子问题 (以"并"/"和"/"同时"等连接)
        2. 计划长度 > 2 (复杂多步任务)
        3. 任务包含分析/比较/评估等高认知需求关键词

        禁用条件:
        1. LLM 不可用 (无法评估候选)
        2. 简单查询 (单步任务)
        3. 闲聊/打招呼
        """
        if self._llm is None:
            return False

        if plan_length <= 1:
            return False

        # 简单问候
        simple_keywords = ["你好", "hello", "hi", "谢谢", "thanks"]
        if any(kw in task.lower() for kw in simple_keywords):
            return False

        # 复杂任务关键词
        complex_keywords = [
            "分析", "比较", "评估", "对比", "建议", "规划",
            "为什么", "如何", "怎么办", "哪种", "哪个更好",
        ]
        if any(kw in task for kw in complex_keywords):
            return True

        # 多子问题
        multi_indicators = ["并", "和", "同时", "然后", "接着", "此外"]
        indicator_count = sum(1 for ind in multi_indicators if ind in task)
        if indicator_count >= 2:
            return True

        # 计划较长
        if plan_length > 2:
            return True

        return False
