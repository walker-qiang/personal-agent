# 投资研究核心契约

在 `codex/investment-research-core` 建立契约与资料准备消费端，不修改现用 V3 研究路由。
长期目标是由 OS 持有资料、公式、执行及写回，Agent 负责规划、补查意图和建议分析。

## 所有权

- 唯一 Schema 源属于 `personal-os/packages/investmentresearch/core.schema.json`。
- 本仓不复制 Schema，不重新实现原子金融公式，也不直接读写 `personal-assets`。
- `matrix.investment.contracts.ResearchContract` 消费可信 API 发来的版本化契约包，并与能力握手摘要核对。
- 第一阶段以独立 Go CLI 提供同样的契约包用于离线验收；不通过读取相邻仓源码建立运行时集成。
- 日期和 URI 格式依赖是必需依赖；缺少时拒绝加载契约，而不是跳过校验。
- 只允许本地定义引用，禁止 Schema 的隐式远程获取。错误指出记录类型和字段位置。

## 使用

```python
from matrix.investment.contracts import ResearchContract

contract = ResearchContract.from_bundle(
    trusted_api_bundle,
    expected_digest=trusted_capability_digest,
)
contract.validate("Plan", proposed_plan)
```

不要从模型输出获取 `trusted_api_bundle` 或 `trusted_capability_digest`。
摘要只能证明与可信能力声明一致，不能把任意来源变成可信来源。

本阶段只提供结构校验。方法是否支持、输入期间是否可比、引用是否真实存在、持仓授权是否匹配、
建议是否有证据支持，后续由 OS 编排与决策层检查；Schema 通过不等于研究完成。

## 切换边界

旧路由继续服务现用 App，第一阶段没有新增对外研究接口。
完成新引擎、专属方法及受控数据迁移后，同步切换 API、Agent 和 App，再删除旧执行路径和重复公式。
这不是长期双协议兼容设计。历史原文保留在统一归档中，不通过旧 Agent 协议重跑。

跨仓验收入口在 OS 的 `tools/acceptance/research_core.py`，全部使用假数据和临时资产仓。
不新增单元测试、不运行真实模型、不直接对真实 Vault 执行迁移或 Git 操作。

## 第二阶段资料准备

OS 提供只读 `/api/investment/research/core/capabilities`，返回与独立 CLI 同源的契约及注册信息。
`ResearchCapabilities` 读取这一可信响应；`PreparedResearchInputs` 绑定操作和标的身份、
引擎、方法及契约摘要，避免把另一标的或旧版本输出当成本轮输入。

`tool_index(method_id)` 仅返回工具 ID、版本和短说明。完整 Schema、来源目录和恢复快照不放进模型提示词。
本仓仍不执行任何金融公式，也不读取相邻 OS 仓源码作为运行时集成方式。

`Preparation.ready` 只表示资料准备要求得到满足，不是投资建议。
全部新方法仍 `analysis_ready=false`；`require_analysis_enabled()` 会拒绝模型执行，
离线回放也不能作为实网研究输入。第三阶段完成方法闭环、权限/预算绑定和建议质量验收后再启用。

第二阶段离线验收在 OS 的 `tools/acceptance/research_preparation.py`，
包括临时 API 能力握手、复合证据身份验证、跨标的绑定以及准备完成不能冒充分析完成。

## 第三阶段离线分析

`matrix.investment.analysis.AnalysisReplayInputs` 消费同一份 Schema 中的 `AnalysisContext`。
它核对目标、父计划、上下文摘要、方法及授权，只生成可检视的精简模型请求数据。
系统提示词仅有规则和输出字段；工具目录仍只有 ID、版本及短说明，不发送完整 Schema 或来源信任目录。
原文明确标为不可信数据，不能当作指令。

`replay_envelope` 包装假模型原文，特意不在包装时丢弃无效 JSON/无效建议：
OS 先归档回执，再执行共享 Schema、引用图、方法覆盖、数值引用和建议维度检查。
Agent 不负责计算、持久化父预算、判定完成或写 Vault。

本模块没有调用真实 `LLMClient` 的入口，也没有新增服务路由。
`require_analysis_enabled()` 的 live gate 保持关闭，不能拿离线回放绕过它发起真实模型请求。
OS 的 `tools/acceptance/research_analysis.py` 提供跨仓隔离验收；不新增单元测试。
真实数值提供方、元数据补全、自主补查/计划修订和真实建议质量仍是未完成工作，
不能把假场景 `completed` 当作第三阶段全部完成。

## 原生准备输入

消费端已接受共享契约的 `MetadataClaim`、已补全标的信息及精度字段。
`PreparedResearchInputs` 检查字段依据、相关指数与标的信息的来源引用必须属于本次准备，
不允许另一标的或未知证据混入。公式、元数据合并与来源适配仍只由 OS 持有。

原生准备目前是冻结分析计划之前的阶段，不是 Agent 可自行启用的真实分析路径。
市场 PE/PB 参考、相关指数和日级报告 NAV 不能被解释成报告倍数、完整穿透关系或实时交易溢价。
全部方法的 live gate 保持关闭，本仓没有添加付费模型调用或 Vault 写入能力。

## 同一研究范围

`CoordinatedResearchInputs` 消费 OS 的 `ResearchRun`，核对稳定研究 ID、范围指纹、已确认身份、
冻结计划、准备子任务与结果归属。预算和计划生成仍由 OS 持有；Agent 不创建新预算或复写身份。
源自公共资料的上下文可只读检视，但 `AnalysisReplayInputs` 明确拒绝它用于假模型重放。
当前没有任何真实模型综合入口，不能因整轮数据账本贯通而宣称投资建议已可用。
