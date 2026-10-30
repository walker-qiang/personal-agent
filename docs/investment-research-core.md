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

## 原生财务与跟踪关系

`research-native-valuations-v7` 新增 `financial_data` 与 `index_valuation` 证据类型。
前者是提供方编制的财务数值，不代替原始 `financial_report` 中的经营与风险证据。
后者必须保留 `tracking_index` 和 `index_reference`；消费端检查与产品条款证据的关联，
不把 `related_index`、净值或产品名称当作实际跟踪关系。

股票 PB、收入 TTM 与归母净利润 TTM 由 Go 持有和重算。
Agent 仍不拼接 EPS，不估算缺项，不因为公式正确就提升提供方数值的核验状态。
精简提示只增加来源语义、跟踪关系和穿透标记，不加入详细适配器说明。

跨仓验证使用 OS 的 `tools/acceptance/research_valuations.py`。
真实数据准备与真实建议有用性分别验收；盘中时点设计、必要财报和基金穿透仍未闭环，
公开上下文继续禁止进入假回放或真实模型执行。

## 研究时间快照

`research-time-window-v8` 在共享契约增加 `ResearchTimePolicy` 与 `ResearchWindow`。
当前研究默认由 OS 在实际采集结束后冻结估值时点；固定历史研究则保持显式指定的截止时间。
新整轮请求不再接受含糊的顶层 `as_of`，旧运行记录不原地重解释。

消费端核对窗口起止、父/子上下文一致性、披露/观察时间与来源回执的关系。
只有冻结快照可以进入离线综合；提示携带冻结时间，不让模型自行刷新时间、指标或事实。
冻结后恢复不等于取得当前新行情，也不能让模型将旧快照解释成恢复时的实时建议。
时间语义与未来数据校验由 Go 执行器持有，Agent 的检查只是消费边界，不复制金融公式。

跨仓隔离验收在 OS 的 `tools/acceptance/research_time.py`。
公开数据仍只准备、不调用真实模型；App、资产迁移和真实建议质量 gate 未切换。

## 股票问题正文

`research-disclosure-evidence-v9` 为公司方法增加明确的风险问题和业务披露要求。
股票输入不再使用统一的原文开头截取，Go 输出带 `question_id`、标题、提取文本摘要及 UTF-8
字节范围的片段。消费端检查全文属性、主题和范围长度，拒绝把摘要或目录当成问题正文。
旧期业务说明与最新财务期间分开保留，不让模型将二者的数值或时点混用。

原始报告或章节覆盖通过，不代表财务数值独立对账、合理估值或买卖建议质量通过。
Agent 不选择发布者账号、不推导机构 ID、不绕过访问拒绝，也不复制 Go 章节算法与金融公式。
系统提示只携带简短语义边界；完整来源目录和适配器实现不进入模型注意力上下文。

跨仓隔离验收在 OS 的 `tools/acceptance/research_disclosures.py`，公开样本验证只取数。
现用路由、真实模型、资产迁移和 App gate 均未切换。

## 估值依据与条件建议

`research-valuation-basis-v10` 增加 `ValuationPolicy`、`ScenarioAssumption`、
`ValuationSeries` 及 `ValuationContext`。配置由 OS 操作者注册，不来自模型草稿；
消费端核对目标、实际测量的同行证券、提供方和证据归属，公司配置不能应用于基金。

系统提示区分事实、假设与 Go 计算结果。历史分位和同行倍数只是相对定价描述，
不能单独支持低估、正常或高估；公司估值需要已批准的条件倍数情景，
声明 `conditional_multiple_scenario` 并引用全部假设 ID。
条件重定价不等于内在价值、报告 EPS 估计或未来价格预测。
若配置必要买入余量，买入建议须引用并满足程序生成的条件入场参考。

模型只收到序列元信息、样本数及原子计算结果，不收到完整历史数组；
工具索引继续仅含 ID、版本与短说明，金融公式、预算、完成状态及写回仍属于 Go。
新约束的跨仓隔离验收在 OS 的 `tools/acceptance/research_valuation_basis.py`。
匿名公开验证只检查参考数据和明确阻断，不连接真实 `LLMClient`，
不将假回放成功当作真实建议质量或 App 可用性通过。

## 可复用基本面规则

`research-fundamental-rules-v11` 消费共享 `ValuationRule`，同一币种和财务画像的规则
跨标的复用，不由模型逐标的生成 PE/PB。
提供方明确的银行/通用类型及年度、已实施分红事实由 Go 核验，
先补足必要事实，再等待缺失的投资参数批准。

综合只发送选中规则、计算结果和必要直接输入，不发送全部模板或原始年度数值表。
银行 PB-ROE、非金融盈利收益率和 DDM 均是带来源及假设的条件结果，
不是审计内在价值、FCFE 或未来承诺；相关模型也不算独立确认。
公司评估使用 `conditional_fundamental_scenario` 并引用选中规则全部假设；
程序确认的方法分歧使用 `uncertain`，不得改写成正常，也不能给出违背保守入场条件的买入建议。

OS 的 `tools/acceptance/research_fundamentals.py` 覆盖跨标的复用、内部补数及建议约束。
新方法仍不连接真实模型或 App，真实建议的有用性未因合成回放成功而通过。

## 风险与受限结果

`research-policy-layers-v12` 消费已批准的七个共享研究数值与独立程序风险项，
不再让模型把亏损、低现金覆盖或资本待核验当作缺少数据。
正风险项通过 `required_risk_refs` 明确传入，必须逐项进入 `risks.refs`。
银行资本上下文仍由 Go 按有来源的机构、层级、方法和有效期匹配；
消费端拒绝没有监管来源依据的要求，不能把匹配比较夸大为完整监管合规。

无适用价格模型时，评估必须为 `not_applicable`、
`fundamental_method_unavailable`，不能伪造合理价或买入条件。
仅一个模型或没有价格模型的已验证意见由 Go 标为 `limited`，
明确尚无实现的机构类型可以返回不含建议的受限结果。
受限建议不是故障或完整估值；买入还必须通过程序风险、模型数量及保守余量检查。
真实付费模型、个人持仓和 App 切换边界不变。

## 显式授权的真实传输

`matrix.investment.synthesis` 新增独立的 `prepare` / `execute` 边界，
只接受 OS 验证并绑定到原始公开快照的追加许可。
`AnalysisInputs` 复用资料、方法和时间校验，`AnalysisReplayInputs` 仅保留假回放入口；
公开资料不能套用假回放信封。原采集许可不会被原地改写成模型外发许可。

`prepare` 不联网，生成完整且不截断的请求，使用可选 `research` 依赖中的
`deepseek-recipe==0.2.0`、固定摘要的官方分词文件和 V4.1 消息编码计量。
完整结构化输出定义参与计量；它不是一个独立的规划、评判或修复模型。
`execute` 再次核对已存请求的字节哈希及计量，只发出一次指定提供方请求，
关闭推理、流式输出、重试和携带凭据的重定向。
返回 HTTP 状态和原始响应正文字节，不先丢弃非 JSON、错误响应或不合格草稿。
Go 持有原始响应档案、预算结算、恢复、最终状态与投资语义校验，
Agent 不自行标记研究成功，不推算缺失金融指标，不读取个人持仓。

2026-10-30 两个真实样本各调用一次，总计 58,964 token。
模型正常返回完整 JSON，实际输入用量与本地编码一致；
但两份意见均因数值正文违反引用协议被拒绝，真实建议有用性没有通过。
还发现公司/基金维度混用、历史期间表述不准及行动解释不足。
失败响应保留在 OS 忽略的验收状态中，可做零模型调用的离线回归，
不能把这些未接受草稿直接当作投资建议展示。
本轮不再付费重试；普通研究路由与 App 门禁仍不切换。
