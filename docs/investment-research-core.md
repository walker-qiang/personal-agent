# 投资研究核心契约

本轮在 `codex/investment-research-core` 建立第一阶段的消费端，不修改现用 V3 研究路由。
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
