---
name: anomaly-diagnosis
title: 持仓异动诊断
description: 对持仓数据进行异动诊断，识别异常变化并进行归因分析。
---

# 持仓异动诊断

## 工作流
1. finance.holdings_summary()
2. finance.recent_snapshots(limit=20)
3. finance.bucket_allocation()

## 输出格式
- 异动资产列表（按变化幅度排序）
- 每项资产的变化金额、变化比例
- 归因分类（市场波动 / 现金流变动 / 数据修正 / 未知）
- 建议行动（关注 / 再平衡 / 无操作）