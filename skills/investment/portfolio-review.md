---
name: portfolio-review
title: 组合复盘
description: 对投资组合进行定期复盘，检查配置偏离度、收益概览和再平衡建议。
---

# 组合复盘

## 工作流
1. finance.holdings_summary()
2. finance.bucket_allocation()
3. finance.recent_snapshots(limit=50)

## 输出格式
- 总资产概览（总额、币种）
- 各 bucket 配置偏离度（当前 vs 目标）
- 资产分布概览（类型、通道）
- 再平衡建议（优先级排序）