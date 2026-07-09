---
name: allocation-check
title: 配置偏离检查
description: 检查当前资产配置与目标配置的偏离度，判断是否需要再平衡操作。
---

# 配置偏离检查

## 工作流
1. finance.bucket_allocation()

## 输出格式
- 各 bucket 当前比例 vs 目标比例
- 偏离度（绝对值）
- 偏离超过 5% 的 bucket 标记为需要关注
- 建议调整方向（增持/减持/保持）