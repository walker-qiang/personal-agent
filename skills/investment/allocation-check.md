---
name: allocation-check
title: 配置偏离检查
description: 检查当前资产配置与目标配置的偏离度，判断是否需要再平衡操作。
trigger_keywords:
  - 配置
  - 偏离
  - 偏离度
  - 分配
  - 仓位
  - 比例
  - 目标
  - 超配
  - 低配
  - 再平衡
---

# 配置偏离检查

## 工作流
1. finance.bucket_allocation()

## 输出格式
- 各 bucket 当前比例 vs 目标比例
- 偏离度（绝对值）
- 偏离超过 5% 的 bucket 标记为需要关注
- 建议调整方向（增持/减持/保持）