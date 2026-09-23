# PlanCompiler 长期设计

## 目标

把 Runtime 的工具执行从“收到调用后立即判断并执行”收敛为：

```text
ToolRequest
    ↓
PlanCompiler
    ↓
ExecutionPlan
    ↓
ToolExecutionGateway
    ↓
ToolRegistry handler
```

Compiler 不执行副作用，只负责把请求转换成经过策略评估的不可变计划。
Gateway 是最终执行边界，执行前必须再次复核计划，避免策略判断与实际执行
之间出现绕过或状态变化。

## 设计边界

### ExecutionPlan

`ExecutionPlan` 当前包含：

- `operation_id`、调用来源和请求上下文；
- 一个或多个 `PlanStep`；
- 每一步对应的 `ToolRequest`、`ToolSpec` 和 `PolicyDecision`；
- 计划状态：`ready`、`require_approval` 或 `deny`；
- 可供日志和用户展示的阻断原因。

计划编译阶段不会调用 `ToolRegistry` handler，因此可以安全地用于执行前
预览、审批展示和后续持久化。

### PlanCompiler

Compiler 只依赖一个工具元数据解析器和 `PolicyEvaluator`：

1. 检查工具是否注册；
2. 解析工具策略类别、恢复策略和审批标记；
3. 对每个步骤执行纯策略评估；
4. 任一步骤被拒绝或需要审批时，整个计划不可执行；
5. 不对计划中的步骤做部分执行。

当前入口通过 `ToolExecutionGateway.compile()` 使用 Compiler，后续可以把多步
计划持久化到 Runtime operation state，而无需改变工具 handler。

### Gateway

Gateway 提供：

- `compile(request, context)`：生成计划；
- `execute(plan)`：复核计划并调用 Registry；
- `call(request, context)`：兼容旧入口，内部等价于 `compile → execute`。

`execute()` 会再次调用 `PolicyEvaluator`。因此即使未来计划被持久化、跨进程
恢复或由其他入口传入，也不能只凭计划中的旧判断直接执行。

## 审批策略

审批采用“显式高风险标记”而不是“所有写入都审批”：

- 查询、外部读取、内部计算：默认不审批；
- 普通 durable write：默认不审批，但仍受 `read_only/writeback` 模式限制；
- 关键、不可逆、对外发送或需要用户确认的工具：声明
  `requires_approval=True`；
- Gateway 仍然验证 `EffectGrant` 的 owner、operation、tool name 和参数摘要。

因此默认路径不会因为普通写入频繁打断用户；工具作者仍可以对单个高风险
操作开启人工审批。

## 非目标

本阶段不引入：

- 通用 DAG 调度器；
- 自动回滚或跨工具事务；
- 让 Compiler 直接理解业务领域；
- 将所有兼容入口一次性删除；
- 依赖真实 MCP、真实 LLM 或外部服务的测试。

后续需要多步计划时，优先扩展 `PlanStep` 的依赖与恢复元数据，再决定是否
持久化，而不是重新实现一套执行器。

## 验证标准

- 编译不调用 handler；
- denied/require approval 计划不会执行任何 handler；
- 普通 durable write 在 writeback 模式下不会自动暂停；
- 显式 `requires_approval=True` 的操作仍会暂停；
- Gateway 执行前会再次复核策略和 EffectGrant；
- 现有 Runtime、ReAct、Skill、HTTP 入口的结果兼容。
