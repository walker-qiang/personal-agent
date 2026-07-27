# Project Matrix 改进设计 — 借鉴 Pi-Agent 的 7 个方向

> 基于 Pi-Agent Book 全 10 章源码级分析，对照 Matrix 现有代码，按"改动小、收益快、风险低"优先排序。
>
> **文档状态**：已实现并提交。本文档与代码同步更新，记录实现过程中的务实调整。

## 总览：改动顺序与依赖关系

```
Phase 1: 工具输出截断统一化     ← 无依赖，纯新增工具模块
Phase 2: 工具执行管道下沉        ← 依赖 Phase 1（截断作为管道一环）
Phase 3: 错误处理规范化          ← 依赖 Phase 2（管道内统一错误编码）
Phase 4: 压缩算法改进            ← 独立，但 Phase 1 的截断减少了压缩压力
Phase 5: 系统提示词动态组装       ← 独立
Phase 6: 事件系统结构化          ← 独立
Phase 7: 会话树                  ← 架构级改动，最后做
```

前 3 个 Phase 构成"工具系统"改进链，4-6 独立可并行，7 单独规划。

---

## Phase 1: 工具输出截断统一化

### 问题

Matrix 当前截断逻辑散落在各工具内部，且不统一：

| 工具 | 改动前做法 | 问题 |
|------|-----------|------|
| `web_fetch` | `text[:max_chars]` 暴力截断 | 可能切坏多字节字符 |
| `code/executor` | `stdout[:max_output_chars]` 暴力截断 | 同上，且只截头不截尾 |
| `finance/holdings` | 无截断 | 持仓数据量大时直接进上下文 |
| `news_search` / `web_search` | 无截断 | 搜索结果多时膨胀 |

L1 ToolResultRefStore 解决了"大结果外置"（>8000 字符），但 3000-8000 字符的中间地带没有截断保护。

### 实现

新建 `src/matrix/tools/truncate.py`，提供统一的截断函数：

```python
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB

def truncate_head(content, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES) -> TruncationResult
def truncate_tail(content, max_lines=DEFAULT_MAX_LINES, max_bytes=DEFAULT_MAX_BYTES) -> TruncationResult
def truncate_result(result: dict, max_array_items=50) -> dict  # 结构化结果截断
def truncate_string(content, max_chars=5000, from_head=True) -> str  # 便捷函数
```

`TruncationResult` dataclass 包含 `content`, `truncated`, `truncated_by`, `total_lines`, `total_bytes`, `output_lines`, `output_bytes`。

关键实现点：
- **双重限制**：行数和字节先触者胜，互相兜底
- **UTF-8 边界安全**：逐行累加字节数（`_byte_len` 用 `len(s.encode("utf-8"))`），`truncate_string` 用 `_is_continuation_byte` 检查代理对
- **截断提示行**：`[已截断: 原始 {total_lines} 行 / {total_bytes}, 保留 {output_lines} 行 / {output_bytes}]`
- **`truncate_result`**：对 dict 结果中的已知数组字段（holdings/buckets/snapshots/assets/results/items/news）超过 50 项时截断，已知文本字段（text/stdout/stderr/content/output/notes/description）过 `truncate_head`

### 应用方式

- `web/fetch.py`：`text[:max_chars]` 替换为 `truncate_head(text, max_bytes=max_chars * 4)`
- `code/executor.py`：`stdout[:max_output_chars]` 替换为 `truncate_tail(stdout, max_bytes=self._max_output_chars)`
- `finance/__init__.py`：注册时包裹 `truncate_result(holdings.holdings_summary(...))`
- `web/__init__.py`：search/news_search/finance_query 注册时包裹 `truncate_result(...)`

### 不改的

- ToolResultRefStore 保持不变 — 截断是第一道关（控制单条体积），外置是第二道关（控制总量）
- 各工具的 `input_schema` 不变

---

## Phase 2: 工具执行管道下沉

### 问题

Matrix 的工具 guards 逻辑散落在 `react.py` 的 `_pass_tool_guards` 和 `_execute_single_tool` 里。`ToolRegistry.call()` 本身只做了 ToolGuard 和 IndirectInjectionGuard，缺少参数 Schema 验证和参数预处理。

### 实现

把五步管道下沉到 `ToolRegistry.call()` 内部：

```python
def call(self, name, arguments=None) -> dict:
    # Step 1: prepareArguments — 兼容性垫片（丢弃未知字段、还原字符串化数组）
    args = self._prepare_arguments(tool, args)
    # Step 2: validateArguments — 轻量 Schema 验证（required + 基本类型）
    ok, reason = self._validate_arguments(tool, args)
    if not ok: return {"error": f"参数验证失败: {reason}"}
    # Step 3: beforeToolCall — guards（ToolGuard + CodeGuard）
    #   注意：guards 仍然 raise ToolGuardError（非返回 error dict），
    #   因为 ReAct 的 circuit breaker 依赖异常行为
    if self._guard: ... raise ToolGuardError(...)
    if self._code_guard: ... raise ToolGuardError(...)
    # Step 4: execute + truncate
    try:
        result = tool.handler(**args)
    except Exception as err:
        return {"error": self._format_error(name, args, err)}
    result = truncate_result(result)  # Phase 1 的截断
    # Step 5: afterToolCall — IndirectInjectionGuard
    if self._injection_guard: result = self._injection_guard.check_and_sanitize(name, result)
    return result
```

### 与设计文档原稿的差异（实现中调整）

1. **Guard 行为**：设计文档原稿说 guards 返回 `{"error": ...}` dict。实际代码中 guards **仍然 raise `ToolGuardError`**。原因：ReAct 的 CircuitBreaker 和测试套件依赖这个异常行为，改为返回 dict 会破坏 circuit breaker 逻辑。`call()` 的文档注释明确写了 "Never raises"，但 `ToolGuardError` 是唯一的例外。

2. **`ToolDefinition` 没有加 `prepare_arguments` 字段**：设计文档原稿说 `base.py` 可选加这个字段。实际没有加 — `_prepare_arguments` 是 registry 里的通用逻辑（丢弃未知字段 + 还原字符串化数组），不是 per-tool 的可定制钩子。当前的通用实现已经能处理主要场景。

3. **`_format_error` 签名**：实际代码是 `f"工具 {name} 执行失败 [{err_type}]: {err_msg}。参数: {args_preview}"`，比设计文档原稿多了 `[err_type]` 标签。

### ReAct 节点调整

`_pass_tool_guards` 里的 dedup / circuit-breaker / max-calls 逻辑保留在 ReAct 层（这些是 ReAct 循环级别的控制）。`_execute_single_tool` 改为检查 `call()` 返回的 error key：

```python
tool_result = agent_tools.call(name, arguments)
if isinstance(tool_result, dict) and "error" in tool_result:
    # trace + circuit breaker record_failure
    return False, {"name": name, "error": tool_result["error"], ...}
# 正常路径...
```

`commander.py` 和 `skills/executor.py` 的调用方也做了同样调整 — 检查 error key 而非 catch `FinanceToolError`。

---

## Phase 3: 错误处理规范化

### 问题

Matrix 工具内部的错误描述质量参差不齐。`FinanceToolError` 可能只 throw `"查询失败"`，模型看到后无法判断是路径错了、数据库锁了、还是数据不存在。

### 实现

#### 3.1 `tool_error()` 辅助函数（已加到 `base.py`）

```python
def tool_error(tool_name, operation, reason, suggestion="", context=None) -> dict:
    parts = [f"[{tool_name}] {operation}失败: {reason}"]
    if suggestion: parts.append(f"建议: {suggestion}")
    if context: parts.append(f"上下文: {ctx_preview}")
    return {"error": "。".join(parts)}
```

#### 3.2 与设计文档原稿的差异（实现中调整）

设计文档原稿说改造 `holdings.py`、`web/fetch.py`、`code/executor.py` 使用 `tool_error()` 返回结构化错误。实际实现做了**不同**的事情：

- **`tool_error()` 函数已加到 `base.py`**，供未来新工具使用
- **现有工具没有改用 `tool_error()`** — 而是改进了 `FinanceToolError` 的**错误消息文本**，使其更具体
- 改动的文件是 `shared.py` 和 `snapshots.py`，不是 `holdings.py` / `fetch.py` / `executor.py`

实际改的错误消息示例：

| 工具 | 改动前 | 改动后 |
|------|--------|--------|
| `shared.py` connect_readonly | `"finance cache does not exist: {path}"` | `"持仓数据库不存在: {path}。请检查 MATRIX_CACHE_PATH 环境变量，或确认 personal-os 已同步数据。"` |
| `shared.py` clamp_int | `"limit must be an integer"` | `"limit 参数必须是整数，得到 {type}: {value}。请传入 1 到 200 之间的整数。"` |
| `snapshots.py` asset_id 校验 | `"asset_id must be a durable ast_* id"` | `"asset_id 必须以 ast_ 开头（durable asset id），得到: {asset_id}。请先用 finance.asset_lookup 查找正确的 asset_id。"` |
| `snapshots.py` 未找到资产 | `"asset not found: {asset_id}"` | `"未找到资产: {asset_id}。请用 finance.asset_lookup 确认该 asset_id 是否存在。"` |

此外，`ToolRegistry._format_error()` 在管道兜底层统一格式化：`"工具 {name} 执行失败 [{err_type}]: {err_msg}。参数: {args_preview}"`，确保即使工具内部 throw 了笼统的错误，模型也能看到工具名、参数和错误类型。

---

## Phase 4: 压缩算法改进

### 问题

Matrix 的 `compaction.py` 已有结构化 handoff（4 section），但有四个差距：固定条数切割、无增量更新、缺 Critical Context、无数据引用跟踪。

### 实现

#### 4.1 切割点改成 token-based 向后遍历

```python
KEEP_RECENT_TOKENS = 20000

def find_cut_point(messages, keep_tokens=KEEP_RECENT_TOKENS) -> int:
    # 从最新消息往回累积 token，直到 >= keep_tokens
    # 找最近的有效切割点（非 tool 消息）
    # 保证 MIN_PRESERVE_MESSAGES 和 MIN_DELETE_MESSAGES
```

#### 4.2 增加 Critical Context section

从 4 section 扩展到 5 section：`user_goal` / `execution_history` / `abandoned_paths` / `critical_context` / `data_references`。

#### 4.3 增量更新

- 新增 `UPDATE_SYSTEM_PROMPT`（增量更新专用 prompt）
- `build_compaction_messages()` 接受 `previous_summary` 参数，传入时用增量 prompt
- `compact_messages()` 接受 `previous_summary: dict | None` 参数
- data_references 跨压缩累积：合并 previous_summary 中的 refs

#### 4.4 Handoff 往返提取（实现后补充）

为支持增量更新，需要从消息列表中提取上一次压缩的 handoff dict。实现方式：

- `build_handoff_message()` 在内容开头嵌入 `<!-- HANDOFF_JSON: {...} -->` 隐藏标记
- `extract_previous_handoff(messages)` 从第一条 system 消息中解析标记，返回 dict 或 None
- `strip_previous_handoff(messages)` 移除旧 handoff 消息，防止重复压缩
- `_run_budget_and_compact()` 调用链：`extract → strip → compact_messages(previous_summary=extracted)`

#### 4.5 与设计文档原稿的差异

设计文档原稿说 `_run_budget_and_compact` 传入 `previous_summary`。初版实现遗漏了这一步，后续修复补上。修复引入了 `extract_previous_handoff` / `strip_previous_handoff` 两个函数和 `HANDOFF_JSON` 标记机制 — 这些是设计文档原稿没有描述的实现细节。

---

## Phase 5: 系统提示词动态组装

### 问题

Matrix 有 `AGENTS.md` 在项目根目录，但 Agent 运行时不会自动读取并注入到系统提示词。Skills 全文也没有进提示词。

### 实现

新建 `src/matrix/orchestration/context_loader.py`：

```python
def load_project_context_files(cwd: Path) -> list[tuple[Path, str]]
    # 从 cwd 向上递归找 AGENTS.md / CLAUDE.md，去重，外→内排序

def build_project_context_section(cwd: Path) -> str
    # 构建 <project_context><project_instructions path="...">...</project_instructions></project_context>

def build_skills_section(agent_def, agent_registry) -> str
    # 构建 <available_skills><skill name="...">description</skill></available_skills>
    # 只放清单，LLM 需要时自己调 read 工具拉全文

def enrich_system_prompt(base_prompt, cwd, agent_def, agent_registry) -> str
    # 组合：base_prompt + project_context + skills
```

注入点：
- `react.py` 的 `react_prepare_node`：在 working memory 和 data index 注入之后调用 `enrich_system_prompt()`
- `commander.py` 的 `_run_domain_agent_react`：同上

验证：从 `personal-agent/` 目录运行时，能找到 2 个 AGENTS.md（`personal-agent/AGENTS.md` + `personal-system/AGENTS.md`），按外→内顺序合并。

---

## Phase 6: 事件系统结构化

### 问题

Matrix 用字符串事件类型（"progress"、"thinking"、"tool_call"、"tool_result"），通过 `queue.Queue` 传递。没有结构化的事件类型定义。

### 实现

新建 `src/matrix/orchestration/events.py`，定义 13 种结构化事件 dataclass：

```
AgentStartEvent / AgentEndEvent          ← Agent 生命周期
TurnStartEvent / TurnEndEvent            ← Turn 生命周期
ThinkingEvent                            ← Message 层
ToolCallEvent / ToolResultEvent          ← Tool Execution 层
ProgressEvent                            ← 通用进度
PlanCreatedEvent / StepStartEvent / StepDoneEvent / StepErrorEvent / ReplanEvent  ← Plan-and-Execute
```

每个事件有 `to_dict()` 方法，`timestamp` 自动填充。

`make_event(event_type: str, payload: dict)` 工厂函数提供向后兼容 — 从字符串类型 + payload dict 创建结构化事件。

### 与设计文档原稿的差异（实现中调整）

设计文档原稿说：
- `_push_event` 改签名为 `def _push_event(cfg, event: AgentSessionEvent)`
- `server/routes/chat.py` SSE 序列化改成 `event_to_sse()`
- `react.py` 和 `commander.py` 的所有 `_push_event` 调用改用结构化事件

**实际实现做了向后兼容方案**：
- `_push_event` 签名**没变**，仍然是 `(cfg, evt_type: str, payload: dict)`
- 内部调用 `make_event()` 创建结构化事件（用于未来迁移），但 put 到 queue 的仍然是 `(evt_type, payload)` 元组
- `chat.py` SSE 序列化层**没改**
- 所有 `_push_event` 调用点**没改**

**原因**：Matrix 的事件消费者（SSE route、前端）都基于 `(string, dict)` 元组。一次性改签名需要同时改所有消费端，风险较大。当前方案是"影子系统" — 事件类型定义已就位，`make_event` 桥接层已就位，后续迁移时只需改 `_push_event` 的 queue.put 和 SSE 序列化层。

### 不做的

- **同步屏障**：Pi 的 `await emit` 在 Matrix 里暂不需要 — `queue.Queue` 天然有缓冲，SSE 消费端是独立线程。如果后续发现 UI 乱序问题再考虑。

---

## Phase 7: 会话树

### 问题

Matrix 的会话存储是线性数组 — `save_message` 追加一行，`get_history` 取最近 N 条。无法回退、分叉、保留多条探索路径。

### 实现

#### 7.1 数据模型

```sql
-- messages 表加列（通过 migration）
ALTER TABLE messages ADD COLUMN message_id TEXT NOT NULL DEFAULT '';
ALTER TABLE messages ADD COLUMN parent_id TEXT;

-- sessions 表加列
ALTER TABLE sessions ADD COLUMN leaf_id TEXT;

-- 新增索引
CREATE INDEX idx_messages_message_id ON messages(message_id);
CREATE INDEX idx_messages_parent_id ON messages(parent_id);
```

#### 7.2 核心操作

```python
def save_message(self, session_id, role, content, user_id="") -> str:
    # 生成 msg_id，parent_id = session 当前 leaf_id
    # 插入消息，更新 session.leaf_id = msg_id
    # 返回 msg_id

def branch(self, session_id, from_message_id) -> bool:
    # 验证 from_message_id 属于该 session
    # UPDATE sessions SET leaf_id = from_message_id
    # 不删任何数据

def get_history(self, session_id, max_turns=8) -> list[dict]:
    # 从 leaf_id 往回遍历 parent_id 链到根
    # 收集 max_turns*2 条消息，reverse 为时间顺序
    # 如果 leaf_id 为空，fallback 到线性查询

def get_leaf_id(self, session_id) -> str | None
def get_branches(self, session_id) -> list[dict]  # 查 parent_id 出现多次的节点
```

#### 7.3 API 端点

```
POST /sessions/{id}/branch    {"from_message_id": "abc123"}  → 回退到指定节点
GET  /sessions/{id}/branches                                  → 返回所有分叉点
GET  /sessions/{id}/leaf                                      → 返回当前 leaf_id
```

#### 7.4 与设计文档原稿的差异

1. **`GET /sessions/{id}/tree` 未实现**：设计文档原稿列了 3 个端点（branch / tree / branches），实际实现了 branch / branches / leaf。`/tree`（返回完整树结构）没有实现，因为前端只需要知道当前位置（leaf）和分叉点（branches），不需要整棵树。`/leaf` 是额外加的。

2. **`chat/_service.py` 没有改**：设计文档原稿说 chat service 对接 leaf_id / branch。实际没改 — store 层的改动已经让线性模式（树的特例）正常工作，branch 功能通过 API 端点直接操作 store 即可，不需要 chat service 参与。

#### 7.5 迁移策略

- 现有消息 `message_id` 用 `'m' || CAST(id AS TEXT)` 生成
- `parent_id` 回填为同 session 中前一条消息的 message_id（按 id 排序）
- `leaf_id` 回填为同 session 中最后一条消息的 message_id
- 线性模式是树模式的特例（每个节点只有一个子节点），迁移后行为不变

---

## 优先级总结

| Phase | 改动量 | 风险 | 收益 | 状态 |
|-------|--------|------|------|------|
| 1. 截断统一 | 小 | 低 | 高 | ✅ 已完成 |
| 2. 管道下沉 | 中 | 中 | 高 | ✅ 已完成 |
| 3. 错误规范 | 小 | 低 | 中 | ✅ 已完成 |
| 4. 压缩改进 | 中 | 中 | 中 | ✅ 已完成 |
| 5. 提示词组装 | 小 | 低 | 中 | ✅ 已完成 |
| 6. 事件结构化 | 中 | 低 | 中 | ✅ 已完成（影子系统，待后续迁移） |
| 7. 会话树 | 大 | 高 | 高 | ✅ 已完成 |

## 后续待办

- Phase 6 迁移：将 `_push_event` 改为直接 put 结构化事件，SSE 序列化层改用 `event.to_dict()`
- Phase 3 扩展：新工具使用 `tool_error()` 函数，现有工具逐步迁移
- Phase 7 扩展：分支摘要（BranchSummaryEntry）— 切换分支时给被放弃的分支生成 LLM 摘要
