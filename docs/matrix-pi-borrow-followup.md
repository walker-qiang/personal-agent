# Pi-Agent 借鉴改进收尾记录

> 基于 7 Phase 实现后的差距评估，记录收尾改动、验收结果和仍保留的长期路线。
> Fix 1-5 已完成；后续新增的 Runtime 查询、分支摘要和重启恢复也已完成。

> **阶段更新（2026-08-15）**：在完成上述 Phase 收尾后，已补充 Runtime 状态与审批历史查询闭环，作为 Agent 可观测性增强，不改变 Runtime 与 LangGraph 的依赖边界。

## 总览

```
Fix 1: 截断入口统一 + 覆盖盲区        ← Phase 1 遗留，改动最小
Fix 2: tool_error() 落地 + 错误规范    ← Phase 3 遗留，改动小
Fix 3: 事件系统迁移完成                ← Phase 6 遗留，改动中等
Fix 4: 会话树前端集成                  ← Phase 7 遗留，改动中等
Fix 5: 提示词组装缓存                  ← Phase 5 遗留，改动最小
```

---

## Fix 1: 截断入口统一 + 覆盖盲区（Phase 1 遗留）

> **实现状态（2026-07-29）**：已完成 ✅。注册层已无 `truncate_result` 包裹；`_ARRAY_FIELDS` 已扩展 `images`，`_TEXT_FIELDS` 已扩展 `prompt`/`url`。

### 问题

1. **双重截断**：finance 和 web 工具在 `__init__.py` 注册时包裹了 `truncate_result`，`registry.call()` 又调了一次。功能上无害（第二次是 no-op），但架构不干净。
2. **覆盖盲区**：agnes（返回 `images` 数组）、rag（返回 `results` 数组，已在列表中）、mcp（返回结构未知）的结果没有被截断。`_ARRAY_FIELDS` 和 `_TEXT_FIELDS` 不覆盖 agnes 的字段名。

### 设计

#### 1.1 去掉注册层的 truncate_result 包裹

`registry.call()` 的 Step 4 已经统一调了 `truncate_result`，注册层的包裹是多余的。

```python
# finance/__init__.py — 改回直接调用
handler=lambda **kwargs: holdings.holdings_summary(cache_path=str(cache_path), **kwargs),
# 不再包裹 truncate_result

# web/__init__.py — 同理
handler=search.web_search,
# 不再包裹 truncate_result
```

#### 1.2 扩展 _ARRAY_FIELDS 和 _TEXT_FIELDS

```python
# src/matrix/tools/truncate.py

_ARRAY_FIELDS = (
    "holdings", "buckets", "snapshots", "assets",
    "results", "items", "news",
    "images",          # agnes 图片生成
    "documents",       # rag 文档（如果有的话）
)

_TEXT_FIELDS = (
    "text", "stdout", "stderr", "content", "output", "notes", "description",
    "prompt",           # agnes 生成 prompt
    "url",              # web_fetch URL
)
```

#### 1.3 web_fetch 去掉自己的 truncate_head 调用

`web_fetch` 当前在函数内部调了 `truncate_head(text, max_bytes=max_chars * 4)`。这个应该由 `registry.call()` 的 `truncate_result` 统一做。但 `web_fetch` 的 `max_chars` 参数是用户可配的（默认 5000），需要保留这个语义。

方案：`web_fetch` 不再自己截断，但 `truncate_result` 对 `text` 字段用 `DEFAULT_MAX_BYTES`（50KB）截断。如果用户传了 `max_chars=5000`，`web_fetch` 在返回前把 `max_chars` 作为 hint 存入结果，`truncate_result` 读取它。

更简单的方案：`web_fetch` 保留自己的截断（它是特殊的 — 用户可配 max_chars），但去掉 `__init__.py` 里多余的 `truncate_result` 包裹。`registry.call()` 的 `truncate_result` 对已经截断过的结果做 no-op。

**选择简单方案**：只做 1.1（去掉注册层包裹）和 1.2（扩展字段列表）。`web_fetch` 和 `code/executor` 保留自己的截断（它们有自定义参数），`registry.call()` 的 `truncate_result` 作为兜底。

### 改动范围

| 文件 | 改动 |
|------|------|
| `src/matrix/tools/finance/__init__.py` | 去掉 `truncate_result` 包裹，去掉 import |
| `src/matrix/tools/web/__init__.py` | 去掉 `truncate_result` 包裹，去掉 import |
| `src/matrix/tools/truncate.py` | `_ARRAY_FIELDS` 加 `images`；`_TEXT_FIELDS` 加 `prompt`, `url` |

---

## Fix 2: tool_error() 落地 + 错误规范（Phase 3 遗留）

> **实现状态（2026-08-16）**：`tool_error()` 已定义并用于 fetch.py、holdings.py、weather.py、search.py；AGENTS.md 已追加错误信息规范 ✅

### 问题

1. `tool_error()` 函数已加但零调用
2. web 和 code 工具的错误消息没有"怎么改"的线索
3. 没有错误信息规范文档

### 设计

#### 2.1 错误信息规范写入 AGENTS.md

在 `personal-agent/AGENTS.md` 末尾追加：

```markdown
# 工具错误信息规范

工具错误信息必须让 LLM 能自主纠错，包含：
1. 什么错了（具体，不是笼统的"失败"）
2. 为什么错（如果能判断）
3. 怎么改（如果能建议）

优先使用 `tool_error()` 函数构造结构化错误返回。

示例：
- ✅ "持仓数据库不存在: {path}。请检查 MATRIX_CACHE_PATH 环境变量。"
- ✅ "asset_id 必须以 ast_ 开头，得到: {asset_id}。请先用 finance.asset_lookup 查找。"
- ❌ "查询失败"
- ❌ "数据库错误"
```

#### 2.2 改造 web/fetch.py 的错误路径

```python
# Before
return {"error": f"获取网页失败: {err}", "text": ""}

# After
from ..base import tool_error
return tool_error(
    "web_fetch", "获取网页",
    str(err)[:200],
    "请检查 URL 是否正确，或稍后重试。如果是搜索结果链接，请使用搜索摘要直接回答。",
    {"url": url},
)
```

#### 2.3 改造 code/executor.py 的错误路径

executor 的错误在 `execute_python` 返回值里，不在 `call()` 管道里。改造方式：

```python
# Before
"error_message": str(e),

# After — error_message 字段保持不变（它是结构化字段，UI 需要它），
# 但在 error_type 为非空时，在 error_message 里附上更多上下文
"error_message": f"{type(e).__name__}: {e}. 检查代码语法和导入是否正确。",
```

#### 2.4 改造 holdings.py

```python
# Before — 文件不存在时 shared.py 抛 FinanceToolError，被 registry 捕获
# After — holdings_summary 提前检查，返回 tool_error

def holdings_summary(cache_path: str = "") -> dict[str, Any]:
    path = Path(cache_path) if cache_path else Path("var/cache/finance.sqlite")
    if not path.exists():
        return tool_error(
            "finance.holdings_summary", "查询持仓",
            f"数据库文件不存在: {path}",
            "请检查 MATRIX_CACHE_PATH 环境变量，或确认 personal-os 已同步数据。",
            {"cache_path": str(path)},
        )
    conn = connect_readonly(path)
    ...
```

### 改动范围

| 文件 | 改动 |
|------|------|
| `AGENTS.md` | 追加错误信息规范 |
| `src/matrix/tools/web/fetch.py` | 3 处错误返回改用 `tool_error()` |
| `src/matrix/tools/finance/holdings.py` | 加文件存在检查 + `tool_error()` |
| `src/matrix/tools/code/executor.py` | `error_message` 附加上下文 |
| `src/matrix/tools/web/search.py` | 搜索失败错误加建议 |
| `src/matrix/tools/web/weather.py` | 天气获取失败加建议 |

---

## Fix 3: 事件系统迁移完成（Phase 6 遗留）

> **实现状态（2026-08-15）**：已完成 ✅。编排节点和 Runtime adapter 现在将结构化事件对象直接写入 queue，SSE 适配层统一通过 `to_dict()` 消费；保留 tuple 读取兼容逻辑，便于旧扩展平滑迁移。

### 问题

`events.py` 定义了 13 种结构化事件。此前 `_push_event` 仍然把 `(string, dict)` 元组写入 queue，导致结构化事件只停留在影子层；现在 queue 的内部写入已经统一为结构化事件对象，SSE 适配层仍输出兼容的事件字典。

### 设计

#### 3.1 迁移路径

事件流：`_push_event` → `queue.Queue` → `stream_chat` → `iter_events` → `sse_event` → 前端 `EventSource.addEventListener`

迁移需要改 4 层，但可以分两步：

**Step A（后端内部）**：queue 里放结构化事件，SSE 序列化时转 dict

```python
# _helpers.py — _push_event 改为直接 put 结构化事件
def _push_event(cfg, evt_type, payload):
    q = cfg.get("event_queue")
    if q is not None:
        event = make_event(evt_type, payload)  # 结构化事件
        try:
            q.put_nowait(event)  # 放 dataclass，不是 tuple
        except queue.Full:
            pass
```

```python
# chat/_service.py — _drain_queue 和 _emit_*_events 改为处理 dataclass
# _drain_queue 当前从 queue 取 (evt_type, payload) tuple
# 改为取 AgentSessionEvent dataclass，调用 .to_dict() 转换

def _drain_queue(q, emitted):
    while True:
        try:
            event = q.get_nowait()
        except queue.Empty:
            break
        if isinstance(event, AgentEvent):
            d = event.to_dict()
            evt_type = d.pop("type", "message")
            payload = d
        else:
            # Backward compat: still handle (string, dict) tuples
            evt_type, payload = event
        ...
```

```python
# chat route iter_events — 无需改
# 它已经从 stream_chat yield 的 dict 里取 type 和 payload
# 只要 stream_chat yield 的格式不变，route 层不用动
```

**Step B（前端）**：前端不需要改

前端用的是 `es.addEventListener('tool_call', ...)` — 它监听的是 SSE event name（字符串），不是事件类型 dataclass。`sse_event(event_type, payload_data)` 仍然用字符串 event_type。所以前端完全不受影响。

#### 3.2 关键认知

迁移**不需要改前端**。`sse_event(event_type, payload)` 的 `event_type` 仍然是字符串 — 它是 SSE 协议的 event name，不是 Python 类型。结构化事件的 `type` 字段就是这个字符串。迁移只改后端内部的事件传递方式（从 tuple 改为 dataclass），SSE 边界的序列化格式不变。

#### 3.3 要不要做？

**诚实评估**：迁移的价值主要是类型安全 — 防止 typo 产生未知事件类型。当前字符串方式在实际运行中没有出过问题。如果优先级不高，可以 defer。

**实现结果**：已完成 Step A（后端内部迁移），没有改动 SSE 协议和前端消费方式；旧 tuple 仍可被 `_drain_queue` 读取。

### 改动范围

| 文件 | 改动 |
|------|------|
| `src/matrix/orchestration/nodes/_helpers.py` | `_push_event` 改为 put dataclass |
| `src/matrix/orchestration/runtime_adapter.py` | Runtime tool event 改为 put dataclass |
| `src/matrix/chat/_utils.py` | `_drain_queue` 处理 dataclass + 兼容 tuple |

---

## Fix 4: 会话树前端集成（Phase 7 遗留）

> **实现状态（2026-07-29）**：后端已完成 ✅ — `get_history` 已返回 `message_id`（`store.py`），branch/branches/leaf 三个端点已实现（`sessions.py`）；前端已实现。

### 问题

后端有 branch / branches / leaf 三个 API 端点，但前端没有任何代码调用它们。用户无法通过 UI 执行 branch 操作。

### 设计

#### 4.1 前端改动点

Matrix 前端是 React SPA。

**交互 A：消息悬停显示"从这里分叉"按钮**

在每条消息（user 或 assistant）的 hover 状态下，显示一个小图标按钮"从此处分叉"。点击后：

```javascript
async function branchFromMessage(sessionId, messageId) {
    const resp = await fetch(`/sessions/${sessionId}/branch`, {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ from_message_id: messageId }),
    });
    if (resp.ok) {
        // 清空当前消息列表，提示用户"已从该节点分叉，请输入新问题"
        clearMessages();
        showBranchBanner('已从分叉点开始新对话，旧分支保留在历史中');
    }
}
```

问题：前端目前不知道每条消息的 `message_id`。`get_history` 返回的是 `[{role, content}]`，没有 `message_id`。需要改 `get_history` 返回 `message_id`。

**交互 B：分支历史侧边栏**

在会话侧边栏中显示分支结构（如果有分叉的话）。调用 `GET /sessions/{id}/branches` 获取分叉点列表，展示为树形或列表。

#### 4.2 后端改动

`get_history` 需要返回 `message_id`：

```python
# store.py — get_history 改为返回 message_id
def get_history(self, session_id, max_turns=8):
    ...
    path.append({
        "role": msg_row[2],
        "content": msg_row[3],
        "message_id": msg_row[0],  # 新增
    })
    ...
```

`GET /sessions/{id}/messages` 端点返回的格式也要包含 `message_id`。

#### 4.3 复杂度评估

前端改动是主要工作量 — 需要：
1. 在消息渲染时存储 `message_id`
2. 添加 hover 按钮和 branch 交互
3. 添加 branch banner 提示
4. 可选：分支树可视化

后端改动很小 — `get_history` 加一个字段。

### 改动范围

| 文件 | 改动 |
|------|------|
| `src/matrix/store.py` | `get_history` 返回 `message_id` |
| `src/matrix/server/routes/sessions.py` | `get_messages` 透传 `message_id` |
| `src/matrix/web/src/App.tsx` | 消息渲染存 message_id + hover 分叉按钮 + branch 交互 |
| `tests/test_server.py` | 会话消息包含 message_id 的测试 |

---

## Fix 5: 提示词组装缓存（Phase 5 遗留）

> **实现状态（2026-07-29）**：已完成 ✅。`context_loader.py` 已实现 `_cache` dict + `_CACHE_TTL = 60`。

### 问题

`load_project_context_files()` 每次调用都读磁盘。在多轮对话中，同一个 AGENTS.md 被反复读取。

### 设计

加一个简单的 TTL 缓存：

```python
# src/matrix/orchestration/context_loader.py

import time

_cache: dict[str, tuple[float, list[tuple[Path, str]]]] = {}
_CACHE_TTL = 60  # seconds

def load_project_context_files(cwd: Path | None = None) -> list[tuple[Path, str]]:
    if cwd is None:
        cwd = Path.cwd()

    cache_key = str(cwd)
    now = time.time()

    cached = _cache.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    # ... 实际读取逻辑 ...

    _cache[cache_key] = (now, results)
    return results
```

### 改动范围

| 文件 | 改动 |
|------|------|
| `src/matrix/orchestration/context_loader.py` | 加 `_cache` dict + TTL 检查 |

---

## 历史实施优先级（已完成）

| Fix | 改动量 | 收益 | 建议 |
|-----|--------|------|------|
| Fix 1: 截断入口统一 | 极小（3 文件，去包裹 + 加字段） | 中（架构干净 + 覆盖 agnes） | ✅ 已完成 |
| Fix 5: 提示词缓存 | 极小（1 文件，加缓存） | 低（性能优化） | ✅ 已完成 |
| Fix 2: tool_error 落地 | 小（5-6 文件改错误路径） | 中（模型纠错能力提升） | ✅ 已完成 |
| Fix 3: 事件迁移 | 中（3 文件改事件流） | 低（类型安全，当前无 bug） | ✅ 已完成 |
| Fix 4: 前端集成 | 中大（后端小改 + 前端 100+ 行 JS） | 高（用户可用 branch 功能） | ✅ 已完成 |

## 阶段 8：Runtime 状态与审批历史查询

> **实现状态（2026-08-15）：已完成 ✅**

Runtime 的 durable snapshot、approval 和 event 已经可以按当前用户、会话和状态查询，供调试和 UI 恢复使用。查询层只返回用户可见字段，不暴露 owner_id 或完整 Runtime state 快照。

### 接口

    GET /api/runtime/operations?session_id=&phase=&limit=
    GET /api/runtime/approvals?session_id=&status=&limit=
    GET /api/runtime/operations/{operation_id}/events?limit=

personal-os 通过 /api/agent/runtime/* 代理这些只读接口；macOS App 在 AI 助理侧栏展示当前会话最近的 Runtime phase 和审批记录数量。原始隐式思维链仍不落盘，DebugTrace 仍只属于当前运行。

### 设计约束

- Store 查询必须带 owner_id，会话过滤在 SQLite JOIN 内完成。
- operation/approval 的创建与恢复流程不改变，查询只读，不参与状态转移。
- Runtime 只依赖自己的 domain/ports；HTTP 路由依赖 Runtime store，但 Runtime 不反向依赖 FastAPI、LangGraph 或 Agent Registry。

## 阶段 9：macOS 会话分支入口

> **实现状态（2026-08-15）：已完成 ✅**

会话树后端原有的 branch 能力已接入当前主 UI（macOS App）：历史消息保留 `message_id`，用户可在消息上下文菜单选择“从此消息分叉”，分叉后重新读取当前 leaf。personal-os 同步代理 POST `/api/agent/sessions/{session_id}/branch`。

本阶段只复用现有 append-only 会话树，不在同步 branch 请求中生成摘要、不删除旧分支，也不改 Runtime 或 LangGraph 边界；异步摘要随后作为阶段 10 独立实现。

## 阶段 10：分支摘要

> **实现状态（2026-08-15）：已完成 ✅**

分支切换后，SessionStore 先识别被放弃分支的消息范围；ChatService 再通过后台 Pipeline LLM 生成摘要。摘要生成不阻塞分支切换，失败会以失败状态写入 `session_entries`，并保留可诊断错误。

### 接口与展示

```text
GET /sessions/{session_id}/branch-summaries
```

摘要以 `branch_summary` entry 保存，仅持久化摘要、关键点、未解决问题和分支标识，不持久化模型原始隐式思维链。macOS App 在 AI 助理侧栏展示最近成功的分支摘要。

### 边界

- 分支切换仍是 append-only，不删除或覆盖旧消息。
- 摘要生成失败不回滚 leaf，也不阻塞下一轮对话。
- 摘要功能位于 Session/Chat 应用层，不进入独立 Runtime 核心，不引入对 LangGraph 的反向依赖。

## 阶段 11：Runtime 进程重启恢复与异常闭环

> **实现状态（2026-08-16）：已完成 ✅**

服务启动时，Runtime Store 会扫描上次进程留下的非终态操作，并按“可安全恢复”与“不可安全重放”分类：

- `waiting_approval` 保留原状态，用户仍可通过现有审批入口恢复；
- `preparing`、`requesting_model`、`executing_tools`、`preparing_next_turn`、`resuming` 等中间态统一转为 `recovery_required`；
- 状态更新与 `recovery_required` durable event 在同一 SQLite 事务中提交，写入前一阶段、原因和恢复时间；
- 不自动重放工具调用，避免进程崩溃发生在外部 effect 已执行但 journal 尚未结算时造成重复副作用；
- 恢复操作具备幂等性，重复启动不会重复追加恢复事件或改变已终止状态。

该策略保持 Runtime 的单向依赖：恢复分类由 Runtime Store/Domain 完成，ChatService 只负责启动时触发扫描和记录日志；不引入对 LangGraph、FastAPI 或 Agent Registry 的反向依赖。
