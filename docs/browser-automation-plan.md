# 浏览器自动化工具集成设计方案

## 1. 背景与目标

### 问题

当前 personal-agent 的 `web_fetch` 工具基于 `urllib` 实现，只能抓取静态 HTML，存在四个关键限制：

| 限制 | 影响 |
|------|------|
| 无法点击 | 无法触发按钮、链接、分页等交互 |
| 无法填表 | 无法输入文本、选择下拉框、提交表单 |
| 无法登录 | 无 Cookie/Session 管理，无法访问需认证的页面 |
| 无法处理动态渲染 | 只拿到初始 HTML，SPA 页面内容为空 |

### 目标

将浏览器自动化能力封装为 personal-agent 的工具，让 agent 按需调用，覆盖以下场景：

- 动态渲染页面抓取（SPA、AJAX 加载内容）
- 表单填写与提交（搜索、筛选、登录）
- 多步骤页面交互（翻页、展开、切换 Tab）
- 登录态页面访问（Cookie/Session 持久化）

### 约束

- 遵循 AGENTS.md：所有工具默认只读，写操作需走受控接口
- 不修改 `personal-os` 代码
- 工具调用必须记录审计日志
- 代码风格：类型注解、dataclass、无隐式全局状态

---

## 2. 方案选型

### 方案对比

| 维度 | 方案 A：Playwright MCP 服务器 | 方案 B：原生工具模块 |
|------|------|------|
| 框架改动 | 零改动，复用现有 MCP Client | 需新增 `tools/browser/` 模块 |
| 进程隔离 | 浏览器崩溃不影响 agent | 浏览器崩溃可能影响 agent |
| async 桥接 | FastMCP 原生支持 async，Playwright async API 天然契合 | 需自行处理 sync/async 桥接（与 MCP Client 类似的线程方案） |
| 一致性 | 与 `utility_tools.py` 模式一致 | 与 `web/fetch.py` 模式一致 |
| 安全控制 | 通过 ToolGuard + MCP 配置控制 | 可直接集成 HITL、ToolGuard |
| 工具命名 | `mcp_browser_*`（受 MCP 前缀约束） | `browser_*`（更简洁） |
| 调试 | 可独立运行 MCP 服务器调试 | 需在 agent 上下文中调试 |
| 依赖 | Playwright 加入 MCP 服务器进程 | Playwright 加入主项目依赖 |

### 选定方案：A（Playwright MCP 服务器）

**核心理由**：

1. **零框架改动**：现有 MCP Client 基础设施（`client.py` → `adapter.py` → `registry.py`）完整可用，只需添加配置和 MCP 服务器脚本
2. **进程隔离**：浏览器自动化是最容易崩溃的操作（页面无响应、JS 错误、内存泄漏），独立进程保证 agent 稳定性
3. **async 天然契合**：FastMCP 支持 async tool 函数，Playwright async API 可直接使用，无需额外的线程桥接
4. **模式一致**：与 `utility_tools.py` 完全相同的模式，维护成本低
5. **可独立调试**：MCP 服务器可独立运行和测试，不依赖 agent 服务

---

## 3. 架构设计

### 整体架构

```
personal-agent (FastAPI)
  │
  ├── ToolRegistry
  │     ├── web_search / web_fetch / news_search  (原生工具)
  │     ├── code.run_python                       (原生工具, HITL)
  │     ├── finance.*                              (原生工具)
  │     ├── mcp_utility_*                          (MCP: utility_tools.py)
  │     └── mcp_browser_*                          (MCP: browser_tools.py)  ← 新增
  │
  └── MCPClientManager (后台 asyncio 线程)
        ├── stdio → utility_tools.py (utility 服务器)
        └── stdio → browser_tools.py (browser 服务器)  ← 新增
                       │
                       └── Playwright async API
                             └── Chromium (headless)
```

### 数据流

```
LLM 生成 tool_call: mcp_browser_navigate(url="...")
  → ToolRegistry.call("mcp_browser_navigate", {url: "..."})
    → ToolGuard.check()          (安全检查)
    → MCP adapter handler()
      → MCPClientManager.call_tool_sync("browser", "navigate", {url: "..."})
        → stdio JSON-RPC → browser_tools.py
          → Playwright page.goto(url)
          → 返回页面快照
        ← JSON-RPC response
      ← dict result
    ← IndirectInjectionGuard.check()  (结果安全扫描)
  → 返回给 LLM
```

### 浏览器实例管理

```
browser_tools.py (MCP 服务器进程)
  │
  ├── 模块级全局状态
  │     _playwright: Playwright 实例 (懒启动)
  │     _browser: Browser 实例 (单例)
  │     _page: Page 实例 (单 tab)
  │
  ├── 生命周期
  │     首次调用 → 启动 Playwright + Chromium
  │     后续调用 → 复用浏览器实例
  │     进程退出 → 自动清理
  │
  └── 超时回收 (可选, Phase 2)
        空闲 5 分钟 → 关闭浏览器 (保留 Playwright 运行时)
```

---

## 4. 工具集设计

### 设计原则

1. **中等粒度**：太细（如 Playwright 原始 API）导致步骤过多，太粗（如 `browse_and_extract`）灵活性不足
2. **LLM 友好**：返回结构化数据，元素引用使用简短 ref ID
3. **最小集优先**：Phase 1 只实现 6 个核心工具，覆盖 90% 场景

### Phase 1：核心工具集（6 个）

#### 4.1 `browser_navigate`

导航到指定 URL，返回页面基本信息和结构快照。

```python
@mcp.tool()
async def browser_navigate(url: str, wait_until: str = "domcontentloaded") -> dict:
    """导航到指定 URL 并返回页面快照。

    Args:
        url: 目标页面 URL（必须是 http/https）
        wait_until: 等待策略 (domcontentloaded | networkidle | load)

    Returns:
        {url, title, status, elements: [{ref, tag, text, role}]}
    """
```

**返回示例**：
```json
{
  "url": "https://example.com/search",
  "title": "搜索结果",
  "status": 200,
  "elements": [
    {"ref": "1", "tag": "input", "role": "searchbox", "text": ""},
    {"ref": "2", "tag": "button", "role": "button", "text": "搜索"},
    {"ref": "3", "tag": "a", "role": "link", "text": "下一页"}
  ]
}
```

#### 4.2 `browser_snapshot`

获取当前页面的完整元素树（用于 navigate 后页面发生变化时刷新引用）。

```python
@mcp.tool()
async def browser_snapshot() -> dict:
    """获取当前页面的可交互元素列表。

    Returns:
        {url, title, elements: [{ref, tag, text, role, attributes}]}
    """
```

#### 4.3 `browser_click`

点击页面上的元素。

```python
@mcp.tool()
async def browser_click(ref: str) -> dict:
    """点击页面上指定元素。

    Args:
        ref: 元素引用 ID（来自 snapshot 或 navigate 返回）

    Returns:
        {success, url_after, title_after, elements: [...]}
    """
```

#### 4.4 `browser_type`

在输入框中输入文本（可选提交）。

```python
@mcp.tool()
async def browser_type(ref: str, text: str, submit: bool = False) -> dict:
    """在指定输入元素中输入文本。

    Args:
        ref: 输入框元素引用 ID
        text: 要输入的文本
        submit: 是否在输入后按回车提交

    Returns:
        {success, url_after, title_after, elements: [...]}
    """
```

#### 4.5 `browser_extract`

提取当前页面的文本内容（支持 CSS 选择器或全文提取）。

```python
@mcp.tool()
async def browser_extract(selector: str = "", max_chars: int = 5000) -> dict:
    """提取当前页面的文本内容。

    Args:
        selector: CSS 选择器（为空则提取全文）
        max_chars: 最大返回字符数（默认 5000，上限 20000）

    Returns:
        {url, text, length}
    """
```

#### 4.6 `browser_screenshot`

截取当前页面的截图。

```python
@mcp.tool()
async def browser_screenshot(full_page: bool = False) -> dict:
    """截取当前页面的截图。

    Args:
        full_page: 是否截取完整页面（默认只截视口区域）

    Returns:
        {url, screenshot_base64, width, height}
    """
```

### Phase 2：扩展工具（按需添加）

| 工具 | 用途 | 优先级 |
|------|------|--------|
| `browser_wait_for` | 等待元素/文本出现 | 高 |
| `browser_press_key` | 模拟按键（Enter, Tab, Escape 等） | 中 |
| `browser_scroll` | 滚动页面（用于懒加载内容） | 中 |
| `browser_select_option` | 选择下拉框选项 | 中 |
| `browser_get_cookies` | 获取当前 Cookie（调试用） | 低 |
| `browser_run_js` | 执行 JavaScript（高风险，需 HITL） | 低 |

---

## 5. 安全设计

### 风险分级

| 风险等级 | 操作 | 示例 | 控制措施 |
|----------|------|------|----------|
| 低 | 只读操作 | navigate, extract, screenshot | ToolGuard 基础检查 |
| 中 | 交互操作 | click, type | ToolGuard + 操作日志 |
| 高 | 敏感操作 | 登录、表单提交、JS 执行 | HITL 确认 (Phase 2) |

### 安全控制层

```
Layer 1: ToolGuard (现有)
  → 参数大小限制 (10KB)
  → URL 协议白名单 (http/https only)
  → 每会话调用频率限制 (100 次)

Layer 2: MCP 服务器内置 (新增)
  → URL 黑名单 (file://, javascript:, data:)
  → 页面超时控制 (30s 加载, 10s 操作)
  → 输出截断 (文本 20000 字符, 截图 1MB base64)
  → 浏览器实例隔离 (每 MCP 进程独立)

Layer 3: HITL 确认 (Phase 2)
  → 交互操作 (click/type) 可选 HITL
  → JavaScript 执行强制 HITL
  → 表单提交强制 HITL
```

### URL 安全策略

```python
# browser_tools.py 中的 URL 校验
_BLOCKED_PROTOCOLS = {"file:", "javascript:", "data:", "chrome:", "about:"}
_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"}

def validate_url(url: str) -> tuple[bool, str]:
    """校验 URL 安全性。"""
    if not url or not url.strip():
        return (False, "URL 为空")
    for proto in _BLOCKED_PROTOCOLS:
        if url.lower().startswith(proto):
            return (False, f"协议被禁止: {proto}")
    # 阻止内网访问
    for host in _BLOCKED_HOSTS:
        if host in url:
            return (False, f"目标地址被禁止: {host}")
    return (True, "")
```

---

## 6. 配置设计

### MCP 服务器配置

在 `config/mcp_servers.json` 中添加 browser 服务器：

```json
{
  "servers": [
    {
      "name": "utility",
      "transport": "stdio",
      "enabled": true,
      "timeout": 10.0,
      "command": "/path/to/python3.10",
      "args": ["var/mcp/utility_tools.py"]
    },
    {
      "name": "browser",
      "transport": "stdio",
      "enabled": true,
      "timeout": 120.0,
      "command": "/path/to/python3.10",
      "args": ["var/mcp/browser_tools.py"],
      "env": {
        "BROWSER_HEADLESS": "true",
        "BROWSER_DEFAULT_TIMEOUT": "30000",
        "BROWSER_MAX_OUTPUT_CHARS": "20000",
        "BROWSER_USER_AGENT": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
      }
    }
  ]
}
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BROWSER_HEADLESS` | `true` | 是否使用无头模式 |
| `BROWSER_DEFAULT_TIMEOUT` | `30000` | 默认操作超时（毫秒） |
| `BROWSER_MAX_OUTPUT_CHARS` | `20000` | 文本输出最大字符数 |
| `BROWSER_USER_AGENT` | Chrome UA | 自定义 User-Agent |
| `BROWSER_VIEWPORT_WIDTH` | `1280` | 视口宽度 |
| `BROWSER_VIEWPORT_HEIGHT` | `720` | 视口高度 |

### 依赖管理

在 `pyproject.toml` 中添加 Playwright 依赖（仅 MCP 服务器需要）：

```toml
[project.optional-dependencies]
browser = ["playwright>=1.40.0"]
```

安装步骤：
```bash
# 1. 安装 Playwright Python 包
pip install playwright

# 2. 安装 Chromium 浏览器二进制文件
playwright install chromium
```

> Playwright 不加入主项目依赖，只在 MCP 服务器进程中使用。
> 如果未安装 Playwright，MCP 服务器启动时会优雅降级并记录警告日志。

---

## 7. Agent 集成

### 工具可用性

| Agent | tools 配置 | 浏览器工具 |
|-------|-----------|-----------|
| Commander | `[]` (空 = 所有工具) | ✅ 自动可用 |
| investment_analyst | `["finance.*", "web_search", ...]` | 需手动添加 `"mcp_browser_*"` |
| media_generator | `["agnes.*"]` | 不需要 |

### investment_analyst 集成

在 `agent/domain_agents/investment_analyst.py` 的 tools 列表中添加浏览器工具：

```python
# 现有
tools = ["finance.*", "web_search", "news_search", "web_fetch", "code.run_python"]

# 修改后
tools = [
    "finance.*", "web_search", "news_search", "web_fetch", "code.run_python",
    "mcp_browser_navigate", "mcp_browser_extract", "mcp_browser_snapshot",
]
```

> 只添加只读浏览器工具（navigate, extract, snapshot），不添加交互工具（click, type），保证投资分析场景的安全性。

### 工具选择策略

agent 如何在 `web_fetch` 和 `browser_*` 之间选择：

```
LLM 决策依据（通过工具 description 引导）：

web_fetch:
  "获取指定网页的完整文本内容。适用于：静态页面、文章全文、API JSON 响应。
   ⚠️ 无法处理需要 JavaScript 渲染的动态页面，无法交互。"

browser_navigate:
  "导航到 URL 并返回页面快照。适用于：动态渲染页面、需要交互的页面、
   SPA 应用、需要登录的页面。比 web_fetch 慢但功能更强大。"
```

---

## 8. 实现计划

### Phase 1：MVP（预计 3 小时）

**目标**：6 个核心工具可用，能完成基本的动态页面抓取。

| 步骤 | 文件 | 工作量 |
|------|------|--------|
| 1. 创建 MCP 服务器脚本 | `var/mcp/browser_tools.py` | 1.5h |
| 2. 添加 MCP 配置 | `config/mcp_servers.json` | 10min |
| 3. 安装 Playwright 依赖 | `pyproject.toml` + `playwright install` | 10min |
| 4. 端到端测试 | 手动验证 + 测试用例 | 1h |

**验收标准**：
- `browser_navigate` 能打开动态页面并返回元素列表
- `browser_extract` 能提取 JS 渲染后的内容
- `browser_click` + `browser_type` 能完成搜索操作
- `browser_screenshot` 能返回截图
- 崩溃恢复：浏览器崩溃后下次调用自动重启

### Phase 2：安全增强（预计 2 小时）

| 步骤 | 说明 |
|------|------|
| 1. HITL 集成 | 交互操作（click/type）标记为高风险，需用户确认 |
| 2. Cookie 持久化 | 支持登录态保存（`storageState`） |
| 3. 超时回收 | 空闲浏览器实例自动关闭 |
| 4. 扩展工具 | `browser_wait_for`, `browser_scroll`, `browser_press_key` |

### Phase 3：Agent 优化（预计 1.5 小时）

| 步骤 | 说明 |
|------|------|
| 1. Prompt 优化 | 在 agent system prompt 中添加浏览器工具使用指南 |
| 2. 评测用例 | 在 `eval_dataset.json` 中添加浏览器自动化测试用例 |
| 3. 性能优化 | 页面快照精简、不必要的资源拦截（图片/CSS） |

---

## 9. 测试计划

### 单元测试

```python
# tests/test_browser_tools.py
class TestBrowserNavigate:
    def test_navigate_static_page(self):
        """导航到静态页面，返回正确标题和元素"""

    def test_navigate_dynamic_page(self):
        """导航到 SPA 页面，JS 渲染后内容可见"""

    def test_navigate_blocked_protocol(self):
        """file:// 协议被拒绝"""

    def test_navigate_blocked_host(self):
        """内网地址被拒绝"""

class TestBrowserClick:
    def test_click_button(self):
        """点击按钮后页面发生变化"""

    def test_click_invalid_ref(self):
        """无效 ref 返回错误"""

class TestBrowserExtract:
    def test_extract_full_page(self):
        """提取全文内容"""

    def test_extract_by_selector(self):
        """通过 CSS 选择器提取"""

    def test_extract_max_chars(self):
        """超长内容被截断"""
```

### 端到端测试

```python
# tests/test_browser_e2e.py
class TestBrowserE2E:
    def test_search_and_extract(self):
        """完整流程：导航 → 输入 → 搜索 → 提取结果"""

    def test_dynamic_content(self):
        """动态渲染页面内容提取（对比 web_fetch 的差异）"""

    def test_browser_crash_recovery(self):
        """浏览器崩溃后自动恢复"""
```

### 评测用例

在 `tests/baselines/eval_dataset.json` 中添加：

```json
{
  "id": "browser-search-extract",
  "query": "搜索最新的 A 股市场行情并提取上证指数数据",
  "expected_tools": ["mcp_browser_navigate", "mcp_browser_type", "mcp_browser_click", "mcp_browser_extract"],
  "expected_pattern": "上证指数|沪指|3000|3100|3200|3300"
}
```

---

## 10. 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `var/mcp/browser_tools.py` | 新增 | Playwright MCP 服务器脚本 |
| `config/mcp_servers.json` | 修改 | 添加 browser 服务器配置 |
| `pyproject.toml` | 修改 | 添加 playwright 可选依赖 |
| `src/matrix/agent/domain_agents/investment_analyst.py` | 修改 | tools 列表添加浏览器工具 |
| `tests/test_browser_tools.py` | 新增 | 单元测试 |
| `tests/test_browser_e2e.py` | 新增 | 端到端测试 |
| `.gitignore` | 确认 | `var/` 已忽略（浏览器缓存不会提交） |

---

## 11. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Playwright 安装失败 | 中 | 阻塞 | MCP 服务器优雅降级，记录警告日志 |
| 浏览器内存泄漏 | 中 | 性能下降 | Phase 2 添加空闲超时回收 |
| 页面加载超时 | 高 | 工具调用失败 | 30s 超时 + 错误重试提示 |
| LLM 工具选择混乱 | 中 | 效率降低 | 通过 description 明确 web_fetch vs browser_* 的使用场景 |
| 工具数量过多影响 LLM | 低 | 准确率下降 | Phase 1 只注册 6 个工具；agent 按需过滤 |
