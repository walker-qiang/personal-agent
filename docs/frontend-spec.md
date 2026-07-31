# 前端功能 Spec

> 本文档描述当前 React 前端和旧版 HTML 资源的功能边界，作为"前端长什么样"的单一事实来源。
> 出问题时对照本文档检查：功能是否还在、行为是否正常。

## 纯 HTML 前端（历史兼容资源）

- **文件**：`src/matrix/server/static/index.html`
- **服务路径**：不再作为默认页面入口
- **定位**：历史实现和静态兼容资源
- **技术**：纯 HTML/CSS/JS，零外部框架，依赖 `marked.min.js`

### 布局

- 三栏布局：左侧边栏（会话列表）| 中间（对话区）| 右侧面板（信息面板）
- 三个可拖拽分隔线：sidebar ↔ main、main ↔ right panel、底部 trace 面板
- 面板宽度持久化到 localStorage
- 顶部栏：左侧标题 "Matrix"，右侧用户头像 + 下拉菜单

### 认证

- 登录/注册弹窗：Tab 切换登录/注册，用户名 + 密码表单
- 用户菜单：显示用户名，下拉菜单含"登出"
- JWT token 存储在 localStorage，每次请求通过 `Authorization: Bearer` 发送

### 模型选择

- 点击左下角模型名称打开模型选择菜单
- 从 `/api/provider` 加载可用模型列表
- 切换后 POST 到 `/api/provider` 保存选择

### 会话管理

- 左侧边栏显示会话列表，含删除按钮
- "新建对话"按钮创建新会话
- 点击会话切换到对应对话，自动加载历史消息
- 会话列表通过 `/sessions` 加载

### 对话

- 输入框 + 发送按钮，支持 Enter 发送
- SSE 流式对话（`/chat`），实时渲染 token
- 消息气泡：用户消息（右对齐）、助手消息（左对齐，含 Markdown 渲染）
- 工具调用/结果：可折叠展开，显示工具名、参数、结果、耗时
- 错误块：红色背景显示错误信息
- 状态指示：思考中/执行中/生成中
- 消息历史：切换会话时通过 `/sessions/{id}/messages` 加载

### 技能管理

- 技能列表弹窗：展示所有技能，含运行/编辑/删除按钮
- 技能编辑弹窗：name、description、prompt、workflow、output_format 编辑
- 新建技能：同上表单
- 技能文件编辑器：为技能添加/编辑/删除 knowledge 文件
- 运行技能：发送技能名作为消息，触发 agent 执行

### MCP 服务器管理

- MCP 列表弹窗：展示所有 MCP 服务器，含启用/禁用开关、编辑、删除
- MCP 编辑弹窗：name、transport（stdio/sse/streamable_http）、command、args、env、url、headers
- 启用/禁用切换：POST `/api/mcp/{name}/toggle`
- 新建/删除 MCP 服务器

### Trace 查看器

- 底部可拖拽 Trace 面板
- 会话列表：显示所有有 trace 的会话
- 详情视图：展示事件列表（工具名、参数、结果预览、耗时、错误）
- 返回列表按钮

### 文件上传

- 输入框旁的附件按钮，选择文件后上传到 `/api/upload`
- 上传后获得 `file_id`，随消息发送

### 右侧信息面板

- 待办事项区域
- 参考信息区域
- 任务产物区域

---

## React SPA 前端（当前默认 UI）

- **源码**：`src/matrix/web/`
- **构建产物**：`src/matrix/server/static/react-app/`
- **服务路径**：`/` 和 `/react-app/`
- **定位**：当前日常对话交互界面
- **技术**：React 18, TypeScript, Vite 5, marked

### 组件清单

| 组件 | 功能 |
|------|------|
| `LoginOverlay` | 登录/注册弹窗，Tab 切换 |
| `SessionList` | 左侧会话列表，右键菜单/长按删除 |
| `MessageBubble` | 消息气泡，Markdown 渲染，流式光标，错误提示 |
| `AgentChain` | 多 Agent 执行链可视化，pending/running/done/error 状态 |
| `ToolSection` | 工具调用结果展示，可折叠，显示耗时和错误 |
| `SkillPanel` | 右侧技能列表，支持发送/编辑/删除/新建 |
| `SkillEditor` | 技能编辑弹窗（name/description/prompt/workflow/output_format） |
| `RightPanel` | 信息面板（待办/任务产物/参考信息，目前为空状态） |
| `ModelSelector` | 模型选择下拉，分组展示（对话/图片/视频模型） |
| `QuickQuestions` | 5 个预设快捷问题按钮 |
| `StatusBar` | 底部状态栏，4 种状态动画（idle/thinking/executing/generating） |
| `FileUpload` | 文件上传，支持拖拽和点击，图片预览 |
| `ConfirmDialog` | HITL 确认弹窗，approve/skip |

### 页面布局

- 左侧边栏（220px）：会话列表 + 快捷问题
- 中间：对话区（消息列表 + 输入框）
- 右侧面板：技能面板 + 信息面板
- 底部：状态栏
- 顶部：标题栏

### 功能清单

- 登录/注册：用户名 + 密码，JWT token 管理
- 会话 CRUD：创建/选择/删除，localStorage 持久化当前会话
- 技能 CRUD：列表/创建/编辑/删除，点击技能发送到对话
- 模型选择：从 `/api/provider` 加载，下拉分组选择
- 对话：SSE 流式，Markdown 渲染，工具调用/结果展示
- Agent Chain：可视化多 Agent 执行步骤
- HITL 确认：危险操作弹窗确认
- 文件上传：拖拽 + 点击，图片预览
- 快捷问题：5 个预设场景
- 流式 thinking/progress 展示
- 状态栏动画

### 与纯 HTML 前端的功能差异

| 功能 | 纯 HTML | React |
|------|:---:|:---:|
| 登录/注册 | ✅ | ✅ |
| 会话列表 | ✅ | ✅ |
| 技能 CRUD | ✅ | ✅ |
| 技能运行 | ✅ | ✅ |
| SSE 流式对话 | ✅ | ✅ |
| 消息历史 | ✅ | ✅ |
| Markdown 渲染 | ✅ | ✅ |
| 工具调用/结果 | ✅ | ✅ |
| 模型切换 | ✅ | ✅ |
| 文件上传 | ✅ | ✅ |
| 右侧信息面板 | ✅ | ✅ |
| MCP 服务器管理 | ✅ | ✅ |
| Trace 查看器 | ✅ | ✅ |
| 技能文件编辑器 | ✅ | ✅ |
| 可拖拽面板 | ✅ | ✅ |
| Agent Chain 可视化 | ❌ | ✅ |
| HITL 确认弹窗 | ❌ | ✅ |
| 快捷问题 | ❌ | ✅ |
| 状态栏动画 | ❌ | ✅ |
| 流式 thinking/progress | ❌ | ✅ |
| 文件拖拽上传 | ❌ | ✅ |

---

## 构建与部署

### 纯 HTML 前端

无需构建，直接修改 `static/index.html`。修改后重启服务生效。

### React SPA

```bash
uv sync
cd src/matrix/web
npm ci
npm run build
```

构建产物输出到 `static/react-app/`，通过 `/react-app/` 路径访问。

React 构建产物同时作为根路径 `/` 的默认页面。后端服务启动前必须先完成一次构建；产物目录被 Git 忽略，源码变更后需要重新执行 `npm run build` 并重启服务。

**注意**：构建输出目录 `static/react-app/` 已在 `.gitignore` 中排除，不会进入版本控制。构建产物不影响纯 HTML 前端。

### 功能一致性

React 已覆盖旧版 HTML 的登录、会话、归档、分支、技能、技能文件、模型、文件上传、流式对话、思考过程、工具结果、Agent Chain、HITL、MCP、Trace、快捷问题和可拖拽面板等功能。旧版 HTML 保留在 `static/index.html`，用于迁移期间的对照和回退，不作为默认入口。

### 启动服务

```bash
cd personal-agent
python3 -m matrix
# 或
./scripts/dev.sh
```

服务默认监听 `127.0.0.1:7101`。

### UI 快照

```bash
./scripts/ui-snapshot.sh status     # 检查前端是否与快照一致
./scripts/ui-snapshot.sh snapshot   # 更新快照（确认前端 OK 后保存）
./scripts/ui-snapshot.sh restore    # 从快照恢复
```

快照存储在 `scripts/snapshots/ui/`，纳入版本控制。
