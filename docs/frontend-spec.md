# 前端功能 Spec

> 本文档描述 React 前端的功能边界，作为“前端长什么样”的单一事实来源。
> 出问题时对照本文档检查：功能是否还在、行为是否正常。

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

---

## 构建与部署


### React SPA

```bash
uv sync
cd src/matrix/web
npm ci
npm run build
```

构建产物输出到 `static/react-app/`，通过 `/react-app/` 路径访问。

React 构建产物同时作为根路径 `/` 的默认页面。后端服务启动前必须先完成一次构建；产物目录被 Git 忽略，源码变更后需要重新执行 `npm run build` 并重启服务。


### 功能一致性

React 已覆盖登录、会话、归档、分支、技能、技能文件、模型、文件上传、流式对话、思考过程、工具结果、Agent Chain、HITL、MCP、Trace、快捷问题和可拖拽面板等全部功能。

### 启动服务

```bash
cd personal-agent
python3 -m matrix
# 或
./scripts/dev.sh
```

服务默认监听 `127.0.0.1:7101`。

