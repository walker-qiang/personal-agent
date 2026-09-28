# personal-agent 记忆系统优化计划

> 制定日期：2026-09-27
> 对齐标准：*Memory in the Age of AI Agents*（arXiv 2512.13564）三轴 taxonomy；Mem0 / Zep-Graphiti / MemOS / MIRIX / MemoryOS 一代实现
> 范围：`personal-agent` 记忆模块。vault（`personal-assets`）仍为 source of truth，架构宪法不变。

## 实施状态（2026-09-27 已全部落地）

Phase 0–5 均已实现，905 → 911 项测试通过。新增模块：

| 文件 | 作用 |
|---|---|
| `memory/writer.py` | 有界后台 worker，替换裸 daemon thread（B-2） |
| `memory/retriever.py` | 向量 + BM25 混合检索，纯 Python BM25，无新依赖 |
| `memory/prompting.py` | policy 全量 + preference top_k + token 预算截断 |
| `memory/decisions.py` | 写入前 ADD/UPDATE/DELETE/NONE 语义判定 |
| `memory/temporal.py` | 事实时间解析 + 相对时间区间解析 |
| `memory/safety.py` | 密钥 / 注入载荷拦截 + PII 脱敏 |

`store.py` 新增列：`source_session_id`、`source_quote`、`confidence`、`valid_from`、
`valid_to`、`fact_time`、`access_count`、`last_accessed_at`、`scope`、`scope_id`，
以及 `memory_embeddings` 表与 messages 的 FTS5 索引。全部为 additive migration，老库可直接升级。

新增 API：`POST /memory/recall`、`GET /memory/stats`，`/memory/list` 返回完整 provenance。

Vault wire format 已升级为 typed profile：`personal-os` 写入
`{"value": "...", "memory_type": "policy|preference"}`，同时兼容读取旧扁平 KV；
Agent 回灌和删除/演化同步均保留类型。旧格式只会按 preference 解释，但不会覆盖
SQLite 中已有的 policy。

2026-09-28 验证：Agent memory focused tests 66 项通过，personal-os API/AssetStore
测试通过，隔离跨仓 E2E 通过；旧 flat payload 经 API 接收后会规范化为 typed JSON。

---

## 0. 现状诊断（本计划的出发点）

### 0.1 已实现的四层

| 层 | 实现 | 位置 |
|---|---|---|
| 工作记忆 | `memory_max_turns=8` + L3 compaction（85% 触发，5 段结构化 handoff）+ token budget | `context/compaction.py`、`context/budget.py` |
| 语义记忆 | `user_profile` KV 表，preference / policy 两类，30 天半衰指数衰减，120 天剔除 | `store.py:645-762` |
| 程序性记忆 | `agent_lessons`（失败教训），Jaccard 中文 2-gram 检索，top_k=3 注入 | `memory/lesson_store.py` |
| 情景记忆 | sessions / messages 表，**只按 session 读，无跨会话检索** | `store.py` |

写入路径：`_remember` → 后台线程 `MEMORY_EXTRACTION_PROMPT` 抽取 → `upsert_profile` → `MemoryEvolution` 四阶段 → `sync_memory_profile` 回写 vault。

### 0.2 实测证据（决定是否先止血的关键）

- `var/agent/sessions.db`：`user_profile` = **0 行**（sessions=18、messages=42）
- `var/agent/lessons.db`：**不存在** —— LessonStore 从未落盘
- `var/log/` 全部日志：`memory_extraction` / `memory_upsert` / `memory_evolved` / `memory_sync` **零命中**
- vault `92-系统/memory/admin.json` 8 条，其中 3 条语义重复（`investment_research_data_preference` / `research_method_preference` / `market_research_tool`）
- `research_date: 2026-08-16` 这类时点事实被当作 preference 做 30 天衰减，120 天后消失

**结论**：跑了 18 个会话没有沉淀任何一条记忆，且系统没有留下任何可供定位的痕迹。在补齐可观测性之前，任何架构改造都是盲改。

### 0.3 已确认的确定性缺陷

| 编号 | 缺陷 | 位置 | 影响 |
|---|---|---|---|
| B-1 | `sync_profile_from_file` 调用 `upsert_profile` 不传 `memory_type`，默认落 `preference` | `store.py:804-818` | vault 回灌时 **policy 降级为 preference**，硬约束失效且进入衰减队列 |
| B-2 | 抽取在 `threading.Thread(daemon=True)` 中执行，异常仅 `logger.warning`，无重试无上报 | `_service.py:2565-2571`、`:2572-2600` | 失败不可见；进程退出时线程被直接丢弃 |
| B-3 | 启动同步仅在 `count > 0` 时打日志 | `server/app.py:242-250` | 路径错 / 文件空 / 解析失败三种情况完全静默 |
| B-4 | `_find_similar` 与 `get_relevant_lessons` 只取最近 50 / 100 行做全表扫描 | `lesson_store.py:296-301`、`:376-389` | 超过窗口的教训永不参与匹配，且 O(n) 逐行算 Jaccard |

---

## 1. 分期计划

### Phase 0 — 止血与可观测（P0，0.5–1 天）

目标：让记忆链路"看得见"，修掉确定性 bug。不做任何架构变更。

| 任务 | 动作 | 验收 |
|---|---|---|
| P0-1 | 修 B-1：vault JSON 支持带 `memory_type` 的结构，回灌时透传；旧格式（纯 KV）保持默认 preference 但不覆盖已有 policy | vault 中一条 policy 重启后仍为 policy |
| P0-2 | 补埋点：`memory_extract_start/success/fail(reason)`、`memory_upsert(count)`、`memory_sync(skipped_reason)`、`evolution(report)`，全部 INFO 级 | 一次对话后日志能完整复现记忆链路 |
| P0-3 | 抽取线程改造：改用带界队列的后台 worker（或 `ThreadPoolExecutor`），异常结构化上报并设置超时；进程退出前 drain | 杀进程不再丢抽取任务；失败有明确 cause |
| P0-4 | 修复 B-3：同步结果无论成败都打日志，含 resolved path | 路径错误 1 秒内可定位 |
| P0-5 | 定位 profile=0 的真正根因并修复（疑似 pipeline LLM 不可用或 `_remember` 未被触发） | 连续 5 轮对话后 profile ≥ 1 条 |

**退出条件**：跑一轮真实对话，日志里能看到完整的 `extract → upsert → sync` 链路，profile 表有数据。

---

### Phase 1 — 检索层改造（P1，2–3 天）

目标：把"全量注入"改成"按需检索"，这是投入产出比最高的一项。

现状：`get_profile_formatted`（`store.py:696`）把全部记忆拼进 system prompt，上限 80 条。记忆增长后 token 不可控，且长 prompt 触发 lost-in-the-middle。

| 任务 | 动作 |
|---|---|
| P1-1 | 新增 `MemoryRetriever`：复用已有 `rag/embedder.py` 的 `LocalEmbedder` + `rag/bm25.py`，做向量 + BM25 混合召回 |
| P1-2 | 建 `memory_embeddings` 表（或独立 Chroma collection），在 `upsert_profile` 时同步写索引，加幂等重建命令 |
| P1-3 | 注入策略改为三段：policy **全量注入**（量小、是硬约束）+ preference **检索 top_k** + 兜底"高频未命中"少量注入 |
| P1-4 | 注入内容加 token 预算上限（建议 800 token），超出按 importance 截断 |
| P1-5 | 保留全量注入开关，用于回归对比 |

**验收**：构造 80 条记忆，单轮注入 token 数较改造前下降 ≥ 60%；人工 20 条召回测试用例命中率 ≥ 90%。

**风险**：LocalEmbedder 首次加载有冷启动开销 → 复用 skills router 已有的加载模式（`skills/router.py:122`），启动时预热或懒加载。

---

### Phase 2 — 写入侧决策（P1，2–3 天）

目标：用 write-time 决策取代"先无脑 upsert、事后靠 evolution 补救"。

现状：evolution 的冲突检测是布尔词表（`evolution.py:147-176` 的 positive/negative 集合），对"研究对象从茅台换成五粮液"这类语义冲突完全无感。且**每轮对话都跑一遍 O(n²) 全量比对**。

| 任务 | 动作 |
|---|---|
| P2-1 | 抽取 prompt 升级为 Mem0 式四操作判定：对候选记忆与已有近邻记忆输出 `ADD / UPDATE / DELETE / NONE` |
| P2-2 | 冲突判定交给 pipeline LLM 做语义判断，布尔词表降级为快速前置过滤 |
| P2-3 | `upsert_profile` 扩展字段：`source_session_id`、`source_quote`、`confidence` |
| P2-4 | `MemoryEvolution` 从"每轮执行"降级为低频批处理（每日一次或 count 超阈值触发），避免每轮 O(n²) |
| P2-5 | 保留 heuristic 路径作为 LLM 不可用时的降级 |

**验收**：给定"我之前研究茅台，现在改研究五粮液"的连续对话，旧条目被 UPDATE 而非并存；单轮 LLM 调用次数下降。

---

### Phase 3 — 时间维度（P2，2 天）

目标：从"只有 updated_at + 硬删除"升级到可回答时间相关问题。

| 任务 | 动作 |
|---|---|
| P3-1 | `user_profile` 加列：`valid_from`、`valid_to`、`fact_time`（事实发生时间，区别于写入时间） |
| P3-2 | 删除改为失效标记（soft delete），`valid_to` 置时间戳，不物理删除 |
| P3-3 | 含明确时间的事实（如 `research_date`）标记为 `dated_fact`，豁免指数衰减 |
| P3-4 | 检索层支持时间范围过滤（"上周""三个月前"） |

**验收**：能正确回答"我三个月前在研究哪只股票"，且该事实在 120 天后仍可检索。

---

### Phase 4 — 程序性与情景记忆（P2，3–4 天）

目标：补上覆盖度最低的两块（当前 20% / 25%）。

| 任务 | 动作 |
|---|---|
| P4-1 | **修 LessonStore 落盘**：确认 `var/agent/lessons.db` 未生成原因，补启动自检与建表断言 |
| P4-2 | **成功路径也沉淀**：当前只从 reflection 失败分支抽取（`commander.py:1042-1085`），扩展为成功策略同样入库，形成可复用 playbook |
| P4-3 | **情景记忆**：`messages` 表加 FTS5 + 向量索引，提供跨会话 recall API（"上次我们聊到哪"） |
| P4-4 | 修 B-4：`_find_similar` 去掉 `LIMIT 50` 的截断，改用索引 + 候选集粗筛 |
| P4-5 | skills 自演化：高命中 lesson / playbook 可提名为 skill 候选，人工确认后入库 |

**验收**：`lessons.db` 有数据；能检索到 3 个会话之前的对话内容；成功策略可被后续同类型任务命中。

---

### Phase 5 — 治理、评测与安全（P3，2–3 天）

目标：让记忆系统的改进可量化、可审计、可防守。

| 任务 | 动作 |
|---|---|
| P5-1 | **自建评测集**：从真实会话构造 LoCoMo 风格 QA 对（单跳 / 多跳 / 时间推理 / 开放域），作为回归基线 |
| P5-2 | 每次记忆改动跑评测，落 `tests/baselines/`，防退化 |
| P5-3 | **记忆投毒防护**：抽取入库前过 PII 过滤 + 来源可信度校验；policy 类写入需用户确认 |
| P5-4 | 记忆可追溯：UI 展示每条记忆的来源会话、写入时间、置信度，支持单条回滚 |
| P5-5 | 作用域扩展：`user / session / agent` 三级（Mem0 的最低档），为后续 project 级留口 |

---

## 2. 排期与依赖

```
Phase 0（止血）──┬── Phase 1（检索）──┬── Phase 3（时间）
                 │                    └── Phase 4（程序/情景）
                 └── Phase 2（写入）────── Phase 5（评测/治理）
```

- **Phase 0 是硬前置**：不做 P0，后续所有改动无法验证。
- Phase 1 与 Phase 2 可并行（改动文件不重叠：检索在 `store.py` 读路径 + 新 `MemoryRetriever`；写入在 `_extract_memories` + `evolution.py`）。
- Phase 3 依赖 Phase 1（时间过滤需要走检索层）。
- Phase 5 的评测集建议在 Phase 1 开始前先建最小版，以便量化 Phase 1 的收益。

## 3. 不做的事（边界）

- 不引入外部向量数据库（Qdrant / Milvus）：个人场景 SQLite + 本地 embedding 足够，避免运维负担。
- 不替换 vault 为 source of truth 的架构：SQLite 仍是运行态缓存，vault 仍是可审计的持久源。
- 不做 parametric memory（MemOS 的 KV-cache / LoRA 路线）：与当前 single-user 场景不匹配。
- 不追求 benchmark 刷分：自建评测集用于防退化，不用于对外比较。

## 4. 优先级判断依据

1. **P0 止血** — 当前系统实际产出为 0，先让它跑起来再谈优化。
2. **P1 检索** — 全量注入是唯一随规模线性恶化的问题，且改造面收敛在注入点。
3. **P1 写入** — 决定记忆质量上限，也是 vault 里已出现重复条目的根因。
4. **P2 时间 / 类型补齐** — 提升能力天花板，但不阻塞当前可用性。
5. **P3 治理** — 长期工程，可在前四期收益兑现后铺开。
