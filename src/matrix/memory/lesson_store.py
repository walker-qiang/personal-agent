"""Cross-session lesson store: persist failure experiences across conversations.

设计理念:
    传统 Reflexion 只在单次会话内积累自省经验. 会话结束后, 这些经验丢失,
    下次遇到类似问题时会重复同样的错误.

    LessonStore 将失败经验持久化到 SQLite, 并在新会话中注入相关教训,
    实现"不犯同样的错误"的跨会话学习能力.

核心组件:
    ┌──────────────────────────────────────────────────────────┐
    │  Lesson                                                  │
    │  数据类: task_pattern + failure_type + lesson_text       │
    │  + agent_id + severity + occurrence_count               │
    ├──────────────────────────────────────────────────────────┤
    │  LessonStore                                             │
    │  SQLite 持久化 + 关键词匹配检索                           │
    │  ┌─ record_lesson(task, failure_type, lesson, agent_id)  │
    │  ├─ get_relevant_lessons(task, agent_id, top_k)          │
    │  ├─ get_all_lessons(agent_id, limit)                     │
    │  └─ delete_lesson(lesson_id)                             │
    └──────────────────────────────────────────────────────────┘

数据流:
    1. reflection_node 检测到回答不足 → 提取教训 → record_lesson()
    2. Runtime agent / _run_domain_agent_react → get_relevant_lessons()
       → _inject_lessons() 注入 system prompt

设计权衡 (个人使用场景):
    - 使用关键词匹配而非 embedding 相似度 (避免额外模型加载)
    - 单表 SQLite, 与 SessionStore 共享数据库文件
    - 教训数量上限 200 条, 超出按 occurrence_count + recency 淘汰
    - user_id 隔离, 不同用户的经验不交叉
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────

_MAX_LESSONS = 200  # 教训数量上限
_FORGET_BATCH = 20  # 超出上限时一次淘汰的数量
_RELEVANCE_THRESHOLD = 0.15  # Jaccard 相似度阈值, 低于此值不返回
_DEFAULT_TOP_K = 5  # 默认返回的教训数量


# ── 数据结构 ─────────────────────────────────────────────────────────────


@dataclass
class Lesson:
    """一条跨会话教训.

    Attributes:
        lesson_id: 唯一标识 (auto-increment).
        task_pattern: 用户任务的关键词摘要 (用于匹配).
        failure_type: 失败类型分类.
        lesson_text: 教训正文 (LLM 可读的自然语言).
        agent_id: 涉及的 agent.
        user_id: 用户 ID (隔离用).
        severity: 严重程度 "low" | "medium" | "high".
        occurrence_count: 此教训被记录的次数 (相同问题重复犯错时累加).
        created_at: 首次创建时间戳.
        updated_at: 最近更新时间戳.
        last_seen_at: 最近一次匹配到的时间戳.
    """
    lesson_id: int = 0
    task_pattern: str = ""
    failure_type: str = ""
    lesson_text: str = ""
    agent_id: str = ""
    user_id: str = ""
    severity: str = "medium"
    occurrence_count: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    last_seen_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """转为字典 (用于序列化/日志)."""
        return {
            "lesson_id": self.lesson_id,
            "task_pattern": self.task_pattern,
            "failure_type": self.failure_type,
            "lesson_text": self.lesson_text,
            "agent_id": self.agent_id,
            "severity": self.severity,
            "occurrence_count": self.occurrence_count,
        }


# ── 关键词工具 ───────────────────────────────────────────────────────────


def _tokenize(text: str) -> set[str]:
    """中文分词 + 英文 token 提取.

    对中文使用 2-gram 滑动窗口 (无需 jieba 依赖),
    对英文使用 \\w+ 正则.
    """
    tokens: set[str] = set()
    # 英文 token
    tokens.update(re.findall(r"[a-zA-Z_]{2,}", text.lower()))
    # 中文 2-gram
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chinese_chars) - 1):
        tokens.add(chinese_chars[i] + chinese_chars[i + 1])
    # 单个中文关键字 (长度 >= 2 的词组)
    if len(chinese_chars) == 1:
        tokens.add(chinese_chars[0])
    return tokens


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard 相似度 (基于 token 集合)."""
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    intersection = ta & tb
    union = ta | tb
    return len(intersection) / len(union)


# ── Schema ────────────────────────────────────────────────────────────────

_LESSON_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_lessons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_pattern    TEXT NOT NULL DEFAULT '',
    failure_type    TEXT NOT NULL DEFAULT '',
    lesson_text     TEXT NOT NULL,
    agent_id        TEXT NOT NULL DEFAULT '',
    user_id         TEXT NOT NULL DEFAULT '',
    severity        TEXT NOT NULL DEFAULT 'medium',
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    last_seen_at    REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_lessons_user_agent
    ON agent_lessons(user_id, agent_id);

CREATE INDEX IF NOT EXISTS idx_lessons_seen
    ON agent_lessons(last_seen_at DESC);
"""


# ── LessonStore ───────────────────────────────────────────────────────────


class LessonStore:
    """SQLite-backed persistent store for cross-session failure lessons.

    Usage::

        store = LessonStore(db_path="~/.personal-agent/agent.db")
        store.record_lesson(
            task_pattern="查询持仓",
            failure_type="missing_data",
            lesson_text="持仓查询前需要先确认用户ID",
            agent_id="investment-analyst",
            user_id="user_123",
        )
        lessons = store.get_relevant_lessons("我的持仓情况", agent_id="investment-analyst")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path), check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize the lessons table."""
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_LESSON_SCHEMA)
            conn.commit()

    # ── 写操作 ─────────────────────────────────────────────────────────

    def record_lesson(
        self,
        task_pattern: str,
        failure_type: str,
        lesson_text: str,
        agent_id: str = "",
        user_id: str = "",
        severity: str = "medium",
    ) -> int:
        """记录一条教训. 如果相似教训已存在, 累加 occurrence_count.

        Args:
            task_pattern: 用户任务的关键词摘要.
            failure_type: 失败类型 (e.g., "missing_data", "wrong_tool", "hallucination").
            lesson_text: 教训正文.
            agent_id: 涉及的 agent.
            user_id: 用户 ID.
            severity: "low" | "medium" | "high".

        Returns:
            lesson_id (新记录的 ID, 或已存在记录的 ID).
        """
        if not lesson_text or not lesson_text.strip():
            return 0

        task_pattern = task_pattern.strip()[:500]
        lesson_text = lesson_text.strip()[:2000]
        now = time.time()

        with self._lock:
            conn = self._get_conn()

            # 检查是否已有相似教训 (相同 agent + 相似 task_pattern)
            existing = self._find_similar(
                conn, task_pattern, agent_id, user_id,
            )

            if existing:
                # 累加 occurrence_count, 更新 lesson_text (取更长的) 和时间戳
                row = existing
                old_lesson = row["lesson_text"]
                new_lesson = lesson_text if len(lesson_text) > len(old_lesson) else old_lesson
                new_count = row["occurrence_count"] + 1
                # severity 取更高的
                new_severity = self._max_severity(row["severity"], severity)

                conn.execute(
                    """UPDATE agent_lessons
                       SET lesson_text=?, occurrence_count=?, severity=?,
                           updated_at=?, last_seen_at=?
                       WHERE id=?""",
                    (new_lesson, new_count, new_severity, now, now, row["id"]),
                )
                conn.commit()
                logger.debug(
                    "lesson_store: updated lesson id=%d count=%d pattern=%s",
                    row["id"], new_count, task_pattern[:50],
                )
                return row["id"]

            # 新记录
            cursor = conn.execute(
                """INSERT INTO agent_lessons
                   (task_pattern, failure_type, lesson_text, agent_id, user_id,
                    severity, occurrence_count, created_at, updated_at, last_seen_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (task_pattern, failure_type, lesson_text, agent_id, user_id,
                 severity, now, now, now),
            )
            conn.commit()
            lesson_id = cursor.lastrowid or 0

            logger.debug(
                "lesson_store: recorded lesson id=%d agent=%s type=%s pattern=%s",
                lesson_id, agent_id, failure_type, task_pattern[:50],
            )

            # 检查是否需要淘汰
            self._maybe_forget(conn, user_id)

            return lesson_id

    def _find_similar(
        self,
        conn: sqlite3.Connection,
        task_pattern: str,
        agent_id: str,
        user_id: str,
    ) -> sqlite3.Row | None:
        """查找相似教训 (相同 agent + Jaccard >= 0.5)."""
        rows = conn.execute(
            """SELECT * FROM agent_lessons
               WHERE user_id=? AND agent_id=?
               ORDER BY updated_at DESC LIMIT 50""",
            (user_id, agent_id),
        ).fetchall()

        for row in rows:
            sim = _jaccard_similarity(task_pattern, row["task_pattern"])
            if sim >= 0.5:
                return row
        return None

    def _maybe_forget(self, conn: sqlite3.Connection, user_id: str) -> None:
        """淘汰低优先级教训 (超过 _MAX_LESSONS 时)."""
        count = conn.execute(
            "SELECT COUNT(*) FROM agent_lessons WHERE user_id=?",
            (user_id,),
        ).fetchone()[0]

        if count <= _MAX_LESSONS:
            return

        # 按 (occurrence_count ASC, last_seen_at ASC) 排序, 删除最低优先级的
        to_forget = min(count - _MAX_LESSONS, _FORGET_BATCH)
        conn.execute(
            """DELETE FROM agent_lessons
               WHERE id IN (
                   SELECT id FROM agent_lessons
                   WHERE user_id=?
                   ORDER BY occurrence_count ASC, last_seen_at ASC
                   LIMIT ?
               )""",
            (user_id, to_forget),
        )
        conn.commit()
        logger.info(
            "lesson_store: forgot %d low-priority lessons for user=%s",
            to_forget, user_id,
        )

    def update_last_seen(self, lesson_ids: Sequence[int]) -> None:
        """更新教训的 last_seen_at (当教训被注入并使用时调用)."""
        if not lesson_ids:
            return
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            for lid in lesson_ids:
                conn.execute(
                    "UPDATE agent_lessons SET last_seen_at=? WHERE id=?",
                    (now, lid),
                )
            conn.commit()

    # ── 读操作 ─────────────────────────────────────────────────────────

    def get_relevant_lessons(
        self,
        task: str,
        agent_id: str = "",
        user_id: str = "",
        top_k: int = _DEFAULT_TOP_K,
    ) -> list[Lesson]:
        """获取与任务相关的教训.

        使用关键词 Jaccard 相似度匹配. 返回按相关度降序排列的教训列表.

        Args:
            task: 用户任务文本.
            agent_id: 限定 agent (空字符串 = 所有 agent).
            user_id: 用户 ID.
            top_k: 最多返回的教训数量.

        Returns:
            相关教训列表, 按 Jaccard 相似度降序.
        """
        with self._lock:
            conn = self._get_conn()

            if agent_id:
                rows = conn.execute(
                    """SELECT * FROM agent_lessons
                       WHERE user_id=? AND agent_id=?
                       ORDER BY updated_at DESC LIMIT 100""",
                    (user_id, agent_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM agent_lessons
                       WHERE user_id=?
                       ORDER BY updated_at DESC LIMIT 100""",
                    (user_id,),
                ).fetchall()

        if not rows:
            return []

        # 计算相似度并排序
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            sim = _jaccard_similarity(task, row["task_pattern"])
            if sim >= _RELEVANCE_THRESHOLD:
                scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 取 top_k, 并在 severity 为 high 时优先返回
        result: list[Lesson] = []
        for sim, row in scored[:top_k]:
            lesson = self._row_to_lesson(row)
            lesson.task_pattern = f"[relevance={sim:.2f}] {lesson.task_pattern}"
            result.append(lesson)

        return result

    def get_all_lessons(
        self,
        user_id: str = "",
        agent_id: str = "",
        limit: int = 50,
    ) -> list[Lesson]:
        """获取所有教训 (按更新时间降序)."""
        with self._lock:
            conn = self._get_conn()
            if agent_id:
                rows = conn.execute(
                    """SELECT * FROM agent_lessons
                       WHERE user_id=? AND agent_id=?
                       ORDER BY updated_at DESC LIMIT ?""",
                    (user_id, agent_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM agent_lessons
                       WHERE user_id=?
                       ORDER BY updated_at DESC LIMIT ?""",
                    (user_id, limit),
                ).fetchall()

        return [self._row_to_lesson(row) for row in rows]

    def delete_lesson(self, lesson_id: int) -> bool:
        """删除一条教训. Returns True if deleted."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM agent_lessons WHERE id=?", (lesson_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def count(self, user_id: str = "") -> int:
        """返回教训总数."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT COUNT(*) FROM agent_lessons WHERE user_id=?",
                (user_id,),
            ).fetchone()
            return row[0] if row else 0

    def close(self) -> None:
        """关闭数据库连接."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ── 内部工具 ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_lesson(row: sqlite3.Row) -> Lesson:
        """将数据库行转换为 Lesson 对象."""
        return Lesson(
            lesson_id=row["id"],
            task_pattern=row["task_pattern"],
            failure_type=row["failure_type"],
            lesson_text=row["lesson_text"],
            agent_id=row["agent_id"],
            user_id=row["user_id"],
            severity=row["severity"],
            occurrence_count=row["occurrence_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_seen_at=row["last_seen_at"] if "last_seen_at" in row.keys() else 0.0,
        )

    @staticmethod
    def _max_severity(a: str, b: str) -> str:
        """取两个 severity 中更高的."""
        order = {"low": 1, "medium": 2, "high": 3}
        return a if order.get(a, 2) >= order.get(b, 2) else b
