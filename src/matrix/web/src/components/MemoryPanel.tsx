import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../utils/api';

// ── Types ────────────────────────────────────────────────────────────────

interface MemoryItem {
  key: string;
  value: string;
  memory_type: 'preference' | 'policy';
  created_at: number;
  updated_at: number;
}

interface EvolutionReport {
  total_before: number;
  total_after: number;
  conflicts_resolved: number;
  memories_consolidated: number;
  memories_forgotten: number;
  details: string[];
}

interface LessonItem {
  lesson_id: number;
  task_pattern: string;
  failure_type: string;
  lesson_text: string;
  agent_id: string;
  severity: string;
  occurrence_count: number;
}

type Tab = 'memories' | 'lessons' | 'evolution';

// ── Component ─────────────────────────────────────────────────────────────

export const MemoryPanel: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const [tab, setTab] = useState<Tab>('memories');
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [lessons, setLessons] = useState<LessonItem[]>([]);
  const [memCount, setMemCount] = useState(0);
  const [lessonCount, setLessonCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<'all' | 'preference' | 'policy'>('all');
  const [editing, setEditing] = useState<MemoryItem | null>(null);
  const [creating, setCreating] = useState(false);
  const [evolving, setEvolving] = useState(false);
  const [lastReport, setLastReport] = useState<EvolutionReport | null>(null);

  const loadMemories = useCallback(async () => {
    try {
      const data = await api<{ memories: MemoryItem[]; count: number; max: number }>('/memory/list');
      setMemories(data.memories || []);
      setMemCount(data.count || 0);
    } catch (e: any) {
      setError(e.message || '加载记忆失败');
    }
  }, []);

  const loadLessons = useCallback(async () => {
    try {
      const data = await api<{ lessons: LessonItem[]; count: number; max: number }>('/memory/lessons');
      setLessons(data.lessons || []);
      setLessonCount(data.count || 0);
    } catch (e: any) {
      setError(e.message || '加载教训失败');
    }
  }, []);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError('');
    await Promise.all([loadMemories(), loadLessons()]);
    setLoading(false);
  }, [loadMemories, loadLessons]);

  useEffect(() => { loadAll(); }, [loadAll]);

  // ── Handlers ────────────────────────────────────────────────────────────

  const handleSave = async (key: string, value: string, memory_type: string) => {
    try {
      await api('/memory', {
        method: 'POST',
        body: JSON.stringify({ key, value, memory_type }),
      });
      setEditing(null);
      setCreating(false);
      await loadMemories();
    } catch (e: any) {
      setError(e.message || '保存失败');
    }
  };

  const handleDelete = async (key: string, isPolicy: boolean) => {
    const msg = isPolicy
      ? `此记忆为 policy 类型（硬约束），删除可能导致 agent 行为变化。\n\n确定删除 "${key}"？`
      : `确定删除记忆 "${key}"？`;
    if (!window.confirm(msg)) return;

    const url = isPolicy
      ? `/memory/${encodeURIComponent(key)}?confirm=true`
      : `/memory/${encodeURIComponent(key)}`;
    try {
      await api(url, { method: 'DELETE' });
      await loadMemories();
    } catch (e: any) {
      setError(e.message || '删除失败');
    }
  };

  const handleDeleteLesson = async (id: number) => {
    if (!window.confirm('确定删除此教训？')) return;
    try {
      await api(`/memory/lessons/${id}`, { method: 'DELETE' });
      await loadLessons();
    } catch (e: any) {
      setError(e.message || '删除教训失败');
    }
  };

  const handleEvolve = async () => {
    if (!window.confirm('手动触发记忆演化？\n这将执行冲突检测、合并去重和主动遗忘。')) return;
    setEvolving(true);
    setError('');
    try {
      const data = await api<{ ok: boolean; report: EvolutionReport }>('/memory/evolve', {
        method: 'POST',
        body: JSON.stringify({}),
      });
      setLastReport(data.report);
      await loadMemories();
    } catch (e: any) {
      setError(e.message || '演化失败');
    }
    setEvolving(false);
  };

  // ── Render ──────────────────────────────────────────────────────────────

  const filteredMemories = filter === 'all'
    ? memories
    : memories.filter(m => m.memory_type === filter);

  const showForm = creating || editing !== null;

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={headerStyle}>
          <span>记忆管理</span>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
              {memCount} / 80 记忆 · {lessonCount} / 200 教训
            </span>
            <button onClick={onClose} style={{ ...btnStyle, background: 'transparent' }}>✕</button>
          </div>
        </div>

        {/* Tabs */}
        <div style={tabsStyle}>
          <button
            style={tab === 'memories' ? { ...tabStyle, ...tabActiveStyle } : tabStyle}
            onClick={() => setTab('memories')}
          >用户记忆</button>
          <button
            style={tab === 'lessons' ? { ...tabStyle, ...tabActiveStyle } : tabStyle}
            onClick={() => setTab('lessons')}
          >经验教训</button>
          <button
            style={tab === 'evolution' ? { ...tabStyle, ...tabActiveStyle } : tabStyle}
            onClick={() => setTab('evolution')}
          >演化日志</button>
        </div>

        {/* Body */}
        <div style={bodyStyle}>
          {error && (
            <div style={{ color: 'var(--error)', marginBottom: 12, fontSize: 13 }}>{error}</div>
          )}

          {tab === 'memories' && (
            <>
              {/* Toolbar */}
              <div style={toolbarStyle}>
                <div style={{ display: 'flex', gap: 6 }}>
                  {(['all', 'preference', 'policy'] as const).map(f => (
                    <button
                      key={f}
                      onClick={() => setFilter(f)}
                      style={filter === f ? { ...chipStyle, ...chipActiveStyle } : chipStyle}
                    >
                      {f === 'all' ? '全部' : f === 'preference' ? '偏好' : '规则'}
                    </button>
                  ))}
                </div>
                <div style={{ display: 'flex', gap: 6 }}>
                  {!showForm && (
                    <button onClick={() => setCreating(true)} style={btnStyle}>+ 新增</button>
                  )}
                  <button
                    onClick={handleEvolve}
                    disabled={evolving}
                    style={{ ...btnStyle, opacity: evolving ? 0.6 : 1 }}
                  >
                    {evolving ? '演化中...' : '触发演化'}
                  </button>
                </div>
              </div>

              {/* Form or Table */}
              {showForm ? (
                <MemoryForm
                  initial={editing}
                  onSave={handleSave}
                  onCancel={() => { setEditing(null); setCreating(false); }}
                />
              ) : loading ? (
                <div style={emptyStyle}>加载中...</div>
              ) : filteredMemories.length === 0 ? (
                <div style={emptyStyle}>暂无记忆数据</div>
              ) : (
                <div style={{ overflow: 'auto' }}>
                  <table style={tableStyle}>
                    <thead>
                      <tr>
                        <th style={thStyle}>Key</th>
                        <th style={thStyle}>Value</th>
                        <th style={{ ...thStyle, width: 80 }}>类型</th>
                        <th style={{ ...thStyle, width: 100 }}>操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredMemories.map(m => (
                        <tr key={m.key} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td style={tdKeyStyle}>{m.key}</td>
                          <td style={tdValueStyle}>{m.value}</td>
                          <td style={tdStyle}>
                            <span style={m.memory_type === 'policy' ? badgePolicyStyle : badgePrefStyle}>
                              {m.memory_type === 'policy' ? '规则' : '偏好'}
                            </span>
                          </td>
                          <td style={tdStyle}>
                            <button
                              onClick={() => setEditing(m)}
                              style={{ ...miniBtnStyle, marginRight: 4 }}
                            >编辑</button>
                            <button
                              onClick={() => handleDelete(m.key, m.memory_type === 'policy')}
                              style={{ ...miniBtnStyle, color: 'var(--error)' }}
                            >删除</button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {tab === 'lessons' && (
            <>
              {loading ? (
                <div style={emptyStyle}>加载中...</div>
              ) : lessons.length === 0 ? (
                <div style={emptyStyle}>暂无经验教训</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, overflow: 'auto' }}>
                  {lessons.map(l => (
                    <div key={l.lesson_id} style={lessonCardStyle}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginBottom: 4 }}>
                            <span style={{ fontSize: 13, fontWeight: 600 }}>{l.task_pattern}</span>
                            <span style={sevBadgeStyle(l.severity)}>{l.severity}</span>
                            {l.occurrence_count > 1 && (
                              <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                                ×{l.occurrence_count}
                              </span>
                            )}
                          </div>
                          <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                            {l.lesson_text}
                          </div>
                          <div style={{ fontSize: 11, color: 'var(--text-secondary)', opacity: 0.6, marginTop: 4 }}>
                            {l.failure_type} · {l.agent_id || 'general'}
                          </div>
                        </div>
                        <button
                          onClick={() => handleDeleteLesson(l.lesson_id)}
                          style={{ ...miniBtnStyle, color: 'var(--error)', flexShrink: 0 }}
                        >删除</button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {tab === 'evolution' && (
            <>
              <div style={{ marginBottom: 16 }}>
                <button
                  onClick={handleEvolve}
                  disabled={evolving}
                  style={{ ...btnStyle, opacity: evolving ? 0.6 : 1 }}
                >
                  {evolving ? '演化进行中...' : '手动触发演化'}
                </button>
              </div>

              {lastReport ? (
                <div style={reportCardStyle}>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>
                    最近一次演化报告
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                    <ReportStat label="演化前" value={lastReport.total_before} color="var(--text-secondary)" />
                    <ReportStat label="演化后" value={lastReport.total_after} color="var(--text)" />
                    <ReportStat label="冲突解决" value={lastReport.conflicts_resolved} color="#F59E0B" />
                    <ReportStat label="合并去重" value={lastReport.memories_consolidated} color="#10B981" />
                  </div>
                  {lastReport.memories_forgotten > 0 && (
                    <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-secondary)' }}>
                      主动遗忘: {lastReport.memories_forgotten} 条
                    </div>
                  )}
                  {lastReport.details.length > 0 && (
                    <div style={{ marginTop: 12, fontSize: 12, color: 'var(--text-secondary)' }}>
                      {lastReport.details.map((d, i) => (
                        <div key={i} style={{ padding: '2px 0' }}>· {d}</div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div style={emptyStyle}>
                  尚未触发过演化。

                  每次对话结束后系统会自动执行演化，
                  也可点击上方按钮手动触发。
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Sub-components ────────────────────────────────────────────────────────

const ReportStat: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
  <div style={{ textAlign: 'center', padding: '12px 8px', background: 'var(--bg2)', borderRadius: 8 }}>
    <div style={{ fontSize: 20, fontWeight: 700, color, fontFamily: 'var(--font-mono, monospace)' }}>{value}</div>
    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>{label}</div>
  </div>
);

const MemoryForm: React.FC<{
  initial: MemoryItem | null;
  onSave: (key: string, value: string, memory_type: string) => void;
  onCancel: () => void;
}> = ({ initial, onSave, onCancel }) => {
  const [key, setKey] = useState(initial?.key || '');
  const [value, setValue] = useState(initial?.value || '');
  const [memory_type, setMemoryType] = useState(initial?.memory_type || 'preference');

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '8px 10px', borderRadius: 6,
    border: '1px solid var(--border)', background: 'var(--bg)',
    color: 'var(--text)', fontSize: 13, boxSizing: 'border-box',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div>
        <label style={labelStyle}>Key</label>
        <input
          style={inputStyle}
          value={key}
          onChange={e => setKey(e.target.value)}
          placeholder="例如: communication_lang"
          disabled={!!initial}
        />
      </div>
      <div>
        <label style={labelStyle}>Value</label>
        <textarea
          style={{ ...inputStyle, minHeight: 80, resize: 'vertical' }}
          value={value}
          onChange={e => setValue(e.target.value)}
          placeholder="记忆内容"
        />
      </div>
      <div>
        <label style={labelStyle}>类型</label>
        <div style={{ display: 'flex', gap: 8 }}>
          {(['preference', 'policy'] as const).map(t => (
            <button
              key={t}
              onClick={() => setMemoryType(t)}
              style={{
                ...btnStyle,
                background: memory_type === t ? 'var(--accent)' : 'var(--bg2)',
                color: memory_type === t ? '#fff' : 'var(--text)',
                fontSize: 12, padding: '6px 14px',
              }}
            >
              {t === 'preference' ? '偏好' : '规则'}
            </button>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
        <button onClick={onCancel} style={{ ...btnStyle, background: 'var(--bg2)' }}>取消</button>
        <button
          onClick={() => key.trim() && value.trim() && onSave(key.trim(), value.trim(), memory_type)}
          style={btnStyle}
          disabled={!key.trim() || !value.trim()}
        >
          {initial ? '保存' : '创建'}
        </button>
      </div>
    </div>
  );
};

// ── Styles ────────────────────────────────────────────────────────────────

const overlayStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.25)', zIndex: 1000,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
};

const modalStyle: React.CSSProperties = {
  background: 'var(--surface)', borderRadius: 12, width: 720, maxHeight: '85vh',
  overflow: 'hidden', display: 'flex', flexDirection: 'column',
  boxShadow: '0 10px 40px rgba(0,0,0,0.15)',
};

const headerStyle: React.CSSProperties = {
  padding: '16px 20px', borderBottom: '1px solid var(--border)',
  fontWeight: 600, fontSize: 16, display: 'flex', justifyContent: 'space-between',
  alignItems: 'center',
};

const tabsStyle: React.CSSProperties = {
  display: 'flex', gap: 0, padding: '0 20px',
  borderBottom: '1px solid var(--border)', background: 'var(--surface)',
};

const tabStyle: React.CSSProperties = {
  padding: '10px 16px', fontSize: 13, color: 'var(--text-secondary)',
  border: 'none', background: 'none', cursor: 'pointer',
  borderBottom: '2px solid transparent', transition: 'all 0.15s',
};

const tabActiveStyle: React.CSSProperties = {
  color: 'var(--accent)', borderBottomColor: 'var(--accent)', fontWeight: 500,
};

const bodyStyle: React.CSSProperties = {
  padding: 16, overflowY: 'auto', flex: 1,
};

const toolbarStyle: React.CSSProperties = {
  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  marginBottom: 12,
};

const tableStyle: React.CSSProperties = {
  width: '100%', borderCollapse: 'collapse',
};

const thStyle: React.CSSProperties = {
  textAlign: 'left', padding: '8px 12px', fontSize: 11, fontWeight: 500,
  color: 'var(--text-secondary)', textTransform: 'uppercase',
  letterSpacing: '0.5px', background: 'var(--bg2)',
  borderBottom: '1px solid var(--border)',
};

const tdStyle: React.CSSProperties = {
  padding: '10px 12px', fontSize: 13, borderBottom: '1px solid var(--border)',
  verticalAlign: 'top' as const,
};

const tdKeyStyle: React.CSSProperties = {
  ...tdStyle,
  fontFamily: 'var(--font-mono, monospace)', fontSize: 12, fontWeight: 500,
};

const tdValueStyle: React.CSSProperties = {
  ...tdStyle,
  color: 'var(--text-secondary)', maxWidth: 260, lineHeight: 1.5,
};

const badgePrefStyle: React.CSSProperties = {
  display: 'inline-block', padding: '2px 8px', fontSize: 11,
  borderRadius: 999, fontWeight: 500,
  background: 'rgba(99,102,241,0.1)', color: 'var(--accent)',
};

const badgePolicyStyle: React.CSSProperties = {
  display: 'inline-block', padding: '2px 8px', fontSize: 11,
  borderRadius: 999, fontWeight: 500,
  background: 'rgba(245,158,11,0.12)', color: '#F59E0B',
};

const chipStyle: React.CSSProperties = {
  padding: '4px 12px', fontSize: 12, borderRadius: 999,
  border: '1px solid var(--border)', background: 'var(--surface)',
  color: 'var(--text-secondary)', cursor: 'pointer',
};

const chipActiveStyle: React.CSSProperties = {
  background: 'var(--accent)', color: '#fff', borderColor: 'var(--accent)',
};

const btnStyle: React.CSSProperties = {
  padding: '6px 14px', borderRadius: 6, border: 'none',
  background: 'var(--accent)', color: '#fff', cursor: 'pointer',
  fontSize: 13, fontWeight: 500,
};

const miniBtnStyle: React.CSSProperties = {
  padding: '2px 8px', fontSize: 11, borderRadius: 4,
  border: '1px solid var(--border)', background: 'var(--surface)',
  color: 'var(--text-secondary)', cursor: 'pointer',
};

const emptyStyle: React.CSSProperties = {
  textAlign: 'center', padding: 40, color: 'var(--text-secondary)',
  fontSize: 13, lineHeight: 1.8,
};

const lessonCardStyle: React.CSSProperties = {
  padding: '10px 12px', background: 'var(--bg2)', borderRadius: 8,
  border: '1px solid var(--border)',
};

const reportCardStyle: React.CSSProperties = {
  padding: 16, background: 'var(--surface)', border: '1px solid var(--border)',
  borderRadius: 12,
};

const labelStyle: React.CSSProperties = {
  display: 'block', fontSize: 12, color: 'var(--text-secondary)',
  marginBottom: 4, fontWeight: 500,
};

function sevBadgeStyle(sev: string): React.CSSProperties {
  const colors: Record<string, string> = {
    high: 'rgba(239,68,68,0.1);color:#EF4444',
    medium: 'rgba(245,158,11,0.12);color:#F59E0B',
    low: 'rgba(99,102,241,0.1);color:var(--accent)',
  };
  const fallback = colors.medium;
  const css = colors[sev] || fallback;
  const [bg, color] = css.split(';');
  return {
    display: 'inline-block', padding: '1px 6px', fontSize: 10,
    borderRadius: 999, fontWeight: 500,
    background: bg.split(':')[1], color: color.split(':')[1],
  };
}
