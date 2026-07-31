import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { SessionItem } from '../types';
import QuickQuestions from './QuickQuestions';

interface Props {
  sessions: SessionItem[];
  currentId: string;
  showArchive: boolean;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onRefresh: () => void;
  onDelete: (id: string) => void;
  onBatchArchive: (ids: string[]) => void;
  onBatchUnarchive: (ids: string[]) => void;
  onBatchDelete: (ids: string[]) => void;
  onToggleArchive: () => void;
  onQuickSend?: (question: string) => void;
  username?: string;
  onLogout?: () => void;
  onOpenSkillEditor?: () => void;
  onOpenMcp?: () => void;
}

const SessionList: React.FC<Props> = ({
  sessions, currentId, showArchive,
    onSelect, onCreate, onRefresh, onDelete,
    onBatchArchive, onBatchUnarchive, onBatchDelete,
    onToggleArchive, onQuickSend,
  username, onLogout, onOpenSkillEditor, onOpenMcp,
}) => {
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const [batchMode, setBatchMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showUserMenu, setShowUserMenu] = useState(false);
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const longPressTarget = useRef<string | null>(null);

  const closeContextMenu = useCallback(() => setContextMenu(null), []);

  useEffect(() => {
    const handler = () => closeContextMenu();
    window.addEventListener('click', handler);
    return () => window.removeEventListener('click', handler);
  }, [closeContextMenu]);

  const handleContextMenu = (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    setContextMenu({ x: e.clientX, y: e.clientY, id });
  };

  const handleTouchStart = (id: string) => {
    longPressTarget.current = id;
    longPressTimer.current = setTimeout(() => {
      if (longPressTarget.current === id) {
        setBatchMode(true);
        setSelectedIds(new Set([id]));
      }
    }, 600);
  };

  const handleTouchEnd = () => {
    if (longPressTimer.current) { clearTimeout(longPressTimer.current); longPressTimer.current = null; }
    longPressTarget.current = null;
  };

  const handleDelete = (id: string) => {
    onDelete(id);
    closeContextMenu();
  };

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      if (next.size === 0) setBatchMode(false);
      return next;
    });
  };

  const exitBatchMode = () => {
    setBatchMode(false);
    setSelectedIds(new Set());
  };

  const formatDate = (dateStr: string) => {
    try {
      // Handle both ISO strings and Unix timestamps (seconds)
      const d = typeof dateStr === 'string' && /^\d+$/.test(dateStr)
        ? new Date(Number(dateStr) * 1000)
        : new Date(dateStr);
      if (isNaN(d.getTime())) return '';
      const now = new Date();
      const diff = now.getTime() - d.getTime();
      if (diff < 86400000) return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
    } catch { return ''; }
  };

  return (
    <div style={styles.container}>
      {/* sb-top: 新建会话 — matches original HTML */}
      <div style={{
        padding: '12px 12px 8px',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        flex: '0 0 auto',
      }}>
        <button
          onClick={onCreate}
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            width: '100%', background: 'rgba(99,102,241,0.08)',
            border: 'none', borderRadius: 8, padding: '9px 12px',
            cursor: 'pointer', fontSize: 13, color: 'var(--accent)',
            fontWeight: 500, fontFamily: 'var(--font)',
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          新建会话
        </button>
      </div>

      {/* Quick Questions section — matches original HTML "常用问题" */}
      <QuickQuestions onSend={onQuickSend || (() => {})} />

      {/* sb-section sb-sessions: 会话 header + list — matches original HTML */}
      <div style={styles.sessionsSection}>
        <div style={styles.sessionsHeader}>
          <span style={styles.headerTitle}>会话</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span
              style={styles.archiveToggle}
              onClick={onToggleArchive}
              title="显示已归档"
            >
              {showArchive ? '全部' : '已归档'}
            </span>
            <svg
              width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round"
              style={{ color: 'var(--text-secondary)', opacity: 0.4, cursor: 'pointer', flexShrink: 0 }}
              aria-label="刷新会话列表"
              onClick={onRefresh}
            >
              <polyline points="23 4 23 10 17 10" />
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
            </svg>
          </div>
        </div>

      {batchMode && selectedIds.size > 0 && (
        <div style={styles.batchBar}>
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>已选 {selectedIds.size}</span>
          <div style={{ display: 'flex', gap: 4 }}>
            <button style={styles.batchBtn} onClick={() => { onBatchArchive([...selectedIds]); exitBatchMode(); }}>归档</button>
            <button style={styles.batchBtn} onClick={() => { onBatchUnarchive([...selectedIds]); exitBatchMode(); }}>取消归档</button>
            <button style={{ ...styles.batchBtn, color: 'var(--error)' }} onClick={() => { onBatchDelete([...selectedIds]); exitBatchMode(); }}>删除</button>
            <button style={styles.batchBtn} onClick={exitBatchMode}>取消</button>
          </div>
        </div>
      )}

      <div style={styles.list}>
        {sessions.map((s) => (
          <div
            key={s.id}
            style={{
              ...styles.item,
              ...(s.id === currentId ? styles.itemActive : {}),
              display: 'flex', alignItems: 'center', gap: 8,
            }}
            onClick={() => {
              if (batchMode) { toggleSelect(s.id); }
              else { onSelect(s.id); }
            }}
            onContextMenu={(e) => handleContextMenu(e, s.id)}
            onTouchStart={() => handleTouchStart(s.id)}
            onTouchEnd={handleTouchEnd}
            onTouchMove={handleTouchEnd}
          >
            {batchMode && (
              <input
                type="checkbox"
                checked={selectedIds.has(s.id)}
                onChange={() => toggleSelect(s.id)}
                style={{ cursor: 'pointer', accentColor: 'var(--accent)' }}
              />
            )}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ ...styles.itemTitle, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.title || '新会话'}</span>
                {s.branch_count !== undefined && s.branch_count > 0 && (
                  <span style={styles.branchBadge}>{s.branch_count} 叉</span>
                )}
              </div>
              <div style={styles.itemMeta}>
                <span>{s.turns} 轮</span>
                <span>{formatDate(s.updated_at)}</span>
              </div>
            </div>
          </div>
        ))}
        {sessions.length === 0 && (
          <div style={styles.empty}>暂无会话</div>
        )}
      </div>

      {contextMenu && (
        <div style={{ ...styles.contextMenu, left: contextMenu.x, top: contextMenu.y }}>
          <button style={{ ...styles.contextMenuItem, color: '#ff5050' }} onClick={() => handleDelete(contextMenu.id)}>
            删除会话
          </button>
        </div>
      )}

      </div>{/* end sb-sessions */}

      {/* User bar — matches original HTML .sb-user-bar */}
      {username && (
        <div style={styles.userBar}>
          <div
            style={styles.userInfo}
            onClick={() => setShowUserMenu(!showUserMenu)}
          >
            <div style={styles.userAvatar}>{username.charAt(0).toUpperCase()}</div>
            <span style={styles.userName}>{username}</span>
            <svg
              width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round"
              style={{ color: 'var(--text-secondary)', transition: 'transform 0.15s', transform: showUserMenu ? 'rotate(180deg)' : 'none' }}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>
          {showUserMenu && (
            <div style={styles.userMenu}>
              <button
                style={styles.userMenuItem}
                onClick={() => { onOpenSkillEditor?.(); setShowUserMenu(false); }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                  <path d="M2 17l10 5 10-5" />
                </svg>
                <span>技能管理</span>
              </button>
              <button
                style={styles.userMenuItem}
                onClick={() => { onOpenMcp?.(); setShowUserMenu(false); }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                  <rect x="2" y="3" width="20" height="14" rx="2" />
                  <line x1="8" y1="21" x2="16" y2="21" />
                  <line x1="12" y1="17" x2="12" y2="21" />
                </svg>
                <span>MCP 服务</span>
              </button>
              <div style={styles.userMenuDivider} />
              <button
                style={{ ...styles.userMenuItem, color: 'var(--error)' }}
                onClick={() => { onLogout?.(); setShowUserMenu(false); }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                  <polyline points="16 17 21 12 16 7" />
                  <line x1="21" y1="12" x2="9" y2="12" />
                </svg>
                <span>退出登录</span>
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    flex: 1, display: 'flex', flexDirection: 'column',
    backgroundColor: 'var(--surface)', borderRight: '1px solid var(--border)',
    userSelect: 'none', overflow: 'hidden',
  },
  // Matches original .sb-section.sb-sessions
  sessionsSection: {
    display: 'flex',
    flexDirection: 'column',
    padding: '8px 0',
    borderTop: '1px solid var(--border)',
    overflowY: 'auto',
    overflowX: 'hidden',
    flex: '1 1 0%',
    minHeight: 0,
  },
  // Matches original .sb-section-hd
  sessionsHeader: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '4px 16px 6px',
  },
  headerTitle: { fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.8px' },
  archiveToggle: { fontSize: 11, cursor: 'pointer', opacity: 0.5, color: 'var(--text-secondary)' },
  batchBar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '8px 12px', borderBottom: '1px solid var(--border)',
    backgroundColor: 'var(--bg2)', flexWrap: 'wrap', gap: 4,
  },
  batchBtn: {
    padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)',
    backgroundColor: 'transparent', color: 'var(--text)', fontSize: 11, cursor: 'pointer',
  },
  list: { flex: '1 1 0%', minHeight: 0, padding: '0 8px', display: 'flex', flexDirection: 'column', gap: 1 },
  item: {
    padding: '6px 8px', borderRadius: 6, cursor: 'pointer',
    transition: 'background 0.1s, color 0.1s', userSelect: 'none',
    color: 'var(--text-secondary)', fontSize: 13,
  },
  itemActive: { backgroundColor: 'rgba(99,102,241,0.08)', color: 'var(--text)' },
  itemTitle: { fontSize: 13, fontWeight: 400, color: 'inherit', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' },
  itemMeta: { display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)', opacity: 0.6 },
  branchBadge: {
    fontSize: 10, fontWeight: 600, padding: '1px 5px', borderRadius: 4,
    backgroundColor: 'rgba(99, 102, 241, 0.12)', color: 'var(--accent)',
    flexShrink: 0,
  },
  empty: { padding: 16, textAlign: 'center', color: 'var(--text-secondary)', fontSize: 12, opacity: 0.6 },
  contextMenu: {
    position: 'fixed', zIndex: 10000, backgroundColor: 'var(--surface)',
    border: '1px solid var(--border)', borderRadius: 8, padding: '4px 0',
    boxShadow: '0 4px 16px rgba(0,0,0,0.12)', minWidth: 140,
  },
  contextMenuItem: {
    display: 'block', width: '100%', padding: '10px 16px', border: 'none',
    backgroundColor: 'transparent', color: 'var(--text)', fontSize: 14,
    cursor: 'pointer', textAlign: 'left' as const, transition: 'background 0.12s',
  },
  // User bar styles — matches original HTML .sb-user-bar
  userBar: {
    flex: '0 0 auto', borderTop: '1px solid var(--border)',
    padding: '8px 10px', position: 'relative' as const,
  },
  userInfo: {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '6px 8px', borderRadius: 8, cursor: 'pointer',
    transition: 'background 0.12s',
  },
  userAvatar: {
    width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
    background: 'var(--accent)', color: '#fff', fontSize: 12, fontWeight: 600,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    position: 'relative' as const,
  },
  userName: {
    flex: 1, fontSize: 13, color: 'var(--text)', fontWeight: 500,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  userMenu: {
    position: 'absolute', bottom: '100%', left: 10, right: 10,
    backgroundColor: 'var(--surface)', border: '1px solid var(--border)',
    borderRadius: 8, padding: '4px 0', boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
    zIndex: 1000,
  },
  userMenuItem: {
    display: 'flex', alignItems: 'center', gap: 10,
    width: '100%', padding: '10px 14px', border: 'none',
    backgroundColor: 'transparent', color: 'var(--text)', fontSize: 13,
    cursor: 'pointer', textAlign: 'left' as const,
    transition: 'background 0.12s',
    fontFamily: 'var(--font)',
  },
  userMenuDivider: {
    height: 1, backgroundColor: 'var(--border)', margin: '4px 0',
  },
};

export default SessionList;
