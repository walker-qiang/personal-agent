import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { SessionItem } from '../types';

interface Props {
  sessions: SessionItem[];
  currentId: string;
  showArchive: boolean;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
  onBatchArchive: (ids: string[]) => void;
  onBatchUnarchive: (ids: string[]) => void;
  onBatchDelete: (ids: string[]) => void;
  onToggleArchive: () => void;
  onBranch: (id: string, messageId?: string) => void;
}

const SessionList: React.FC<Props> = ({
  sessions, currentId, showArchive,
  onSelect, onCreate, onDelete,
  onBatchArchive, onBatchUnarchive, onBatchDelete,
  onToggleArchive, onBranch,
}) => {
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const [batchMode, setBatchMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
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

  const handleBranch = (id: string) => {
    onBranch(id);
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
      const d = new Date(dateStr);
      const now = new Date();
      const diff = now.getTime() - d.getTime();
      if (diff < 86400000) return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
    } catch { return ''; }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.headerTitle}>会话</span>
        <div style={{ display: 'flex', gap: 4 }}>
          <button
            style={{ ...styles.smallBtn, background: showArchive ? 'var(--accent)' : 'transparent' }}
            onClick={onToggleArchive}
            title="切换归档"
          >
            {showArchive ? '收' : '档'}
          </button>
          <button style={styles.smallBtn} onClick={onCreate} title="新建会话">+</button>
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
          <div style={styles.empty}>暂无会话，点击 + 新建</div>
        )}
      </div>

      {contextMenu && (
        <div style={{ ...styles.contextMenu, left: contextMenu.x, top: contextMenu.y }}>
          <button style={styles.contextMenuItem} onClick={() => handleBranch(contextMenu.id)}>
            从此处分叉
          </button>
          <button style={{ ...styles.contextMenuItem, color: '#ff5050' }} onClick={() => handleDelete(contextMenu.id)}>
            删除会话
          </button>
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    height: '100%', display: 'flex', flexDirection: 'column',
    backgroundColor: 'var(--bg, #1a1a2e)', borderRight: '1px solid var(--rule, #333)',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '16px 16px 12px', borderBottom: '1px solid var(--rule, #333)',
  },
  headerTitle: { fontSize: 16, fontWeight: 600, color: 'var(--text, #e0e0e0)' },
  smallBtn: {
    width: 28, height: 28, borderRadius: 6, border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)', color: 'var(--accent, #5b8def)',
    fontSize: 14, fontWeight: 600, cursor: 'pointer', display: 'flex',
    alignItems: 'center', justifyContent: 'center', lineHeight: 1,
  },
  batchBar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '8px 12px', borderBottom: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2)', flexWrap: 'wrap', gap: 4,
  },
  batchBtn: {
    padding: '3px 8px', borderRadius: 4, border: '1px solid var(--rule, #333)',
    backgroundColor: 'transparent', color: 'var(--text)', fontSize: 11, cursor: 'pointer',
  },
  list: { flex: 1, overflowY: 'auto', padding: '8px' },
  item: {
    padding: '12px 14px', borderRadius: 10, cursor: 'pointer', marginBottom: 4,
    transition: 'background-color 0.15s', userSelect: 'none',
  },
  itemActive: { backgroundColor: 'var(--bg2, #222)', border: '1px solid var(--accent, #5b8def)' },
  itemTitle: { fontSize: 14, fontWeight: 500, color: 'var(--text, #e0e0e0)', marginBottom: 4 },
  itemMeta: { display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--muted, #888)' },
  branchBadge: {
    fontSize: 10, fontWeight: 600, padding: '1px 5px', borderRadius: 4,
    backgroundColor: 'rgba(99, 102, 241, 0.2)', color: 'var(--accent)',
    flexShrink: 0,
  },
  empty: { padding: 24, textAlign: 'center', color: 'var(--muted, #888)', fontSize: 13 },
  contextMenu: {
    position: 'fixed', zIndex: 10000, backgroundColor: 'var(--bg2, #222)',
    border: '1px solid var(--rule, #333)', borderRadius: 8, padding: '4px 0',
    boxShadow: '0 4px 16px rgba(0,0,0,0.4)', minWidth: 140,
  },
  contextMenuItem: {
    display: 'block', width: '100%', padding: '10px 16px', border: 'none',
    backgroundColor: 'transparent', color: 'var(--text)', fontSize: 14,
    cursor: 'pointer', textAlign: 'left' as const, transition: 'background-color 0.15s',
  },
};

export default SessionList;