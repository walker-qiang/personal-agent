import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { SessionItem } from '../types';

interface Props {
  sessions: SessionItem[];
  currentId: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
}

const SessionList: React.FC<Props> = ({ sessions, currentId, onSelect, onCreate, onDelete }) => {
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const longPressTarget = useRef<string | null>(null);

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

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
        handleDelete(id);
      }
    }, 600);
  };

  const handleTouchEnd = () => {
    if (longPressTimer.current) {
      clearTimeout(longPressTimer.current);
      longPressTimer.current = null;
    }
    longPressTarget.current = null;
  };

  const handleDelete = (id: string) => {
    onDelete(id);
    closeContextMenu();
  };

  const formatDate = (dateStr: string) => {
    try {
      const d = new Date(dateStr);
      const now = new Date();
      const diff = now.getTime() - d.getTime();
      if (diff < 86400000) {
        return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
      }
      return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
    } catch {
      return '';
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.headerTitle}>会话</span>
        <button style={styles.newBtn} onClick={onCreate} title="新建会话">
          +
        </button>
      </div>

      <div style={styles.list}>
        {sessions.map((s) => (
          <div
            key={s.id}
            style={{
              ...styles.item,
              ...(s.id === currentId ? styles.itemActive : {}),
            }}
            onClick={() => onSelect(s.id)}
            onContextMenu={(e) => handleContextMenu(e, s.id)}
            onTouchStart={() => handleTouchStart(s.id)}
            onTouchEnd={handleTouchEnd}
            onTouchMove={handleTouchEnd}
          >
            <div style={styles.itemTitle}>{s.title || '新会话'}</div>
            <div style={styles.itemMeta}>
              <span>{s.turns} 轮</span>
              <span>{formatDate(s.updated_at)}</span>
            </div>
          </div>
        ))}
        {sessions.length === 0 && (
          <div style={styles.empty}>暂无会话，点击 + 新建</div>
        )}
      </div>

      {contextMenu && (
        <div
          style={{
            ...styles.contextMenu,
            left: contextMenu.x,
            top: contextMenu.y,
          }}
        >
          <button
            style={styles.contextMenuItem}
            onClick={() => handleDelete(contextMenu.id)}
          >
            删除会话
          </button>
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: 260,
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: 'var(--bg, #1a1a2e)',
    borderRight: '1px solid var(--rule, #333)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 16px 12px',
    borderBottom: '1px solid var(--rule, #333)',
  },
  headerTitle: {
    fontSize: 16,
    fontWeight: 600,
    color: 'var(--text, #e0e0e0)',
  },
  newBtn: {
    width: 32,
    height: 32,
    borderRadius: 8,
    border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)',
    color: 'var(--accent, #5b8def)',
    fontSize: 20,
    fontWeight: 600,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    lineHeight: 1,
    transition: 'background-color 0.2s',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    padding: '8px',
  },
  item: {
    padding: '12px 14px',
    borderRadius: 10,
    cursor: 'pointer',
    marginBottom: 4,
    transition: 'background-color 0.15s',
    userSelect: 'none',
  },
  itemActive: {
    backgroundColor: 'var(--bg2, #222)',
    border: '1px solid var(--accent, #5b8def)',
  },
  itemTitle: {
    fontSize: 14,
    fontWeight: 500,
    color: 'var(--text, #e0e0e0)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    marginBottom: 4,
  },
  itemMeta: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: 12,
    color: 'var(--muted, #888)',
  },
  empty: {
    padding: 24,
    textAlign: 'center',
    color: 'var(--muted, #888)',
    fontSize: 13,
  },
  contextMenu: {
    position: 'fixed',
    zIndex: 10000,
    backgroundColor: 'var(--bg2, #222)',
    border: '1px solid var(--rule, #333)',
    borderRadius: 8,
    padding: '4px 0',
    boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
    minWidth: 140,
  },
  contextMenuItem: {
    display: 'block',
    width: '100%',
    padding: '10px 16px',
    border: 'none',
    backgroundColor: 'transparent',
    color: '#ff5050',
    fontSize: 14,
    cursor: 'pointer',
    textAlign: 'left' as const,
    transition: 'background-color 0.15s',
  },
};

export default SessionList;