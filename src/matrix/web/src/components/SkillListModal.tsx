import React from 'react';
import type { SkillItem } from '../types';

interface Props {
  skills: SkillItem[];
  onEdit: (skill: SkillItem) => void;
  onCreate: () => void;
  onClose: () => void;
}

const SkillListModal: React.FC<Props> = ({ skills, onEdit, onCreate, onClose }) => {
  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div style={styles.overlay} onClick={handleOverlayClick}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>技能管理</h2>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={() => { onClose(); onCreate(); }}
              style={styles.newBtn}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" style={{ width: 14, height: 14 }}>
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              新建
            </button>
            <button onClick={onClose} style={styles.closeBtn}>✕</button>
          </div>
        </div>

        {/* Skill list */}
        <div style={styles.body}>
          {skills.length === 0 ? (
            <div style={styles.empty}>暂无技能，点击右上角"新建"创建</div>
          ) : (
            skills.map(s => (
              <div
                key={s.name}
                style={styles.item}
                onClick={() => { onClose(); onEdit(s); }}
                onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg2)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" style={styles.skillIcon}>
                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                  <path d="M2 17l10 5 10-5" />
                </svg>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={styles.itemTitle}>{s.title || s.name}</div>
                  {s.description && (
                    <div style={styles.itemDesc}>{s.description}</div>
                  )}
                </div>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" style={styles.editIcon}>
                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                </svg>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 10000,
    display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
    paddingTop: 60, paddingBottom: 40,
    backgroundColor: 'rgba(0, 0, 0, 0.25)',
  },
  modal: {
    width: 480, maxWidth: '94vw', borderRadius: 14,
    backgroundColor: 'var(--surface)',
    boxShadow: '0 20px 60px rgba(0,0,0,0.12)',
    overflow: 'hidden', display: 'flex', flexDirection: 'column',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '20px 24px 16px', borderBottom: '1px solid var(--border)',
  },
  title: {
    margin: 0, fontSize: 18, fontWeight: 700, color: 'var(--text)',
  },
  newBtn: {
    background: 'var(--accent)', border: 'none', color: '#fff',
    fontSize: 12, fontWeight: 500, cursor: 'pointer',
    padding: '7px 14px', borderRadius: 6,
    display: 'flex', alignItems: 'center', gap: 4,
    fontFamily: 'var(--font)',
  },
  closeBtn: {
    background: 'none', border: 'none', fontSize: 18, cursor: 'pointer',
    color: 'var(--text-secondary)', padding: '4px 8px',
    fontFamily: 'var(--font)',
  },
  body: {
    padding: '8px 0', maxHeight: '55vh', overflowY: 'auto',
  },
  empty: {
    padding: '40px 24px', textAlign: 'center',
    color: 'var(--text-secondary)', fontSize: 13,
  },
  item: {
    display: 'flex', alignItems: 'center', gap: 10,
    padding: '10px 24px', cursor: 'pointer',
    transition: 'background 0.12s',
  },
  skillIcon: {
    width: 18, height: 18, flexShrink: 0, color: 'var(--accent)',
  },
  itemTitle: {
    fontSize: 14, fontWeight: 500, color: 'var(--text)',
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  itemDesc: {
    fontSize: 12, color: 'var(--text-secondary)', marginTop: 2,
    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
  },
  editIcon: {
    width: 14, height: 14, flexShrink: 0, color: 'var(--text-secondary)',
  },
};

export default SkillListModal;