import React, { useState } from 'react';

interface Props {
  onSend: (question: string) => void;
}

// Maps display text to the actual message sent (matches original HTML data-msg)
const HINTS: { label: string; msg: string }[] = [
  { label: '持仓异动诊断', msg: '诊断我的持仓异动' },
  { label: '组合复盘', msg: '做一次组合复盘' },
  { label: '配置偏离检查', msg: '检查配置偏离度' },
  { label: '持仓总览', msg: '当前持仓总金额是多少？' },
  { label: '配置分析', msg: '分析我当前的资产配置，并与目标对比' },
];

const QuickQuestions: React.FC<Props> = ({ onSend }) => {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div style={styles.section}>
      <div
        style={styles.header}
        onClick={() => setCollapsed(!collapsed)}
      >
        <svg
          width="14" height="14" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round"
          style={{
            ...styles.chevron,
            transform: collapsed ? 'rotate(-90deg)' : 'none',
          }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
        <span style={styles.headerText}>常用问题</span>
      </div>
      {!collapsed && (
        <div style={styles.list}>
          {HINTS.map((h) => (
            <span
              key={h.label}
              style={styles.hint}
              onClick={() => onSend(h.msg)}
            >
              {h.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  // Matches original .sb-section
  section: {
    display: 'flex',
    flexDirection: 'column',
    padding: '8px 0',
    borderTop: '1px solid var(--border)',
    overflow: 'hidden',
    flexGrow: 0,
    flexShrink: 0,
    minHeight: 0,
  },
  // Matches original .sb-section-hd
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '4px 16px 6px',
    cursor: 'pointer',
    userSelect: 'none',
  },
  headerText: {
    fontSize: 11,
    fontWeight: 600,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase',
    letterSpacing: '0.8px',
  },
  // Matches original .sb-chevron
  chevron: {
    width: 14,
    height: 14,
    flexShrink: 0,
    color: 'var(--text-secondary)',
    opacity: 0.5,
    transition: 'transform 0.15s',
  },
  // Matches original .sb-list (inside .sb-section)
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 1,
    padding: '0 8px',
  },
  // Matches original .sb-hint
  hint: {
    display: 'block',
    padding: '6px 16px 6px 28px',
    fontSize: 12,
    cursor: 'pointer',
    color: 'var(--text-secondary)',
    transition: 'color 0.15s',
  },
};

export default QuickQuestions;