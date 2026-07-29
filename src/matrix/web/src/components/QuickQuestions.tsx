import React, { useState } from 'react';

interface Props {
  onSend: (question: string) => void;
}

const PRESET_QUESTIONS = [
  '持仓异动诊断',
  '组合复盘',
  '配置偏离检查',
  '投资研究',
  '生成图片',
];

const QuickQuestions: React.FC<Props> = ({ onSend }) => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div style={styles.container}>
      <div
        style={styles.header}
        onClick={() => setCollapsed(!collapsed)}
      >
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>
          常用问题
        </span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          {collapsed ? '▶' : '▼'}
        </span>
      </div>
      {!collapsed && (
        <div style={styles.buttons}>
          {PRESET_QUESTIONS.map((q) => (
            <button
              key={q}
              style={styles.button}
              onClick={() => onSend(q)}
            >
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    borderBottom: '1px solid var(--rule, #333)',
  },
  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '10px 16px', cursor: 'pointer', userSelect: 'none',
  },
  buttons: {
    display: 'flex', flexWrap: 'wrap', gap: 8, padding: '0 16px 12px',
  },
  button: {
    padding: '6px 14px', borderRadius: 16, border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)', color: 'var(--text, #e0e0e0)',
    fontSize: 12, fontWeight: 500, cursor: 'pointer', transition: 'all 0.15s',
    whiteSpace: 'nowrap' as const,
  },
};

export default QuickQuestions;