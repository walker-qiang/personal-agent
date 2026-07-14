import React from 'react';

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
  return (
    <div style={styles.container}>
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
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 8,
    padding: '12px 16px',
  },
  button: {
    padding: '8px 16px',
    borderRadius: 20,
    border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)',
    color: 'var(--text, #e0e0e0)',
    fontSize: 13,
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.15s',
    whiteSpace: 'nowrap' as const,
  },
};

export default QuickQuestions;