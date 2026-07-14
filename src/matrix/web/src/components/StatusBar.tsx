import React from 'react';

interface Props {
  status: 'idle' | 'thinking' | 'executing' | 'generating';
  text?: string;
}

const STATUS_LABELS: Record<Props['status'], string> = {
  idle: '就绪',
  thinking: '思考中',
  executing: '执行中',
  generating: '生成中',
};

const DOT_COLORS: Record<Props['status'], string> = {
  idle: '#4caf50',
  thinking: '#ff9800',
  executing: '#2196f3',
  generating: '#9c27b0',
};

const StatusBar: React.FC<Props> = ({ status, text }) => {
  const label = text || STATUS_LABELS[status];
  const dotColor = DOT_COLORS[status];
  const isAnimating = status !== 'idle';

  return (
    <div style={styles.container}>
      <div style={styles.dots}>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            style={{
              ...styles.dot,
              backgroundColor: dotColor,
              ...(isAnimating ? styles.dotAnimating : {}),
              animationDelay: `${i * 0.2}s`,
            }}
          />
        ))}
      </div>
      <span style={styles.text}>{label}</span>
    </div>
  );
};

const dotKeyframes = `
@keyframes status-dot {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}
`;

// Inject keyframes once
if (typeof document !== 'undefined') {
  const styleId = 'status-bar-dots';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = dotKeyframes;
    document.head.appendChild(style);
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    padding: '8px 16px',
    backgroundColor: 'var(--bg, #1a1a2e)',
    borderTop: '1px solid var(--rule, #333)',
    height: 36,
  },
  dots: {
    display: 'flex',
    gap: 4,
    alignItems: 'center',
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: '50%',
    display: 'inline-block',
  },
  dotAnimating: {
    animation: 'status-dot 1.4s ease-in-out infinite',
  },
  text: {
    fontSize: 12,
    fontWeight: 500,
    color: 'var(--muted, #888)',
  },
};

export default StatusBar;