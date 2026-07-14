import React from 'react';
import type { AgentStep } from '../types';

interface Props {
  steps: AgentStep[];
}

const statusIcon: Record<AgentStep['status'], string> = {
  pending: '\u23F3',
  running: '\u23F3',
  done: '\u2705',
  error: '\u274C',
};

const AgentChain: React.FC<Props> = ({ steps }) => {
  if (!steps || steps.length === 0) return null;

  return (
    <div style={styles.container}>
      {steps.map((step, idx) => (
        <div
          key={idx}
          style={{
            ...styles.step,
            ...(step.status === 'running' ? styles.stepRunning : {}),
          }}
        >
          <span style={styles.icon}>{statusIcon[step.status]}</span>
          <div style={styles.stepContent}>
            <span style={styles.agentName}>{step.agent}</span>
            <span style={styles.task}>{step.task}</span>
          </div>
        </div>
      ))}
    </div>
  );
};

const pulseKeyframes = `
@keyframes agent-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
`;

// Inject keyframes once
if (typeof document !== 'undefined') {
  const styleId = 'agent-chain-pulse';
  if (!document.getElementById(styleId)) {
    const style = document.createElement('style');
    style.id = styleId;
    style.textContent = pulseKeyframes;
    document.head.appendChild(style);
  }
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    marginBottom: 10,
    padding: '10px 14px',
    borderRadius: 10,
    backgroundColor: 'var(--bg2, #222)',
    border: '1px solid var(--rule, #333)',
  },
  step: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 8,
    fontSize: 13,
    padding: '4px 0',
  },
  stepRunning: {
    animation: 'agent-pulse 1.5s ease-in-out infinite',
  },
  icon: {
    fontSize: 14,
    lineHeight: '20px',
    flexShrink: 0,
  },
  stepContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    minWidth: 0,
  },
  agentName: {
    fontWeight: 600,
    color: 'var(--accent, #5b8def)',
    fontSize: 12,
  },
  task: {
    color: 'var(--muted, #888)',
    fontSize: 12,
    lineHeight: 1.5,
    wordBreak: 'break-word',
  },
};

export default AgentChain;