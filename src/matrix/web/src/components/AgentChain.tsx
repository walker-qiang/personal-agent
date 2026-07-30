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
    flexWrap: 'wrap',
    gap: 8,
    margin: '0 0 8px',
    padding: 0,
  },
  step: {
    display: 'flex',
    alignItems: 'center',
    gap: 7,
    fontSize: 12,
    padding: '6px 12px',
    borderRadius: 20,
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    color: 'var(--text-secondary)',
  },
  stepRunning: {
    borderColor: 'var(--accent)',
    color: 'var(--accent)',
  },
  icon: {
    fontSize: 12,
    flexShrink: 0,
  },
  stepContent: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    minWidth: 0,
  },
  agentName: {
    fontWeight: 600,
    color: 'inherit',
    fontSize: 12,
  },
  task: {
    color: 'inherit',
    fontSize: 12,
    maxWidth: 220,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
};

export default AgentChain;