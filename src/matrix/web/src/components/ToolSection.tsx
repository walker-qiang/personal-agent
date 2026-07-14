import React, { useState } from 'react';
import type { ToolResult } from '../types';
import { formatToolResult, formatDuration } from '../utils/format';

interface Props {
  results: ToolResult[];
}

const ToolSection: React.FC<Props> = ({ results }) => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggle = (id: string) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  if (!results || results.length === 0) return null;

  return (
    <div style={styles.container}>
      {results.map((r) => {
        const isOpen = !!expanded[r.id];
        const formatted = formatToolResult(r.name, r.result);
        const hasError = !!r.error;

        return (
          <div key={r.id} style={styles.item}>
            <button
              style={styles.header}
              onClick={() => toggle(r.id)}
            >
              <span style={styles.arrow}>{isOpen ? '\u25BC' : '\u25B6'}</span>
              <span style={styles.toolName}>{r.name}</span>
              {r.duration_ms != null && (
                <span style={styles.duration}>{formatDuration(r.duration_ms)}</span>
              )}
              {hasError && <span style={styles.errorBadge}>错误</span>}
            </button>

            {isOpen && (
              <div style={styles.body}>
                {hasError && (
                  <div style={styles.error}>{r.error}</div>
                )}
                <div
                  style={styles.result}
                  dangerouslySetInnerHTML={{ __html: formatted }}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
    marginTop: 8,
  },
  item: {
    borderRadius: 8,
    border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    width: '100%',
    padding: '8px 12px',
    border: 'none',
    backgroundColor: 'transparent',
    color: 'var(--text, #e0e0e0)',
    fontSize: 13,
    cursor: 'pointer',
    transition: 'background-color 0.15s',
    textAlign: 'left' as const,
  },
  arrow: {
    fontSize: 10,
    color: 'var(--muted, #888)',
    flexShrink: 0,
  },
  toolName: {
    fontWeight: 600,
    color: 'var(--accent, #5b8def)',
  },
  duration: {
    marginLeft: 'auto',
    fontSize: 11,
    color: 'var(--muted, #888)',
    flexShrink: 0,
  },
  errorBadge: {
    fontSize: 11,
    padding: '2px 8px',
    borderRadius: 4,
    backgroundColor: 'rgba(255, 80, 80, 0.15)',
    color: '#ff5050',
    flexShrink: 0,
  },
  body: {
    padding: '0 12px 12px',
    borderTop: '1px solid var(--rule, #333)',
  },
  error: {
    marginTop: 10,
    padding: '8px 12px',
    borderRadius: 6,
    backgroundColor: 'rgba(255, 80, 80, 0.1)',
    color: '#ff5050',
    fontSize: 12,
    lineHeight: 1.5,
  },
  result: {
    marginTop: 10,
    fontSize: 13,
    color: 'var(--text, #e0e0e0)',
    lineHeight: 1.6,
    overflowX: 'auto',
  },
};

export default ToolSection;