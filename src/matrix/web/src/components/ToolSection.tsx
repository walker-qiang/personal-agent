import React, { useState } from 'react';
import type { ToolResult } from '../types';
import { formatToolResult, formatDuration } from '../utils/format';
import { sanitizeHtml } from '../utils/markdown';

interface Props {
  results: ToolResult[];
  /** When true, tool sections are rendered without outer border/radius (inside thinking-group-body). */
  embedded?: boolean;
}

const ToolSection: React.FC<Props> = ({ results, embedded = false }) => {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const toggle = (id: string) => {
    setExpanded((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  if (!results || results.length === 0) return null;

  return (
    <div style={embedded ? styles.embeddedContainer : styles.container}>
      {results.map((r) => {
        const isOpen = !!expanded[r.id];
        const formatted = sanitizeHtml(formatToolResult(r.name, r.result));
        const hasError = !!r.error;

        return (
          <div key={r.id} style={embedded ? styles.embeddedItem : styles.item}>
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
                  className="tool-body"
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
    gap: 0,
    margin: '6px 0',
  },
  item: {
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    overflow: 'hidden',
  },
  // Embedded mode: inside thinking-group-body, no outer border/radius
  embeddedContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: 0,
    margin: 0,
  },
  embeddedItem: {
    borderTop: '1px solid var(--border)',
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    width: '100%',
    padding: '7px 12px',
    border: 'none',
    backgroundColor: 'var(--tool-call)',
    color: 'var(--text-secondary)',
    fontSize: 12,
    cursor: 'pointer',
    textAlign: 'left' as const,
    fontFamily: 'var(--font)',
  },
  arrow: {
    fontSize: 10,
    color: 'var(--text-secondary)',
    flexShrink: 0,
    marginLeft: 'auto',
    transition: 'transform 0.15s',
  },
  toolName: {
    fontFamily: 'var(--font-mono)',
    color: 'var(--warning)',
    fontSize: 12,
  },
  duration: {
    fontSize: 11,
    color: 'var(--text-secondary)',
    flexShrink: 0,
  },
  errorBadge: {
    fontSize: 11,
    padding: '2px 8px',
    borderRadius: 4,
    backgroundColor: 'rgba(255, 59, 48, 0.1)',
    color: 'var(--error)',
    flexShrink: 0,
  },
  body: {
    padding: '10px 12px',
    borderTop: '1px solid var(--border)',
    background: 'var(--tool-result)',
    fontSize: 12,
    overflowX: 'auto',
  },
  error: {
    marginBottom: 8,
    padding: '8px 12px',
    borderRadius: 'var(--radius)',
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    border: '1px solid var(--error)',
    color: 'var(--error)',
    fontSize: 12,
    lineHeight: 1.5,
  },
  result: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
    fontFamily: 'var(--font-mono)',
    whiteSpace: 'pre-wrap',
  },
};

export default ToolSection;
