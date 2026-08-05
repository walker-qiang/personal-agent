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
        // Check for error in both r.error and r.result.error (error dicts)
        const resultAsObj = typeof r.result === 'object' && r.result !== null ? r.result as Record<string, unknown> : null;
        const errorInResult = resultAsObj && 'error' in resultAsObj && typeof resultAsObj.error === 'string';
        const hasError = !!r.error || !!errorInResult;
        const errorText = r.error || (errorInResult ? String((resultAsObj as Record<string, unknown>).error) : '');
        const displayResult = errorInResult ? null : r.result;
        const formatted = displayResult != null ? sanitizeHtml(formatToolResult(r.name, displayResult)) : '';
        const isEmpty = !formatted || formatted.trim() === '';

        return (
          <div key={r.id} style={styles.item}>
            <button
              style={styles.header}
              onClick={() => toggle(r.id)}
            >
              {/* Left: status icon + tool name */}
              <span style={hasError ? styles.toolIconError : styles.toolIconDone}>
                {hasError ? '✕' : '✓'}
              </span>
              <span style={styles.toolName}>{r.name}</span>

              {/* Right: duration + status label + expand arrow */}
              <span style={styles.rightSection}>
                {r.duration_ms != null && (
                  <span style={styles.duration}>{formatDuration(r.duration_ms)}</span>
                )}
                <span style={styles.expandArrow}>{isOpen ? '▾' : '▸'}</span>
              </span>
            </button>

            {isOpen && (
              <div style={styles.body}>
                {hasError && (
                  <div style={styles.error}>{errorText}</div>
                )}
                {isEmpty ? (
                  <div style={styles.emptyResult}>未返回数据</div>
                ) : (
                  <div
                    className="tool-body"
                    style={styles.result}
                    dangerouslySetInnerHTML={{ __html: formatted }}
                  />
                )}
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
    gap: 2,
    margin: '4px 0',
  },
  embeddedContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    margin: 0,
  },
  item: {
    borderRadius: 'var(--radius-sm)',
    overflow: 'hidden',
    background: 'transparent',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    width: '100%',
    padding: '6px 12px',
    border: 'none',
    background: 'transparent',
    color: 'var(--text-secondary)',
    fontSize: 12,
    cursor: 'pointer',
    textAlign: 'left' as const,
    fontFamily: 'var(--font)',
    transition: 'background 0.15s',
    borderRadius: 'var(--radius-sm)',
  },
  toolIconDone: {
    fontSize: 11,
    color: 'var(--success)',
    flexShrink: 0,
    width: 14,
    textAlign: 'center' as const,
  },
  toolIconError: {
    fontSize: 11,
    color: 'var(--error)',
    flexShrink: 0,
    width: 14,
    textAlign: 'center' as const,
  },
  toolName: {
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-secondary)',
    fontSize: 12,
    fontWeight: 400,
    opacity: 0.7,
  },
  rightSection: {
    marginLeft: 'auto',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  duration: {
    fontSize: 11,
    color: 'var(--text-secondary)',
    opacity: 0.5,
    flexShrink: 0,
    fontFamily: 'var(--font-mono)',
  },
  expandArrow: {
    fontSize: 9,
    color: 'var(--text-secondary)',
    opacity: 0.3,
    flexShrink: 0,
  },
  body: {
    padding: '6px 12px 8px 34px',
    background: 'transparent',
    fontSize: 12,
    overflowX: 'auto',
  },
  error: {
    marginBottom: 6,
    padding: '6px 10px',
    borderRadius: 'var(--radius-sm)',
    backgroundColor: 'rgba(239, 68, 68, 0.06)',
    color: 'var(--error)',
    fontSize: 11.5,
    lineHeight: 1.5,
  },
  result: {
    fontSize: 11.5,
    color: 'var(--text-secondary)',
    lineHeight: 1.55,
    fontFamily: 'var(--font-mono)',
    whiteSpace: 'pre-wrap',
    opacity: 0.8,
  },
  emptyResult: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    opacity: 0.5,
    fontStyle: 'italic',
  },
};

export default ToolSection;
