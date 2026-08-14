import React, { useMemo, useState, useEffect, useRef } from 'react';
import type { Message } from '../types';
import { renderMarkdown } from '../utils/markdown';
import AgentChain from './AgentChain';
import ToolSection from './ToolSection';

interface Props {
  message: Message;
  onBranch?: (messageId: string) => void;
}

const MessageBubble: React.FC<Props> = ({ message, onBranch }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const [hovering, setHovering] = useState(false);
  // Thinking accordion: default collapsed, auto-expand during streaming
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const userToggledRef = useRef(false);
  const hadContentRef = useRef(false);

  const renderedContent = useMemo(() => {
    if (isUser || isSystem) return null;
    return renderMarkdown(message.content);
  }, [message.content, isUser, isSystem]);

  const hasThinking = message.thinking && message.thinking.length > 0;
  const hasToolResults = message.toolResults && message.toolResults.length > 0;
  const hasDebugTrace = message.debugTrace && message.debugTrace.length > 0;
  const hasThinkingGroup = hasThinking || hasToolResults || hasDebugTrace;

  // Auto-expand during streaming, collapse when answer arrives
  const expandTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (userToggledRef.current) return;
    if (!hasThinkingGroup) return;

    if (message.isStreaming && !message.content) {
      setThinkingOpen(true);
    } else if (message.content && !hadContentRef.current) {
      hadContentRef.current = true;
      if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
      expandTimerRef.current = setTimeout(() => {
        if (!userToggledRef.current) {
          setThinkingOpen(false);
        }
      }, 600);
    }
  }, [message.isStreaming, message.content, hasThinkingGroup]);

  useEffect(() => {
    return () => {
      if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
    };
  }, []);

  const handleThinkingToggle = () => {
    userToggledRef.current = true;
    setThinkingOpen(!thinkingOpen);
  };

  // Build accordion meta: step count + tool count
  const headerParts: string[] = [];
  if (hasThinking) headerParts.push(`${message.thinking!.length} 步`);
  if (hasToolResults) headerParts.push(`${message.toolResults!.length} 次调用`);

  return (
    <div
      style={{ ...styles.wrapper, ...(isUser ? styles.wrapperUser : styles.wrapperAI) }}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      {message.agentChain && message.agentChain.length > 0 && (
        <AgentChain steps={message.agentChain} />
      )}

      {isUser ? (
        <div style={{ ...styles.bubble, ...styles.bubbleUser }}>
          <div style={styles.contentText}>{message.content}</div>
          {message.isStreaming && <span style={styles.cursor} />}
          {message.error && <div style={styles.error}>{message.error}</div>}
          {onBranch && message.message_id && hovering && (
            <button style={styles.branchBtn} onClick={() => onBranch(message.message_id!)} title="从此处分叉">↷</button>
          )}
        </div>
      ) : isSystem ? (
        <div style={{ ...styles.bubble, ...styles.bubbleSystem }}>
          <div style={styles.contentText}>{message.content}</div>
        </div>
      ) : (
        <>
          {/* Thinking accordion — "正在分析查询 ▼", default collapsed */}
          {hasThinkingGroup && (
            <div style={styles.thinkingGroup}>
              <div
                style={styles.thinkingHeader}
                onClick={handleThinkingToggle}
              >
                <span style={styles.thinkingIcon}>
                  {message.isStreaming && !message.content ? '◐' : '✓'}
                </span>
                <span style={styles.thinkingLabel}>
                  {message.isStreaming && !message.content ? '正在分析查询' : '分析完成'}
                </span>
                {headerParts.length > 0 && (
                  <span style={styles.thinkingMeta}>
                    {headerParts.join(' · ')}
                  </span>
                )}
                <span style={{ ...styles.thinkingArrow, transform: thinkingOpen ? 'rotate(180deg)' : 'none' }}>▼</span>
              </div>
              {thinkingOpen && (
                <div style={styles.thinkingBody}>
                  {hasThinking && message.thinking!.map((t, i) => (
                    <div key={`think-${i}`} style={styles.thinkingItem}>{t}</div>
                  ))}
                  {hasToolResults && (
                    <ToolSection results={message.toolResults!} embedded />
                  )}
                  {hasDebugTrace && (
                    <div style={styles.debugTraceBody}>
                      <div style={styles.debugTraceTitle}>Runtime DebugTrace（临时）</div>
                      {message.debugTrace!.map((trace) => (
                        <div key={`${trace.sequence}-${trace.kind}`} style={styles.debugTraceItem}>
                          <div style={styles.debugTraceKind}>
                            #{trace.sequence} {trace.kind}
                          </div>
                          <pre style={styles.debugTracePayload}>
                            {JSON.stringify(trace.payload, null, 2)}
                          </pre>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Final answer — independent card */}
          {(renderedContent || message.error) && (
            <div style={{ ...styles.bubble, ...styles.bubbleAI }}>
              {renderedContent && (
                <div
                  className="markdown-body"
                  style={styles.contentMarkdown}
                  dangerouslySetInnerHTML={{ __html: renderedContent }}
                />
              )}
              {message.isStreaming && renderedContent && <span style={styles.cursor} />}
              {message.error && <div style={styles.error}>{message.error}</div>}
            </div>
          )}

          {/* Empty state — lightweight hint */}
          {!renderedContent && !message.error && !message.isStreaming && hasThinkingGroup && (
            <div style={styles.emptyState}>未返回数据</div>
          )}

          {onBranch && message.message_id && hovering && (
            <button style={styles.branchBtn} onClick={() => onBranch(message.message_id!)} title="从此处分叉">↷</button>
          )}
        </>
      )}

      {message.duration && (
        <div style={styles.duration}>{message.duration}</div>
      )}

      {message.interrupted && (
        <div style={styles.interrupted}>回答因页面刷新被中断</div>
      )}

      {message.progress && message.progress.length > 0 && (
        <div style={styles.progressSection}>
          {message.progress.map((p, i) => (
            <div key={i} style={styles.progressItem}>
              <span style={styles.progressDot}>⏳</span>
              <span>{p}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: 'flex', flexDirection: 'column', maxWidth: 800, width: '100%',
  },
  wrapperUser: { marginRight: 0, marginLeft: 'auto', alignItems: 'flex-end', maxWidth: '80%' },
  wrapperAI: { marginRight: 'auto', marginLeft: 0, alignItems: 'flex-start' },
  bubble: { padding: '12px 16px', borderRadius: 'var(--radius-bubble)', fontSize: 14, lineHeight: 1.6, wordBreak: 'break-word', display: 'inline-block', textAlign: 'left' },
  bubbleUser: {
    backgroundColor: 'var(--user-bubble)', color: '#fff',
    maxWidth: '100%',
  },
  bubbleAI: {
    backgroundColor: 'var(--surface)', color: 'var(--text)',
    border: '1px solid var(--border)',
    boxShadow: 'var(--shadow-sm)',
    padding: '20px 24px', maxWidth: '100%', width: '100%',
    lineHeight: 1.6, fontSize: 14,
    borderRadius: 'var(--radius)',
  },
  bubbleSystem: {
    backgroundColor: 'rgba(255, 200, 60, 0.1)', color: 'var(--text-secondary)',
    fontSize: 12, fontStyle: 'italic', alignSelf: 'center', maxWidth: '90%',
    textAlign: 'center', border: '1px dashed var(--border)',
  },
  contentText: { whiteSpace: 'normal', wordBreak: 'break-word' },
  contentMarkdown: {},
  cursor: {
    display: 'inline-block', width: 8, height: 16, backgroundColor: 'var(--accent)',
    marginLeft: 2, verticalAlign: 'text-bottom', animation: 'blink 1s step-end infinite',
  },
  error: { marginTop: 8, padding: '10px 14px', borderRadius: 'var(--radius)', backgroundColor: 'rgba(239,68,68,0.1)', border: '1px solid var(--error)', color: 'var(--error)', fontSize: 13 },
  duration: { marginTop: 6, fontSize: 11, color: 'var(--text-secondary)', opacity: 0.6 },
  interrupted: { marginTop: 6, fontSize: 11, color: 'var(--error)', opacity: 0.8 },
  branchBtn: {
    width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)',
    backgroundColor: 'var(--surface)', color: 'var(--text-secondary)',
    fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center',
    justifyContent: 'center', flexShrink: 0, opacity: 0.8,
  },
  // Thinking accordion — fintech style
  thinkingGroup: {
    borderRadius: 'var(--radius)',
    margin: '4px 0',
    overflow: 'hidden',
    background: 'transparent',
  },
  thinkingHeader: {
    padding: '8px 14px',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: 13,
    color: 'var(--text-secondary)',
    cursor: 'pointer',
    userSelect: 'none',
    background: 'transparent',
    borderRadius: 'var(--radius)',
    transition: 'background 0.15s',
  },
  thinkingIcon: { fontSize: 14, color: 'var(--accent)', flexShrink: 0 },
  thinkingLabel: { fontWeight: 500, color: 'var(--text-secondary)', fontSize: 13 },
  thinkingMeta: { fontSize: 11, color: 'var(--text-secondary)', opacity: 0.6 },
  thinkingArrow: { marginLeft: 'auto', fontSize: 10, transition: 'transform 0.2s', color: 'var(--text-secondary)', opacity: 0.4 },
  thinkingBody: {
    maxHeight: 2000, overflowY: 'auto' as const,
    padding: '4px 0',
  },
  thinkingItem: {
    padding: '8px 14px', fontSize: 13, lineHeight: 1.6,
    color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
    borderLeft: '2px solid rgba(99, 102, 241, 0.15)',
    margin: '4px 0',
    borderRadius: '0 var(--radius-sm) var(--radius-sm) 0',
    background: 'rgba(99, 102, 241, 0.02)',
    maxHeight: 200, overflowY: 'auto' as const,
  },
  debugTraceBody: {
    margin: '8px 14px', padding: '8px', borderRadius: 'var(--radius-sm)',
    background: 'rgba(15, 23, 42, 0.04)', border: '1px solid var(--border)',
  },
  debugTraceTitle: {
    marginBottom: 6, fontSize: 11, color: 'var(--text-secondary)', fontWeight: 600,
  },
  debugTraceItem: {
    marginTop: 6, paddingTop: 6, borderTop: '1px solid var(--border)',
  },
  debugTraceKind: {
    fontSize: 11, color: 'var(--accent)', fontFamily: 'var(--font-mono, monospace)',
  },
  debugTracePayload: {
    margin: '4px 0 0', maxHeight: 240, overflow: 'auto', whiteSpace: 'pre-wrap',
    wordBreak: 'break-word', fontSize: 10, lineHeight: 1.4,
    color: 'var(--text-secondary)', fontFamily: 'var(--font-mono, monospace)',
  },
  // Empty state — lightweight hint
  emptyState: {
    padding: '12px 16px', fontSize: 13, color: 'var(--text-secondary)',
    opacity: 0.6, fontStyle: 'italic',
  },
  progressSection: {
    marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4,
    padding: '8px 12px', borderRadius: 'var(--radius)',
    backgroundColor: 'rgba(16, 185, 129, 0.06)', border: '1px solid rgba(16, 185, 129, 0.15)',
  },
  progressItem: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' },
  progressDot: { fontSize: 12, flexShrink: 0 },
};

export default MessageBubble;
