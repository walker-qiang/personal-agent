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
  // Thinking group: auto-expand during streaming, auto-collapse when answer starts
  // (matches original HTML: create as "open", collapse on first token)
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const userToggledRef = useRef(false);
  const hadContentRef = useRef(false);

  const renderedContent = useMemo(() => {
    if (isUser || isSystem) return null;
    return renderMarkdown(message.content);
  }, [message.content, isUser, isSystem]);

  const hasThinking = message.thinking && message.thinking.length > 0;
  const hasToolResults = message.toolResults && message.toolResults.length > 0;
  // Original HTML creates thinking-group when either thinking or tool_call arrives
  const hasThinkingGroup = hasThinking || hasToolResults;

  // Auto-expand/collapse to match original HTML behavior:
  // 1. When streaming and thinking group exists but no answer content yet → expand
  // 2. When answer content starts arriving → collapse
  // 3. After user manually toggles, stop auto-control
  // 4. Historical (non-streaming) messages: default expanded so user can see what happened
  const expandTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (userToggledRef.current) return;
    if (!hasThinkingGroup) return;

    if (message.isStreaming && !message.content) {
      // Streaming with thinking/tools but no answer yet → expand
      setThinkingOpen(true);
    } else if (message.content && !hadContentRef.current) {
      // First answer content arriving → collapse after a short delay
      // so user has time to see the thinking process
      hadContentRef.current = true;
      if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
      expandTimerRef.current = setTimeout(() => {
        if (!userToggledRef.current) {
          setThinkingOpen(false);
        }
      }, 800);
    }
  }, [message.isStreaming, message.content, hasThinkingGroup]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
    };
  }, []);

  const handleThinkingToggle = () => {
    userToggledRef.current = true;
    setThinkingOpen(!thinkingOpen);
  };

  // Build header parts: "思考过程 · X 步思考 · Y 次工具调用" (matches original HTML)
  const headerParts: string[] = [];
  if (hasThinking) headerParts.push(`${message.thinking!.length} 步思考`);
  if (hasToolResults) headerParts.push(`${message.toolResults!.length} 次工具调用`);

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
          {/* Thinking group — contains thinking text AND tool results, BEFORE answer content (matches original HTML) */}
          {hasThinkingGroup && (
            <div style={styles.thinkingGroup}>
              <div
                style={styles.thinkingHeader}
                onClick={handleThinkingToggle}
              >
                <span style={styles.thinkingIcon}>💡</span>
                <span style={styles.thinkingLabel}>
                  思考过程{headerParts.length > 0 ? ' · ' + headerParts.join(' · ') : ''}
                </span>
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
                </div>
              )}
            </div>
          )}

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

          {onBranch && message.message_id && hovering && (
            <button style={styles.branchBtn} onClick={() => onBranch(message.message_id!)} title="从此处分叉">↷</button>
          )}
        </>
      )}

      {message.duration && (
        <div style={styles.duration}>{message.duration}</div>
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
  wrapperUser: { marginRight: 0, marginLeft: 'auto', alignItems: 'flex-end' },
  wrapperAI: { marginRight: 'auto', marginLeft: 0, alignItems: 'flex-start' },
  bubble: { padding: '10px 16px', borderRadius: 'var(--radius)', fontSize: 14, lineHeight: 1.6, wordBreak: 'break-word', display: 'inline-block', maxWidth: '85%', textAlign: 'left' },
  bubbleUser: { backgroundColor: 'var(--accent)', color: '#fff' },
  bubbleAI: { backgroundColor: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--border)', padding: '20px 24px', maxWidth: '100%', lineHeight: 1.85, fontSize: 14 },
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
  branchBtn: {
    width: 28, height: 28, borderRadius: 6, border: '1px solid var(--border)',
    backgroundColor: 'var(--surface)', color: 'var(--text-secondary)',
    fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center',
    justifyContent: 'center', flexShrink: 0, opacity: 0.8,
  },
  // Thinking group — matches original HTML .thinking-group
  thinkingGroup: {
    border: '1px solid var(--border)', borderRadius: 'var(--radius)',
    margin: '8px 0', overflow: 'hidden',
  },
  thinkingHeader: {
    background: 'var(--tool-call)', padding: '7px 12px',
    display: 'flex', alignItems: 'center', gap: 8,
    fontSize: 12, color: 'var(--text-secondary)', cursor: 'pointer',
    userSelect: 'none',
  },
  thinkingIcon: { fontSize: 13 },
  thinkingLabel: { fontWeight: 600, color: 'var(--text-secondary)', fontSize: 12 },
  thinkingArrow: { marginLeft: 'auto', fontSize: 10, transition: 'transform 0.15s', color: 'var(--text-secondary)' },
  thinkingBody: {
    maxHeight: 2000, overflowY: 'auto' as const,
  },
  thinkingItem: {
    padding: '10px 12px', fontSize: 12, lineHeight: 1.6,
    color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' as const,
    wordBreak: 'break-word' as const,
    borderBottom: '1px solid var(--border)', background: 'var(--tool-result)',
    maxHeight: 200, overflowY: 'auto' as const,
  },
  progressSection: {
    marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4,
    padding: '8px 12px', borderRadius: 'var(--radius)',
    backgroundColor: 'rgba(52, 199, 89, 0.06)', border: '1px solid rgba(52, 199, 89, 0.15)',
  },
  progressItem: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text-secondary)' },
  progressDot: { fontSize: 12, flexShrink: 0 },
};

export default MessageBubble;