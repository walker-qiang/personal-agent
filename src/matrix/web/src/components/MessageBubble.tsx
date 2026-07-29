import React, { useMemo, useState } from 'react';
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

  const renderedContent = useMemo(() => {
    if (isUser || isSystem) return null;
    return renderMarkdown(message.content);
  }, [message.content, isUser, isSystem]);

  return (
    <div
      style={{ ...styles.wrapper, ...(isUser ? styles.wrapperUser : styles.wrapperAI) }}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
    >
      {message.agentChain && message.agentChain.length > 0 && (
        <AgentChain steps={message.agentChain} />
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexDirection: isUser ? 'row-reverse' : 'row' }}>
        <div
          style={{
            ...styles.bubble,
            ...(isUser ? styles.bubbleUser : styles.bubbleAI),
            ...(isSystem ? styles.bubbleSystem : {}),
          }}
        >
          {isUser || isSystem ? (
            <div style={styles.contentText}>{message.content}</div>
          ) : (
            <div
              style={styles.contentMarkdown}
              dangerouslySetInnerHTML={{ __html: renderedContent || '' }}
            />
          )}

          {message.isStreaming && (
            <span style={styles.cursor} />
          )}

          {message.error && (
            <div style={styles.error}>{message.error}</div>
          )}
        </div>

        {onBranch && message.message_id && hovering && (
          <button
            style={styles.branchBtn}
            onClick={() => onBranch(message.message_id!)}
            title="从此处分叉"
          >
            ↷
          </button>
        )}
      </div>

      {message.duration && (
        <div style={styles.duration}>{message.duration}</div>
      )}

      {message.thinking && message.thinking.length > 0 && (
        <div style={styles.thinkingSection}>
          <div style={styles.thinkingHeader}>
            <span style={styles.thinkingIcon}>💭</span>
            <span style={styles.thinkingLabel}>思考过程 ({message.thinking.length} 条)</span>
          </div>
          {message.thinking.map((t, i) => (
            <div key={i} style={styles.thinkingItem}>{t}</div>
          ))}
        </div>
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

      {message.toolResults && message.toolResults.length > 0 && (
        <ToolSection results={message.toolResults} />
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: 'flex', flexDirection: 'column', marginBottom: 16, maxWidth: '85%',
  },
  wrapperUser: { alignSelf: 'flex-end', alignItems: 'flex-end' },
  wrapperAI: { alignSelf: 'flex-start', alignItems: 'flex-start' },
  bubble: { padding: '12px 16px', borderRadius: 14, fontSize: 14, lineHeight: 1.7, wordBreak: 'break-word' },
  bubbleUser: { backgroundColor: 'var(--accent, #5b8def)', color: '#fff', borderBottomRightRadius: 4 },
  bubbleAI: { backgroundColor: 'var(--bg2, #222)', color: 'var(--text, #e0e0e0)', borderBottomLeftRadius: 4, border: '1px solid var(--rule, #333)' },
  bubbleSystem: {
    backgroundColor: 'rgba(255, 200, 60, 0.1)', color: 'var(--muted, #888)',
    fontSize: 12, fontStyle: 'italic', alignSelf: 'center', maxWidth: '90%',
    textAlign: 'center', border: '1px dashed var(--rule, #333)',
  },
  contentText: { whiteSpace: 'pre-wrap' },
  contentMarkdown: {},
  cursor: {
    display: 'inline-block', width: 8, height: 16, backgroundColor: 'var(--accent, #5b8def)',
    marginLeft: 2, verticalAlign: 'text-bottom', animation: 'blink 1s step-end infinite',
  },
  error: { marginTop: 8, padding: '8px 12px', borderRadius: 8, backgroundColor: 'rgba(255, 80, 80, 0.1)', color: '#ff5050', fontSize: 12 },
  duration: { marginTop: 4, fontSize: 11, color: 'var(--muted, #888)' },
  branchBtn: {
    width: 26, height: 26, borderRadius: 13, border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)', color: 'var(--accent, #5b8def)',
    fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center',
    justifyContent: 'center', flexShrink: 0, transition: 'opacity 0.15s',
    opacity: 0.8,
  },
  thinkingSection: {
    marginTop: 8, padding: '10px 14px', borderRadius: 10,
    backgroundColor: 'rgba(99, 102, 241, 0.06)', border: '1px solid rgba(99, 102, 241, 0.15)',
    fontSize: 12, color: 'var(--muted)', lineHeight: 1.6,
  },
  thinkingHeader: { display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 },
  thinkingIcon: { fontSize: 13 },
  thinkingLabel: { fontWeight: 600, color: 'var(--text-secondary)' },
  thinkingItem: {
    padding: '4px 0', borderTop: '1px solid rgba(99, 102, 241, 0.08)',
    whiteSpace: 'pre-wrap', wordBreak: 'break-word',
  },
  progressSection: {
    marginTop: 8, display: 'flex', flexDirection: 'column', gap: 4,
    padding: '8px 12px', borderRadius: 8,
    backgroundColor: 'rgba(52, 199, 89, 0.06)', border: '1px solid rgba(52, 199, 89, 0.15)',
  },
  progressItem: { display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--muted)' },
  progressDot: { fontSize: 12, flexShrink: 0 },
};

export default MessageBubble;