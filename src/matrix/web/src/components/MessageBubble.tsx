import React, { useMemo } from 'react';
import type { Message } from '../types';
import { renderMarkdown } from '../utils/markdown';
import AgentChain from './AgentChain';
import ToolSection from './ToolSection';

interface Props {
  message: Message;
}

const MessageBubble: React.FC<Props> = ({ message }) => {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';

  const renderedContent = useMemo(() => {
    if (isUser || isSystem) return null;
    return renderMarkdown(message.content);
  }, [message.content, isUser, isSystem]);

  return (
    <div style={{ ...styles.wrapper, ...(isUser ? styles.wrapperUser : styles.wrapperAI) }}>
      {message.agentChain && message.agentChain.length > 0 && (
        <AgentChain steps={message.agentChain} />
      )}

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

      {message.duration && (
        <div style={styles.duration}>{message.duration}</div>
      )}

      {message.toolResults && message.toolResults.length > 0 && (
        <ToolSection results={message.toolResults} />
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    marginBottom: 16,
    maxWidth: '85%',
  },
  wrapperUser: {
    alignSelf: 'flex-end',
    alignItems: 'flex-end',
  },
  wrapperAI: {
    alignSelf: 'flex-start',
    alignItems: 'flex-start',
  },
  bubble: {
    padding: '12px 16px',
    borderRadius: 14,
    fontSize: 14,
    lineHeight: 1.7,
    wordBreak: 'break-word',
  },
  bubbleUser: {
    backgroundColor: 'var(--accent, #5b8def)',
    color: '#fff',
    borderBottomRightRadius: 4,
  },
  bubbleAI: {
    backgroundColor: 'var(--bg2, #222)',
    color: 'var(--text, #e0e0e0)',
    borderBottomLeftRadius: 4,
    border: '1px solid var(--rule, #333)',
  },
  bubbleSystem: {
    backgroundColor: 'rgba(255, 200, 60, 0.1)',
    color: 'var(--muted, #888)',
    fontSize: 12,
    fontStyle: 'italic',
    alignSelf: 'center',
    maxWidth: '90%',
    textAlign: 'center',
    border: '1px dashed var(--rule, #333)',
  },
  contentText: {
    whiteSpace: 'pre-wrap',
  },
  contentMarkdown: {
    // Markdown content styling is handled via global CSS
  },
  cursor: {
    display: 'inline-block',
    width: 8,
    height: 16,
    backgroundColor: 'var(--accent, #5b8def)',
    marginLeft: 2,
    verticalAlign: 'text-bottom',
    animation: 'blink 1s step-end infinite',
  },
  error: {
    marginTop: 8,
    padding: '8px 12px',
    borderRadius: 8,
    backgroundColor: 'rgba(255, 80, 80, 0.1)',
    color: '#ff5050',
    fontSize: 12,
  },
  duration: {
    marginTop: 4,
    fontSize: 11,
    color: 'var(--muted, #888)',
  },
};

export default MessageBubble;