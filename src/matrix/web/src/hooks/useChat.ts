import { useState, useCallback, useRef, useEffect } from 'react';
import type { Message, ToolCall, ToolResult, AgentStep } from '../types';
import { buildStreamUrl } from '../utils/api';

interface ConfirmAction {
  agent: string;
  tool: string;
  args: Record<string, unknown>;
  summary: string;
}

export interface UseChatReturn {
  messages: Message[];
  send: (message: string, sessionId: string, fileId?: string) => void;
  sending: boolean;
  clearMessages: () => void;
  confirmRequired: boolean;
  confirmActions: ConfirmAction[];
  confirmSessionId: string;
  confirm: (decision: 'approve' | 'skip') => void;
  dismissConfirm: () => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sending, setSending] = useState<boolean>(false);
  const [confirmRequired, setConfirmRequired] = useState(false);
  const [confirmActions, setConfirmActions] = useState<ConfirmAction[]>([]);
  const [confirmSessionId, setConfirmSessionId] = useState('');
  const eventSourceRef = useRef<EventSource | null>(null);
  const pendingSessionRef = useRef<string>('');

  // cleanup on unmount
  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  const setupEventListeners = useCallback((es: EventSource, isResume: boolean = false) => {
    const assistantId = isResume ? crypto.randomUUID() : '';
    if (isResume) {
      const assistantMsg: Message = {
        id: assistantId,
        role: 'assistant',
        content: '',
        isStreaming: true,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    }

    // token
    es.addEventListener('token', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const token: string =
          typeof parsed === 'string'
            ? parsed
            : parsed.content || parsed.data?.content || '';
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          const last = updated[lastIdx];
          if (last && last.role === 'assistant') {
            updated[lastIdx] = {
              ...last,
              content: last.content + token,
            };
          }
          return updated;
        });
      } catch {
        // ignore
      }
    });

    // classify
    es.addEventListener('classify', () => {});

    // tool_call
    es.addEventListener('tool_call', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const toolCall: ToolCall = parsed.data || parsed;
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          const last = updated[lastIdx];
          if (last && last.role === 'assistant') {
            updated[lastIdx] = {
              ...last,
              toolCalls: [...(last.toolCalls || []), toolCall],
            };
          }
          return updated;
        });
      } catch {
        // ignore
      }
    });

    // tool_result
    es.addEventListener('tool_result', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const toolResult: ToolResult = parsed.data || parsed;
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          const last = updated[lastIdx];
          if (last && last.role === 'assistant') {
            updated[lastIdx] = {
              ...last,
              toolResults: [...(last.toolResults || []), toolResult],
            };
          }
          return updated;
        });
      } catch {
        // ignore
      }
    });

    // agent_result
    es.addEventListener('agent_result', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const step: AgentStep = parsed.data || parsed;
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          const last = updated[lastIdx];
          if (last && last.role === 'assistant') {
            updated[lastIdx] = {
              ...last,
              agentChain: [...(last.agentChain || []), step],
            };
          }
          return updated;
        });
      } catch {
        // ignore
      }
    });

    // confirm_required
    es.addEventListener('confirm_required', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const actions = parsed.actions || parsed.data?.actions || [];
        const sid = parsed.session_id || parsed.data?.session_id || '';
        setConfirmActions(actions);
        setConfirmSessionId(sid);
        setConfirmRequired(true);
        pendingSessionRef.current = sid;
      } catch {
        // ignore
      }
      es.close();
      eventSourceRef.current = null;
      setSending(false);
      // Mark last assistant message as not streaming
      setMessages((prev) => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        const last = updated[lastIdx];
        if (last && last.role === 'assistant') {
          updated[lastIdx] = { ...last, isStreaming: false };
        }
        return updated;
      });
    });

    // done
    es.addEventListener('done', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const duration: string | undefined =
          parsed.duration || parsed.data?.duration;
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          const last = updated[lastIdx];
          if (last && last.role === 'assistant') {
            updated[lastIdx] = {
              ...last,
              isStreaming: false,
              ...(duration ? { duration } : {}),
            };
          }
          return updated;
        });
      } catch {
        // ignore
      }
      es.close();
      eventSourceRef.current = null;
      setSending(false);
    });

    // error
    es.addEventListener('error', (event: MessageEvent) => {
      let errorMessage = 'Stream error';
      try {
        if (event.data) {
          const parsed = JSON.parse(event.data);
          errorMessage =
            parsed.error || parsed.data?.error || parsed.message || errorMessage;
        }
      } catch {
        // use default
      }
      setMessages((prev) => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        const last = updated[lastIdx];
        if (last && last.role === 'assistant') {
          updated[lastIdx] = {
            ...last,
            isStreaming: false,
            error: errorMessage,
          };
        }
        return updated;
      });
      es.close();
      eventSourceRef.current = null;
      setSending(false);
    });

    // EventSource connection error
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          const last = updated[lastIdx];
          if (last && last.role === 'assistant' && last.isStreaming) {
            updated[lastIdx] = {
              ...last,
              isStreaming: false,
              error: 'Connection lost',
            };
          }
          return updated;
        });
        es.close();
        eventSourceRef.current = null;
        setSending(false);
      }
    };
  }, []);

  const send = useCallback(
    (message: string, sessionId: string, fileId?: string) => {
      eventSourceRef.current?.close();
      setSending(true);
      setConfirmRequired(false);

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        content: message,
      };

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      const url = buildStreamUrl(message, sessionId, fileId);
      const es = new EventSource(url);
      eventSourceRef.current = es;
      setupEventListeners(es);
    },
    [setupEventListeners],
  );

  const confirm = useCallback(
    (decision: 'approve' | 'skip') => {
      eventSourceRef.current?.close();
      setSending(true);
      setConfirmRequired(false);

      const sid = pendingSessionRef.current;
      if (!sid) {
        setSending(false);
        return;
      }

      const token = localStorage.getItem('mx_token') || '';
      const es = new EventSource(
        `/chat/confirm?session_id=${encodeURIComponent(sid)}&decision=${decision}&token=${encodeURIComponent(token)}`,
      );
      eventSourceRef.current = es;
      setupEventListeners(es, true);
    },
    [setupEventListeners],
  );

  const dismissConfirm = useCallback(() => {
    setConfirmRequired(false);
    setConfirmActions([]);
    setConfirmSessionId('');
  }, []);

  const clearMessages = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    setSending(false);
    setMessages([]);
    setConfirmRequired(false);
  }, []);

  return { messages, send, sending, clearMessages, confirmRequired, confirmActions, confirmSessionId, confirm, dismissConfirm };
}