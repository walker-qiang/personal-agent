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
  switchSession: (sessionId: string | null) => void;
  confirmRequired: boolean;
  confirmActions: ConfirmAction[];
  confirmSessionId: string;
  confirm: (decision: 'approve' | 'skip') => void;
  dismissConfirm: () => void;
}

// Per-session state: messages + EventSource
interface SessionState {
  messages: Message[];
  eventSource: EventSource | null;
  sending: boolean;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sending, setSending] = useState<boolean>(false);
  const [confirmRequired, setConfirmRequired] = useState(false);
  const [confirmActions, setConfirmActions] = useState<ConfirmAction[]>([]);
  const [confirmSessionId, setConfirmSessionId] = useState('');

  // Per-session state map: sessionId -> { messages, eventSource, sending }
  const sessionStatesRef = useRef<Map<string, SessionState>>(new Map());
  // Current active session ID
  const activeSessionRef = useRef<string | null>(null);
  // Pending confirm session ref
  const pendingSessionRef = useRef<string>('');

  // Helper: update messages for a specific session (or current if no sessionId)
  const updateSessionMessages = useCallback((
    sessionId: string | null,
    updater: (prev: Message[]) => Message[],
  ) => {
    const sid = sessionId ?? activeSessionRef.current;
    if (!sid) {
      setMessages(prev => updater(prev));
      return;
    }
    const state = sessionStatesRef.current.get(sid);
    if (state) {
      state.messages = updater(state.messages);
      // Only update React state if this is the active session
      if (sid === activeSessionRef.current) {
        setMessages(state.messages);
      }
    } else {
      // Fallback: update React state directly
      setMessages(prev => updater(prev));
    }
  }, []);

  // Helper: update sending state for a session
  const updateSessionSending = useCallback((sessionId: string, value: boolean) => {
    const state = sessionStatesRef.current.get(sessionId);
    if (state) {
      state.sending = value;
      if (sessionId === activeSessionRef.current) {
        setSending(value);
      }
    }
  }, []);

  // Helper: get or create session state
  const getOrCreateSession = useCallback((sessionId: string): SessionState => {
    let state = sessionStatesRef.current.get(sessionId);
    if (!state) {
      state = { messages: [], eventSource: null, sending: false };
      sessionStatesRef.current.set(sessionId, state);
    }
    return state;
  }, []);

  // cleanup on unmount: close all EventSource connections
  useEffect(() => {
    return () => {
      sessionStatesRef.current.forEach(state => {
        state.eventSource?.close();
      });
      sessionStatesRef.current.clear();
    };
  }, []);

  const setupEventListeners = useCallback((es: EventSource, sessionId: string, isResume: boolean = false) => {
    const assistantId = isResume ? crypto.randomUUID() : '';
    if (isResume) {
      updateSessionMessages(sessionId, (prev) => [...prev, {
        id: assistantId,
        role: 'assistant',
        content: '',
        isStreaming: true,
      }]);
    }

    // token
    es.addEventListener('token', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const token: string =
          typeof parsed === 'string'
            ? parsed
            : parsed.content || parsed.data?.content || '';
        updateSessionMessages(sessionId, (prev) => {
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
        updateSessionMessages(sessionId, (prev) => {
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
        updateSessionMessages(sessionId, (prev) => {
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
        updateSessionMessages(sessionId, (prev) => {
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

    // thinking
    es.addEventListener('thinking', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const content: string = parsed.content || parsed.data?.content || '';
        if (content) {
          updateSessionMessages(sessionId, (prev) => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            const last = updated[lastIdx];
            if (last && last.role === 'assistant') {
              updated[lastIdx] = {
                ...last,
                thinking: [...(last.thinking || []), content],
              };
            }
            return updated;
          });
        }
      } catch {
        // ignore
      }
    });

    // progress
    es.addEventListener('progress', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const msg: string = parsed.message || parsed.data?.message || '';
        if (msg) {
          updateSessionMessages(sessionId, (prev) => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            const last = updated[lastIdx];
            if (last && last.role === 'assistant') {
              updated[lastIdx] = {
                ...last,
                progress: [...(last.progress || []), msg],
              };
            }
            return updated;
          });
        }
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
      const state = sessionStatesRef.current.get(sessionId);
      if (state) state.eventSource = null;
      updateSessionSending(sessionId, false);
      // Mark last assistant message as not streaming
      updateSessionMessages(sessionId, (prev) => {
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
        updateSessionMessages(sessionId, (prev) => {
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
      const state = sessionStatesRef.current.get(sessionId);
      if (state) state.eventSource = null;
      updateSessionSending(sessionId, false);
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
      updateSessionMessages(sessionId, (prev) => {
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
      const state = sessionStatesRef.current.get(sessionId);
      if (state) state.eventSource = null;
      updateSessionSending(sessionId, false);
    });

    // EventSource connection error
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        updateSessionMessages(sessionId, (prev) => {
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
        const state = sessionStatesRef.current.get(sessionId);
        if (state) state.eventSource = null;
        updateSessionSending(sessionId, false);
      }
    };
  }, [updateSessionMessages, updateSessionSending]);

  const send = useCallback(
    (message: string, sessionId: string, fileId?: string) => {
      // Close only the current session's EventSource (not others)
      const currentSession = activeSessionRef.current;
      if (currentSession) {
        const state = sessionStatesRef.current.get(currentSession);
        if (state?.eventSource) {
          state.eventSource.close();
          state.eventSource = null;
        }
      }

      const sid = sessionId || currentSession || '';
      if (!sid) return;

      // Ensure session state exists
      const sessionState = getOrCreateSession(sid);
      activeSessionRef.current = sid;

      // Update React state for the active session
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

      // Update session messages
      sessionState.messages = [...sessionState.messages, userMsg, assistantMsg];
      sessionState.sending = true;
      setMessages(sessionState.messages);

      const url = buildStreamUrl(message, sid, fileId);
      const es = new EventSource(url);
      sessionState.eventSource = es;
      setupEventListeners(es, sid);
    },
    [getOrCreateSession, setupEventListeners],
  );

  const switchSession = useCallback((sessionId: string | null) => {
    // Save current session's React-visible state is already in the map
    // (updates via updateSessionMessages keep the map in sync)

    if (!sessionId) {
      // New session: clear visible state but keep background sessions alive
      activeSessionRef.current = null;
      setMessages([]);
      setSending(false);
      setConfirmRequired(false);
      return;
    }

    // Switch to target session
    activeSessionRef.current = sessionId;
    const state = getOrCreateSession(sessionId);
    setMessages(state.messages);
    setSending(state.sending);
    setConfirmRequired(false);
  }, [getOrCreateSession]);

  const confirm = useCallback(
    (decision: 'approve' | 'skip') => {
      const sid = pendingSessionRef.current;
      if (!sid) {
        setSending(false);
        return;
      }

      // Close current EventSource for that session
      const state = sessionStatesRef.current.get(sid);
      if (state?.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
      }

      setSending(true);
      setConfirmRequired(false);

      const token = localStorage.getItem('mx_token') || '';
      const es = new EventSource(
        `/chat/confirm?session_id=${encodeURIComponent(sid)}&decision=${decision}&token=${encodeURIComponent(token)}`,
      );
      if (state) state.eventSource = es;
      setupEventListeners(es, sid, true);
    },
    [setupEventListeners],
  );

  const dismissConfirm = useCallback(() => {
    setConfirmRequired(false);
    setConfirmActions([]);
    setConfirmSessionId('');
  }, []);

  const clearMessages = useCallback(() => {
    // Only clear the current visible state, don't touch background sessions
    activeSessionRef.current = null;
    setMessages([]);
    setSending(false);
    setConfirmRequired(false);
  }, []);

  return {
    messages,
    send,
    sending,
    clearMessages,
    switchSession,
    confirmRequired,
    confirmActions,
    confirmSessionId,
    confirm,
    dismissConfirm,
  };
}
