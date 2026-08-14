import { useState, useCallback, useRef, useEffect } from 'react';
import type { Message, ToolCall, ToolResult, AgentStep, DebugTraceEvent } from '../types';
import { buildStreamUrl, fetchStreamTicket, api } from '../utils/api';
import { genId } from '../utils/format';

interface ConfirmAction {
  agent?: string;
  tool?: string;
  name?: string;
  args: Record<string, unknown>;
  summary?: string;
  reason?: string;
  risk?: string;
  approval_id?: string;
  operation_id?: string;
}

export interface RightPanelData {
  intent: string;
  todos: string[];
  artifacts: string[];
  refs: string[];
}

export interface UseChatReturn {
  messages: Message[];
  send: (message: string, sessionId: string, fileId?: string, debugTrace?: boolean) => void;
  stop: () => void;
  sending: boolean;
  clearMessages: () => void;
  switchSession: (sessionId: string | null) => void;
  confirmRequired: boolean;
  confirmActions: ConfirmAction[];
  confirmSessionId: string;
  confirm: (decision: 'approve' | 'skip') => void;
  dismissConfirm: () => void;
  rightPanel: RightPanelData;
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
  const [rightPanel, setRightPanel] = useState<RightPanelData>({ intent: '', todos: [], artifacts: [], refs: [] });

  // Per-session state map: sessionId -> { messages, eventSource, sending }
const sessionStatesRef = useRef<Map<string, SessionState>>(new Map());
  // Current active session ID
  const activeSessionRef = useRef<string | null>(null);
  // Pending confirm session ref
  const pendingSessionRef = useRef<string>('');
  // Token batching: accumulate SSE tokens per session and flush to React
  // state at most once per interval, to avoid one render per token.
  const pendingTokensRef = useRef<Map<string, string>>(new Map());
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

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

  // cleanup on unmount (or page refresh): close all EventSource connections.
  // Sessions that were mid-stream get a sessionStorage flag so that when their
  // history is reloaded we can mark the truncated reply as interrupted.
  useEffect(() => {
    return () => {
      if (flushTimerRef.current) clearTimeout(flushTimerRef.current);
      sessionStatesRef.current.forEach((state, sid) => {
        if (state.sending) {
          try {
            sessionStorage.setItem('mx_interrupted_' + sid, '1');
          } catch { /* storage full/blocked — non-fatal */ }
        }
        state.eventSource?.close();
      });
      sessionStatesRef.current.clear();
    };
  }, []);

  // Flush buffered tokens into message state (throttled).
  const flushTokens = useCallback(() => {
    flushTimerRef.current = null;
    const pending = pendingTokensRef.current;
    if (pending.size === 0) return;
    pending.forEach((text, sessionId) => {
      pendingTokensRef.current.delete(sessionId);
      updateSessionMessages(sessionId, (prev) => {
        const updated = [...prev];
        const lastIdx = updated.length - 1;
        const last = updated[lastIdx];
        if (last && last.role === 'assistant') {
          updated[lastIdx] = { ...last, content: last.content + text };
        }
        return updated;
      });
    });
  }, [updateSessionMessages]);

  const scheduleTokenFlush = useCallback(() => {
    if (flushTimerRef.current) return;
    flushTimerRef.current = setTimeout(flushTokens, 50);
  }, [flushTokens]);

  const setupEventListeners = useCallback((es: EventSource, sessionId: string, isResume: boolean = false) => {
    const assistantId = isResume ? genId() : '';
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
        if (!token) return;
        const prev = pendingTokensRef.current.get(sessionId) || '';
        pendingTokensRef.current.set(sessionId, prev + token);
        scheduleTokenFlush();
      } catch {
        // ignore
      }
    });

    // classify
    es.addEventListener('classify', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const data = parsed.data || parsed;
        const intent = data.intent || '';
        const intentLabel = intent === 'simple' ? '直接回答' :
          intent === 'delegate' ? '多 Agent 协同' :
          intent === 'skill' ? '技能: ' + (data.skill_name || '') : '分析中';
        const plan = data.delegation_plan || [];
        const planStr = plan.length > 0 ? plan.map((p: { agent_id: string; task: string }) => p.agent_id + ': ' + p.task).join(' | ') : '';
        setRightPanel(prev => ({
          intent,
          todos: [intentLabel],
          refs: planStr ? [planStr] : prev.refs,
          artifacts: [],
        }));
      } catch { /* ignore */ }
    });

    // tool_call
    es.addEventListener('tool_call', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const toolCall: ToolCall = parsed.data || parsed;
        setRightPanel(prev => ({
          ...prev,
          refs: [...prev.refs, toolCall.name],
        }));
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
        const raw = parsed.data || parsed;

        // Backend sends 'preview' (JSON string) and/or 'result'/'error' fields.
        // Parse preview as fallback if result/error are missing.
        let result: unknown = raw.result;
        let error: string | undefined = raw.error;

        if (result === undefined && raw.preview) {
          try {
            const parsedPreview = JSON.parse(raw.preview);
            if (parsedPreview && typeof parsedPreview === 'object' && 'error' in parsedPreview) {
              error = String((parsedPreview as Record<string, unknown>).error);
              result = null;
            } else {
              result = parsedPreview;
            }
          } catch {
            result = raw.preview;
          }
        }

        const toolResult: ToolResult = {
          id: raw.id || genId(),
          name: raw.name || '',
          result: result ?? null,
          ...(error ? { error } : {}),
          ...(raw.duration_ms ? { duration_ms: raw.duration_ms } : {}),
        };

        const preview = typeof toolResult.result === 'string'
          ? toolResult.result.substring(0, 80)
          : (toolResult.result ? '结果已返回' : (error ? error.substring(0, 80) : ''));
        if (preview) {
          setRightPanel(prev => ({
            ...prev,
            artifacts: [...prev.artifacts, toolResult.name + ': ' + preview],
          }));
        }
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

    // ephemeral Runtime debug trace — never loaded from session history
    es.addEventListener('debug_trace', (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        const raw = parsed.data || parsed;
        const trace = raw.event || raw.trace || raw;
        if (!trace || !trace.kind) return;
        const item: DebugTraceEvent = {
          sequence: Number(trace.sequence || 0),
          kind: String(trace.kind),
          timestamp: Number(trace.timestamp || 0),
          payload: (trace.payload && typeof trace.payload === 'object')
            ? trace.payload as Record<string, unknown>
            : {},
        };
        updateSessionMessages(sessionId, (prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          const last = updated[lastIdx];
          if (last && last.role === 'assistant') {
            updated[lastIdx] = {
              ...last,
              debugTrace: [...(last.debugTrace || []), item],
            };
          }
          return updated;
        });
      } catch {
        // ignore malformed optional diagnostics
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
      flushTokens();
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
          updated[lastIdx] = { ...last, isStreaming: false, progress: [] };
        }
        return updated;
      });
    });

    // done
    es.addEventListener('done', (event: MessageEvent) => {
      flushTokens();
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
              progress: [],  // Clear progress messages (matches original HTML hideStatus)
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
      flushTokens();
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
            progress: [],  // Clear progress messages on error too
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
        flushTokens();
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
  }, [updateSessionMessages, updateSessionSending, flushTokens]);

  const send = useCallback(
    (message: string, sessionId: string, fileId?: string, debugTrace = false) => {
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
        id: genId(),
        role: 'user',
        content: message,
      };

      const assistantMsg: Message = {
        id: genId(),
        role: 'assistant',
        content: '',
        isStreaming: true,
      };

      // Update session messages
      sessionState.messages = [...sessionState.messages, userMsg, assistantMsg];
      sessionState.sending = true;
      setMessages(sessionState.messages);

      // EventSource cannot send Authorization headers, so exchange the JWT
      // for a one-time ticket first, then open the stream with the ticket.
      fetchStreamTicket()
        .then((ticket) => {
          // If the user stopped or switched away while we were fetching,
          // don't open a stale stream.
          if (activeSessionRef.current !== sid || !sessionState.sending) return;
          const url = buildStreamUrl(message, sid, ticket, fileId, debugTrace);
          const es = new EventSource(url);
          sessionState.eventSource = es;
          setupEventListeners(es, sid);
        })
        .catch((err: unknown) => {
          const msg = err instanceof Error ? err.message : '无法建立连接';
          updateSessionMessages(sid, (prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = { ...last, isStreaming: false, error: msg };
            }
            return updated;
          });
          updateSessionSending(sid, false);
        });
    },
    [getOrCreateSession, setupEventListeners, updateSessionMessages, updateSessionSending],
  );

  const stop = useCallback(() => {
    const sid = activeSessionRef.current;
    if (!sid) return;
    flushTokens();
    const state = sessionStatesRef.current.get(sid);
    if (state?.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    // Mark last assistant message as not streaming, keep partial content
    updateSessionMessages(sid, (prev) => {
      const updated = [...prev];
      const lastIdx = updated.length - 1;
      const last = updated[lastIdx];
      if (last && last.role === 'assistant') {
        updated[lastIdx] = {
          ...last,
          isStreaming: false,
          progress: [],
          ...(last.content ? {} : { content: '（已停止）' }),
        };
      }
      return updated;
    });
    updateSessionSending(sid, false);
  }, [updateSessionMessages, updateSessionSending, flushTokens]);

  const switchSession = useCallback((sessionId: string | null) => {
    // Save current session's React-visible state is already in the map
    // (updates via updateSessionMessages keep the map in sync)

    if (!sessionId) {
      // New session: clear visible state but keep background sessions alive
      activeSessionRef.current = null;
      setMessages([]);
      setSending(false);
      setConfirmRequired(false);
      setRightPanel({ intent: '', todos: [], artifacts: [], refs: [] });
      return;
    }

    // Switch to target session
    activeSessionRef.current = sessionId;
    const state = getOrCreateSession(sessionId);
    setMessages(state.messages);
    setSending(state.sending);
    setConfirmRequired(false);
    setRightPanel({ intent: '', todos: [], artifacts: [], refs: [] });

    // If session has no messages in memory (e.g., page refresh), load from server
    if (state.messages.length === 0) {
      api<{ messages: Array<{ role: string; content: string; message_id?: string }> }>(
        `/sessions/${sessionId}/messages`,
      )
        .then((data) => {
          const msgs = data.messages || [];
          if (msgs.length === 0) return;
          // Only proceed if still the active session
          if (activeSessionRef.current !== sessionId) return;

          const historyMessages: Message[] = msgs.map((m) => ({
            id: m.message_id || `hist-${genId()}`,
            role: (m.role as 'user' | 'assistant' | 'system') || 'assistant',
            content: m.content || '',
            message_id: m.message_id,
          }));

          // If this session was mid-stream when the page was refreshed,
          // the last assistant reply was truncated — flag it for the UI.
          try {
            if (sessionStorage.getItem('mx_interrupted_' + sessionId)) {
              sessionStorage.removeItem('mx_interrupted_' + sessionId);
              const last = historyMessages[historyMessages.length - 1];
              if (last && last.role === 'assistant') {
                last.interrupted = true;
              }
            }
          } catch { /* storage blocked — non-fatal */ }

          // Update session state
          const st = sessionStatesRef.current.get(sessionId);
          if (st) {
            st.messages = historyMessages;
            // Only update React state if still active
            if (activeSessionRef.current === sessionId) {
              setMessages(historyMessages);
            }
          }
        })
        .catch(() => {
          // Silently fail — user will see empty chat
        });
    }
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

      fetchStreamTicket()
        .then((ticket) => {
          const es = new EventSource(
            `/chat/confirm?session_id=${encodeURIComponent(sid)}&decision=${decision}&ticket=${encodeURIComponent(ticket)}`,
          );
          if (state) state.eventSource = es;
          setupEventListeners(es, sid, true);
        })
        .catch((err: unknown) => {
          const msg = err instanceof Error ? err.message : '无法建立连接';
          updateSessionMessages(sid, (prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = { ...last, isStreaming: false, error: msg };
            }
            return updated;
          });
          updateSessionSending(sid, false);
        });
    },
    [setupEventListeners, updateSessionMessages, updateSessionSending],
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
    stop,
    sending,
    clearMessages,
    switchSession,
    confirmRequired,
    confirmActions,
    confirmSessionId,
    confirm,
    dismissConfirm,
    rightPanel,
  };
}
