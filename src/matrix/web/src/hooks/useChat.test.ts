import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useChat } from './useChat';
import { MockEventSource } from '../test/helpers/mockEventSource';

// --- Mock fetch helpers ---

function mockFetchTicket(ticket = 'test-ticket') {
  vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
    new Response(JSON.stringify({ ticket }), { status: 200 }),
  );
}

function mockFetchMessages(messages: Array<{ role: string; content: string; message_id?: string }>) {
  vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
    new Response(JSON.stringify({ messages }), { status: 200 }),
  );
}

function mockFetchError(status = 500, detail = 'Server error') {
  vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
    new Response(JSON.stringify({ detail }), { status }),
  );
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

// --- Helpers ---

/** Send a message and return the MockEventSource that was created. */
async function sendMessage(
  result: ReturnType<typeof renderHook<ReturnType<typeof useChat>>>,
  message = 'hello',
  sessionId = 's1',
  fileId?: string,
): Promise<MockEventSource> {
  mockFetchTicket();

  await act(async () => {
    result.current.send(message, sessionId, fileId);
  });
  await act(async () => {
    vi.advanceTimersByTimeAsync(1);
  });

  return MockEventSource.lastInstance();
}

/** Dispatch an SSE event wrapped in act() so React state updates flush.
 *  String payloads are JSON-encoded (matching how the backend sends plain-string tokens).
 */
async function dispatch(
  es: MockEventSource,
  type: string,
  payload: Record<string, unknown> | string,
): Promise<void> {
  const data = JSON.stringify(payload);
  await act(async () => {
    es.dispatchEvent(type, { data });
  });
}

// ============================================================================
//  Tests
// ============================================================================

describe('useChat — initial state', () => {
  it('returns empty messages and idle sending state', () => {
    const { result } = renderHook(() => useChat());

    expect(result.current.messages).toEqual([]);
    expect(result.current.sending).toBe(false);
    expect(result.current.confirmRequired).toBe(false);
    expect(result.current.confirmActions).toEqual([]);
    expect(result.current.confirmSessionId).toBe('');
    expect(result.current.rightPanel).toEqual({
      intent: '',
      todos: [],
      artifacts: [],
      refs: [],
    });
  });
});

describe('useChat — send()', () => {
  it('creates user and assistant messages and sets sending=true', async () => {
    const { result } = renderHook(() => useChat());

    await sendMessage(result, 'hello world', 's1');

    expect(result.current.sending).toBe(true);
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].role).toBe('user');
    expect(result.current.messages[0].content).toBe('hello world');
    expect(result.current.messages[1].role).toBe('assistant');
    expect(result.current.messages[1].content).toBe('');
    expect(result.current.messages[1].isStreaming).toBe(true);
  });

  it('opens EventSource with correct URL', async () => {
    const { result } = renderHook(() => useChat());

    await sendMessage(result, 'test message', 'session-abc');

    expect(MockEventSource.lastUrl).toContain('/chat/stream?');
    expect(MockEventSource.lastUrl).toContain('message=test+message');
    expect(MockEventSource.lastUrl).toContain('session_id=session-abc');
    expect(MockEventSource.lastUrl).toContain('ticket=test-ticket');
  });

  it('includes file_id in URL when provided', async () => {
    const { result } = renderHook(() => useChat());

    await sendMessage(result, 'msg', 's1', 'file-001');

    expect(MockEventSource.lastUrl).toContain('file_id=file-001');
  });

  it('does nothing when sessionId is empty', async () => {
    const { result } = renderHook(() => useChat());

    await act(async () => {
      result.current.send('hello', '');
    });

    expect(result.current.messages).toHaveLength(0);
    expect(result.current.sending).toBe(false);
  });

  it('sets error when fetchStreamTicket fails', async () => {
    const { result } = renderHook(() => useChat());
    mockFetchError(500, 'Ticket service down');

    await act(async () => {
      result.current.send('hello', 's1');
    });
    await act(async () => {
      vi.advanceTimersByTimeAsync(1);
    });

    expect(result.current.sending).toBe(false);
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[1].isStreaming).toBe(false);
    expect(result.current.messages[1].error).toBe('Ticket service down');
  });
});

describe('useChat — token event', () => {
  it('accumulates tokens and flushes to message content', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hello', 's1');

    await dispatch(es, 'token', { content: 'Hello' });
    await dispatch(es, 'token', { content: ' world' });

    await act(async () => {
      vi.advanceTimersByTimeAsync(60);
    });

    expect(result.current.messages[1].content).toBe('Hello world');
  });

  it('handles plain string token format', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'token', 'plain text token');

    await act(async () => {
      vi.advanceTimersByTimeAsync(60);
    });

    expect(result.current.messages[1].content).toBe('plain text token');
  });

  it('ignores empty tokens', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'token', '');
    await dispatch(es, 'token', { content: '' });

    await act(async () => {
      vi.advanceTimersByTimeAsync(60);
    });

    expect(result.current.messages[1].content).toBe('');
  });
});

describe('useChat — classify event', () => {
  it('updates rightPanel for simple intent', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'classify', { data: { intent: 'simple' } });

    expect(result.current.rightPanel.intent).toBe('simple');
    expect(result.current.rightPanel.todos).toEqual(['直接回答']);
  });

  it('updates rightPanel with skill name for skill intent', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'classify', { data: { intent: 'skill', skill_name: 'portfolio-review' } });

    expect(result.current.rightPanel.todos).toEqual(['技能: portfolio-review']);
  });

  it('updates rightPanel with delegation plan', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'classify', {
      data: {
        intent: 'delegate',
        delegation_plan: [
          { agent_id: 'analyst', task: 'research' },
          { agent_id: 'coder', task: 'implement' },
        ],
      },
    });

    expect(result.current.rightPanel.intent).toBe('delegate');
    expect(result.current.rightPanel.todos).toEqual(['多 Agent 协同']);
    expect(result.current.rightPanel.refs).toEqual(['analyst: research | coder: implement']);
  });
});

describe('useChat — tool_call event', () => {
  it('adds toolCall to assistant message', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    const toolCall = { id: 'tc1', name: 'finance_query', arguments: { code: '600585' } };
    await dispatch(es, 'tool_call', { data: toolCall });

    expect(result.current.messages[1].toolCalls).toHaveLength(1);
    expect(result.current.messages[1].toolCalls![0]).toEqual(toolCall);
    expect(result.current.rightPanel.refs).toContain('finance_query');
  });

  it('appends multiple toolCalls', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'tool_call', { data: { id: 'tc1', name: 'tool_a', arguments: {} } });
    await dispatch(es, 'tool_call', { data: { id: 'tc2', name: 'tool_b', arguments: {} } });

    expect(result.current.messages[1].toolCalls).toHaveLength(2);
  });
});

describe('useChat — tool_result event', () => {
  it('adds toolResult to assistant message', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'tool_result', {
      data: { id: 'tr1', name: 'finance_query', result: { price: 3400.12 }, duration_ms: 150 },
    });

    expect(result.current.messages[1].toolResults).toHaveLength(1);
    expect(result.current.messages[1].toolResults![0].name).toBe('finance_query');
    expect(result.current.messages[1].toolResults![0].duration_ms).toBe(150);
  });

  it('parses preview as fallback when result is missing', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'tool_result', {
      data: { id: 'tr1', name: 'search', preview: JSON.stringify({ found: 3 }) },
    });

    expect(result.current.messages[1].toolResults![0].result).toEqual({ found: 3 });
  });

  it('extracts error from preview', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'tool_result', {
      data: { id: 'tr1', name: 'search', preview: JSON.stringify({ error: 'Rate limit exceeded' }) },
    });

    expect(result.current.messages[1].toolResults![0].error).toBe('Rate limit exceeded');
    expect(result.current.messages[1].toolResults![0].result).toBeNull();
  });

  it('uses preview as raw string when not valid JSON', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'tool_result', {
      data: { id: 'tr1', name: 'search', preview: 'raw text result' },
    });

    expect(result.current.messages[1].toolResults![0].result).toBe('raw text result');
  });
});

describe('useChat — agent_result event', () => {
  it('adds agentChain step to assistant message', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    const step = { agent: 'analyst', task: 'research stock', status: 'done' as const, result: 'found' };
    await dispatch(es, 'agent_result', { data: step });

    expect(result.current.messages[1].agentChain).toHaveLength(1);
    expect(result.current.messages[1].agentChain![0]).toEqual(step);
  });
});

describe('useChat — thinking event', () => {
  it('adds thinking content to assistant message', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'thinking', { content: 'Analyzing the query...' });

    expect(result.current.messages[1].thinking).toEqual(['Analyzing the query...']);
  });

  it('appends multiple thinking entries', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'thinking', { content: 'Step 1' });
    await dispatch(es, 'thinking', { content: 'Step 2' });

    expect(result.current.messages[1].thinking).toEqual(['Step 1', 'Step 2']);
  });
});

describe('useChat — progress event', () => {
  it('adds progress message to assistant message', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'progress', { message: 'Loading data...' });

    expect(result.current.messages[1].progress).toEqual(['Loading data...']);
  });
});

describe('useChat — done event', () => {
  it('marks assistant message as done and closes EventSource', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'token', { content: 'final answer' });
    await dispatch(es, 'done', { duration: '2.5s' });

    expect(result.current.messages[1].isStreaming).toBe(false);
    expect(result.current.messages[1].content).toBe('final answer');
    expect(result.current.messages[1].duration).toBe('2.5s');
    expect(result.current.messages[1].progress).toEqual([]);
    expect(result.current.sending).toBe(false);
    expect(es.isClosed()).toBe(true);
  });

  it('works without duration field', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'done', {});

    expect(result.current.messages[1].isStreaming).toBe(false);
    expect(result.current.messages[1].duration).toBeUndefined();
  });
});

describe('useChat — error event', () => {
  it('sets error on assistant message and closes EventSource', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'error', { error: 'Internal server error' });

    expect(result.current.messages[1].isStreaming).toBe(false);
    expect(result.current.messages[1].error).toBe('Internal server error');
    expect(result.current.messages[1].progress).toEqual([]);
    expect(result.current.sending).toBe(false);
    expect(es.isClosed()).toBe(true);
  });

  it('uses default error message when no data', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await act(async () => {
      es.dispatchEvent('error', { data: '' });
    });

    expect(result.current.messages[1].error).toBe('Stream error');
  });

  it('handles connection loss via onerror', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await act(async () => {
      es.simulateConnectionError();
    });

    expect(result.current.messages[1].isStreaming).toBe(false);
    expect(result.current.messages[1].error).toBe('Connection lost');
    expect(result.current.sending).toBe(false);
  });
});

describe('useChat — confirm_required event', () => {
  it('sets confirmRequired with actions and session ID', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    const actions = [
      { tool: 'delete_file', args: { path: '/test' }, reason: 'Needs confirmation' },
    ];

    await dispatch(es, 'confirm_required', { actions, session_id: 's1' });

    expect(result.current.confirmRequired).toBe(true);
    expect(result.current.confirmActions).toEqual(actions);
    expect(result.current.confirmSessionId).toBe('s1');
    expect(result.current.sending).toBe(false);
    expect(es.isClosed()).toBe(true);
  });

  it('marks last assistant message as not streaming', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'confirm_required', { actions: [], session_id: 's1' });

    expect(result.current.messages[1].isStreaming).toBe(false);
  });
});

describe('useChat — stop()', () => {
  it('closes EventSource and marks message as stopped', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hello', 's1');

    await dispatch(es, 'token', { content: 'partial' });

    await act(async () => {
      result.current.stop();
    });

    expect(result.current.messages[1].isStreaming).toBe(false);
    expect(result.current.sending).toBe(false);
    expect(es.isClosed()).toBe(true);
  });

  it('shows (已停止) when no content was received', async () => {
    const { result } = renderHook(() => useChat());
    await sendMessage(result, 'hello', 's1');

    await act(async () => {
      result.current.stop();
    });

    expect(result.current.messages[1].content).toBe('（已停止）');
  });

  it('keeps partial content when stopped mid-stream', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hello', 's1');

    await dispatch(es, 'token', { content: 'partial content' });

    await act(async () => {
      result.current.stop();
    });

    expect(result.current.messages[1].content).toBe('partial content');
  });
});

describe('useChat — confirm()', () => {
  it('fetches new ticket and opens confirm EventSource on approve', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'confirm_required', {
      actions: [{ tool: 'test_tool', args: {}, reason: 'test' }],
      session_id: 's1',
    });

    expect(result.current.confirmRequired).toBe(true);

    mockFetchTicket('confirm-ticket');
    await act(async () => {
      result.current.confirm('approve');
    });
    await act(async () => {
      vi.advanceTimersByTimeAsync(1);
    });

    expect(result.current.confirmRequired).toBe(false);
    expect(result.current.sending).toBe(true);
    expect(MockEventSource.lastUrl).toContain('/chat/confirm');
    expect(MockEventSource.lastUrl).toContain('session_id=s1');
    expect(MockEventSource.lastUrl).toContain('decision=approve');
    expect(MockEventSource.lastUrl).toContain('ticket=confirm-ticket');
  });

  it('sets error when ticket fetch fails', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'confirm_required', {
      actions: [{ tool: 'test_tool', args: {}, reason: 'test' }],
      session_id: 's1',
    });

    mockFetchError(500, 'Ticket error');
    await act(async () => {
      result.current.confirm('approve');
    });
    await act(async () => {
      vi.advanceTimersByTimeAsync(1);
    });

    expect(result.current.sending).toBe(false);
    const lastMsg = result.current.messages[result.current.messages.length - 1];
    expect(lastMsg.error).toBe('Ticket error');
  });
});

describe('useChat — dismissConfirm()', () => {
  it('clears confirm state', async () => {
    const { result } = renderHook(() => useChat());
    const es = await sendMessage(result, 'hi', 's1');

    await dispatch(es, 'confirm_required', {
      actions: [{ tool: 'test', args: {}, reason: '' }],
      session_id: 's1',
    });

    expect(result.current.confirmRequired).toBe(true);

    await act(async () => {
      result.current.dismissConfirm();
    });

    expect(result.current.confirmRequired).toBe(false);
    expect(result.current.confirmActions).toEqual([]);
    expect(result.current.confirmSessionId).toBe('');
  });
});

describe('useChat — switchSession()', () => {
  it('clears visible state when switching to null (new session)', async () => {
    const { result } = renderHook(() => useChat());

    await sendMessage(result, 'hello', 's1');

    expect(result.current.messages).toHaveLength(2);

    await act(async () => {
      result.current.switchSession(null);
    });

    expect(result.current.messages).toEqual([]);
    expect(result.current.sending).toBe(false);
    expect(result.current.confirmRequired).toBe(false);
  });

  it('loads messages from API when session has no in-memory messages', async () => {
    const { result } = renderHook(() => useChat());

    const serverMessages = [
      { role: 'user', content: 'Hello', message_id: 'm1' },
      { role: 'assistant', content: 'Hi there', message_id: 'm2' },
    ];
    mockFetchMessages(serverMessages);

    await act(async () => {
      result.current.switchSession('session-from-server');
    });
    await act(async () => {
      vi.advanceTimersByTimeAsync(1);
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].content).toBe('Hello');
    expect(result.current.messages[0].message_id).toBe('m1');
    expect(result.current.messages[1].content).toBe('Hi there');
    expect(result.current.messages[1].role).toBe('assistant');
  });

  it('marks last assistant message as interrupted when sessionStorage flag is set', async () => {
    const { result } = renderHook(() => useChat());

    sessionStorage.setItem('mx_interrupted_s2', '1');

    const serverMessages = [
      { role: 'user', content: 'Q', message_id: 'm1' },
      { role: 'assistant', content: 'partial answer', message_id: 'm2' },
    ];
    mockFetchMessages(serverMessages);

    await act(async () => {
      result.current.switchSession('s2');
    });
    await act(async () => {
      vi.advanceTimersByTimeAsync(1);
    });

    expect(result.current.messages[1].interrupted).toBe(true);
    expect(sessionStorage.getItem('mx_interrupted_s2')).toBeNull();
  });

  it('switches to in-memory session without API call', async () => {
    const { result } = renderHook(() => useChat());

    await sendMessage(result, 'hello', 'session-1');
    const es1 = MockEventSource.lastInstance();
    await dispatch(es1, 'done', {});

    await act(async () => {
      result.current.switchSession(null);
    });

    await act(async () => {
      result.current.switchSession('session-1');
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].content).toBe('hello');
  });
});

describe('useChat — clearMessages()', () => {
  it('clears visible messages and sending state', async () => {
    const { result } = renderHook(() => useChat());

    await sendMessage(result, 'hello', 's1');

    expect(result.current.messages).toHaveLength(2);

    await act(async () => {
      result.current.clearMessages();
    });

    expect(result.current.messages).toEqual([]);
    expect(result.current.sending).toBe(false);
    expect(result.current.confirmRequired).toBe(false);
  });
});

describe('useChat — unmount cleanup', () => {
  it('closes all EventSource connections on unmount', async () => {
    const { result, unmount } = renderHook(() => useChat());

    const es = await sendMessage(result, 'hello', 's1');

    expect(es.isClosed()).toBe(false);

    unmount();

    expect(es.isClosed()).toBe(true);
  });
});

describe('useChat — multiple sessions', () => {
  it('maintains separate message lists per session', async () => {
    const { result } = renderHook(() => useChat());

    // Session 1
    await sendMessage(result, 'msg1', 'session-1');
    const es1 = MockEventSource.lastInstance();
    await dispatch(es1, 'done', {});

    // Switch to new session and send
    await act(async () => {
      result.current.switchSession(null);
    });
    await sendMessage(result, 'msg2', 'session-2');

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].content).toBe('msg2');

    // Switch back to session-1
    await act(async () => {
      result.current.switchSession('session-1');
    });

    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0].content).toBe('msg1');
  });
});
