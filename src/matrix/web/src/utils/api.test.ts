import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  getToken,
  setToken,
  getSessionId,
  setSessionId,
  api,
  buildStreamUrl,
  fetchStreamTicket,
} from './api';

// Reset module-level token state between tests
beforeEach(async () => {
  localStorage.clear();
  // Re-import to reset module-level _token
  vi.resetModules();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('token management', () => {
  it('getToken returns null when nothing stored', () => {
    expect(getToken()).toBeNull();
  });

  it('setToken stores in localStorage', () => {
    setToken('my-jwt');
    expect(getToken()).toBe('my-jwt');
    expect(localStorage.getItem('mx_token')).toBe('my-jwt');
  });

  it('setToken(null) removes from localStorage', () => {
    setToken('my-jwt');
    setToken(null);
    expect(getToken()).toBeNull();
    expect(localStorage.getItem('mx_token')).toBeNull();
  });
});

describe('session ID management', () => {
  it('getSessionId returns null when nothing stored', () => {
    expect(getSessionId()).toBeNull();
  });

  it('setSessionId stores in localStorage', () => {
    setSessionId('session-123');
    expect(getSessionId()).toBe('session-123');
  });

  it('setSessionId(null) removes from localStorage', () => {
    setSessionId('session-123');
    setSessionId(null);
    expect(getSessionId()).toBeNull();
  });
});

describe('buildStreamUrl', () => {
  it('builds correct URL with required params', () => {
    const url = buildStreamUrl('hello', 's1', 'ticket-abc');
    expect(url).toContain('/chat/stream?');
    expect(url).toContain('message=hello');
    expect(url).toContain('session_id=s1');
    expect(url).toContain('ticket=ticket-abc');
  });

  it('includes file_id when provided', () => {
    const url = buildStreamUrl('hello', 's1', 't', 'file-001');
    expect(url).toContain('file_id=file-001');
  });

  it('omits file_id when not provided', () => {
    const url = buildStreamUrl('hello', 's1', 't');
    expect(url).not.toContain('file_id');
  });

  it('URL-encodes special characters in message', () => {
    const url = buildStreamUrl('hello & world', 's1', 't');
    expect(url).toContain('message=hello+%26+world');
  });
});

describe('api()', () => {
  it('sends GET request and returns parsed JSON', async () => {
    const mockJson = { data: 'test' };
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify(mockJson), { status: 200 }),
    );

    const result = await api('/test/endpoint');
    expect(result).toEqual(mockJson);
  });

  it('includes Authorization header when token is set', async () => {
    setToken('my-jwt');
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response('{"ok":true}', { status: 200 }),
    );

    await api('/test');

    const request = fetchSpy.mock.calls[0][1];
    expect(request?.headers).toMatchObject({
      Authorization: 'Bearer my-jwt',
      'Content-Type': 'application/json',
    });
  });

  it('omits Authorization header when no token', async () => {
    setToken(null); // Ensure module-level _token is cleared (previous test may have set it)
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response('{"ok":true}', { status: 200 }),
    );

    await api('/test');

    const request = fetchSpy.mock.calls[0][1];
    expect(request?.headers).not.toHaveProperty('Authorization');
  });

  it('throws Error with detail on 4xx/5xx', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: 'Bad request' }), { status: 400 }),
    );

    await expect(api('/test')).rejects.toThrow('Bad request');
  });

  it('uses generic error message when no detail', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response('', { status: 500 }),
    );

    await expect(api('/test')).rejects.toThrow('HTTP 500');
  });

  it('dispatches auth:expired event on 401 and clears token', async () => {
    setToken('expired-token');
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response('{"detail":"Unauthorized"}', { status: 401 }),
    );

    const eventSpy = vi.fn();
    window.addEventListener('auth:expired', eventSpy);

    await expect(api('/test')).rejects.toThrow('Unauthorized');

    expect(eventSpy).toHaveBeenCalledTimes(1);
    expect(getToken()).toBeNull();

    window.removeEventListener('auth:expired', eventSpy);
  });

  it('supports custom method and body', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response('{"ok":true}', { status: 200 }),
    );

    await api('/submit', {
      method: 'POST',
      body: JSON.stringify({ key: 'value' }),
    });

    const request = fetchSpy.mock.calls[0][1];
    expect(request?.method).toBe('POST');
    expect(request?.body).toBe(JSON.stringify({ key: 'value' }));
  });
});

describe('fetchStreamTicket', () => {
  it('POSTs to /api/auth/stream-ticket and returns ticket string', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
      new Response(JSON.stringify({ ticket: 'one-time-ticket' }), { status: 200 }),
    );

    const ticket = await fetchStreamTicket();
    expect(ticket).toBe('one-time-ticket');

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe('/api/auth/stream-ticket');
    expect(init?.method).toBe('POST');
  });
});
