const TOKEN_KEY = 'mx_token';
const SESSION_KEY = 'mx_session';

let _token: string | null = localStorage.getItem(TOKEN_KEY);

export function getToken(): string | null {
  return _token;
}

export function setToken(token: string | null): void {
  _token = token;
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export function getSessionId(): string | null {
  return localStorage.getItem(SESSION_KEY);
}

export function setSessionId(sid: string | null): void {
  if (sid) {
    localStorage.setItem(SESSION_KEY, sid);
  } else {
    localStorage.removeItem(SESSION_KEY);
  }
}

export async function api<T = unknown>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };
  const token = getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const res = await fetch(path, { ...options, headers });
  if (res.status === 401) {
    setToken(null);
    window.dispatchEvent(new CustomEvent('auth:expired'));
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export function buildStreamUrl(
  message: string,
  sessionId: string,
  fileId?: string,
): string {
  const params = new URLSearchParams({
    message,
    session_id: sessionId,
    token: getToken() || '',
  });
  if (fileId) {
    params.set('file_id', fileId);
  }
  return `/chat/stream?${params.toString()}`;
}