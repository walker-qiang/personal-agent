import { useState, useCallback, useEffect } from 'react';
import { getToken, setToken } from '../utils/api';

const TOKEN_KEY = 'mx_token';
const USERNAME_KEY = 'mx_username';

/** Decode JWT payload to extract username (fallback when mx_username is not persisted). */
function decodeUsernameFromToken(token: string): string {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return '';
    // base64url → base64 → JSON
    const b64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
    const json = JSON.parse(atob(b64));
    return json.sub || '';
  } catch {
    return '';
  }
}

function getAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  const token = getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export interface UseAuthReturn {
  authenticated: boolean;
  username: string;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  error: string | null;
}

export function useAuth(): UseAuthReturn {
  const [authenticated, setAuthenticated] = useState<boolean>(
    () => !!localStorage.getItem(TOKEN_KEY),
  );
  const [username, setUsername] = useState<string>(() => {
    // 1. Try mx_username from localStorage (set by login/register)
    const stored = localStorage.getItem(USERNAME_KEY);
    if (stored) return stored;
    // 2. Fallback: decode from JWT token (for users who logged in before this fix)
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) {
      const decoded = decodeUsernameFromToken(token);
      if (decoded) {
        localStorage.setItem(USERNAME_KEY, decoded);
        return decoded;
      }
    }
    return '';
  });
  const [error, setError] = useState<string | null>(null);

  // 监听 auth:expired 自定义事件
  useEffect(() => {
    const handler = () => {
      setAuthenticated(false);
      setUsername('');
    };
    window.addEventListener('auth:expired', handler);
    return () => window.removeEventListener('auth:expired', handler);
  }, []);

  const login = useCallback(async (user: string, password: string) => {
    setError(null);
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ username: user, password }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
          (body as { detail?: string }).detail || 'Login failed',
        );
      }
      const data = await res.json();
      if (data.token) {
        setToken(data.token);
      }
      const resolvedUsername = data.username || user;
      localStorage.setItem(USERNAME_KEY, resolvedUsername);
      setAuthenticated(true);
      setUsername(resolvedUsername);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Login failed';
      setError(message);
      setAuthenticated(false);
      throw e;
    }
  }, []);

  const register = useCallback(async (user: string, password: string) => {
    setError(null);
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ username: user, password }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
          (body as { detail?: string }).detail || 'Register failed',
        );
      }
      const data = await res.json();
      if (data.token) {
        setToken(data.token);
      }
      const resolvedUsername = data.username || user;
      localStorage.setItem(USERNAME_KEY, resolvedUsername);
      setAuthenticated(true);
      setUsername(resolvedUsername);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Register failed';
      setError(message);
      setAuthenticated(false);
      throw e;
    }
  }, []);

  const logout = useCallback(async () => {
    setError(null);
    try {
      await fetch('/api/auth/logout', {
        method: 'POST',
        headers: getAuthHeaders(),
      });
    } catch {
      // 即使 logout 请求失败也清除本地状态
    } finally {
      setToken(null);
      localStorage.removeItem(USERNAME_KEY);
      setAuthenticated(false);
      setUsername('');
    }
  }, []);

  return { authenticated, username, login, register, logout, error };
}