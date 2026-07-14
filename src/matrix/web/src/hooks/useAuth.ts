import { useState, useCallback, useEffect } from 'react';
import { getToken, setToken } from '../utils/api';

const TOKEN_KEY = 'mx_token';

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
  const [username, setUsername] = useState<string>('');
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
      setAuthenticated(true);
      setUsername(data.username || user);
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
      setAuthenticated(true);
      setUsername(data.username || user);
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
      setAuthenticated(false);
      setUsername('');
    }
  }, []);

  return { authenticated, username, login, register, logout, error };
}