import { useState, useCallback } from 'react';
import type { SessionItem } from '../types';
import { getToken } from '../utils/api';

const SESSION_KEY = 'mx_session';

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

export interface UseSessionsReturn {
  sessions: SessionItem[];
  currentId: string | null;
  setCurrentId: (id: string | null) => void;
  load: () => Promise<void>;
  create: () => Promise<SessionItem | null>;
  remove: (id: string) => Promise<void>;
}

export function useSessions(): UseSessionsReturn {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [currentId, setCurrentIdState] = useState<string | null>(
    () => localStorage.getItem(SESSION_KEY),
  );

  const setCurrentId = useCallback((id: string | null) => {
    if (id) {
      localStorage.setItem(SESSION_KEY, id);
    } else {
      localStorage.removeItem(SESSION_KEY);
    }
    setCurrentIdState(id);
  }, []);

  const load = useCallback(async () => {
    const res = await fetch('/sessions', {
      headers: getAuthHeaders(),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(
        (body as { detail?: string }).detail || 'Failed to load sessions',
      );
    }
    const data: SessionItem[] = await res.json();
    setSessions(data);
  }, []);

  const create = useCallback(async (): Promise<SessionItem | null> => {
    const res = await fetch('/sessions', {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({}),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(
        (body as { detail?: string }).detail || 'Failed to create session',
      );
    }
    const session: SessionItem = await res.json();
    setSessions((prev) => [session, ...prev]);
    setCurrentId(session.id);
    return session;
  }, [setCurrentId]);

  const remove = useCallback(
    async (id: string) => {
      const res = await fetch(`/sessions/${id}`, {
        method: 'DELETE',
        headers: getAuthHeaders(),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
          (body as { detail?: string }).detail || 'Failed to delete session',
        );
      }
      setSessions((prev) => prev.filter((s) => s.id !== id));
      // 如果删除的是当前会话，清除当前 ID
      setCurrentIdState((prev) => {
        if (prev === id) {
          localStorage.removeItem(SESSION_KEY);
          return null;
        }
        return prev;
      });
    },
    [],
  );

  return { sessions, currentId, setCurrentId, load, create, remove };
}