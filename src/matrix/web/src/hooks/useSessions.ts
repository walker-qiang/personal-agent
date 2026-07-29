import { useState, useCallback } from 'react';
import type { SessionItem, BranchInfo } from '../types';
import { api } from '../utils/api';

const SESSION_KEY = 'mx_session';

export function useSessions() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [currentId, setCurrentIdState] = useState<string | null>(
    () => localStorage.getItem(SESSION_KEY),
  );
  const [showArchive, setShowArchive] = useState(false);

  const setCurrentId = useCallback((id: string | null) => {
    if (id) {
      localStorage.setItem(SESSION_KEY, id);
    } else {
      localStorage.removeItem(SESSION_KEY);
    }
    setCurrentIdState(id);
  }, []);

  const load = useCallback(async () => {
    const data = await api<SessionItem[]>(`/sessions?include_hidden=${showArchive}`);
    setSessions(Array.isArray(data) ? data : []);
  }, [showArchive]);

  const create = useCallback(async (): Promise<SessionItem | null> => {
    const session = await api<SessionItem>('/sessions', {
      method: 'POST',
      body: JSON.stringify({}),
    });
    setSessions((prev) => [session, ...prev]);
    setCurrentId(session.id);
    return session;
  }, [setCurrentId]);

  const remove = useCallback(async (id: string) => {
    await api(`/sessions/${id}`, { method: 'DELETE' });
    setSessions((prev) => prev.filter((s) => s.id !== id));
    setCurrentIdState((prev) => {
      if (prev === id) {
        localStorage.removeItem(SESSION_KEY);
        return null;
      }
      return prev;
    });
  }, []);

  const batchArchive = useCallback(async (ids: string[]) => {
    await api('/sessions/batch-archive', {
      method: 'POST',
      body: JSON.stringify({ session_ids: ids }),
    });
    await load();
  }, [load]);

  const batchUnarchive = useCallback(async (ids: string[]) => {
    await api('/sessions/batch-unarchive', {
      method: 'POST',
      body: JSON.stringify({ session_ids: ids }),
    });
    await load();
  }, [load]);

  const batchDelete = useCallback(async (ids: string[]) => {
    await api('/sessions/batch-delete', {
      method: 'POST',
      body: JSON.stringify({ session_ids: ids }),
    });
    setSessions((prev) => prev.filter((s) => !ids.includes(s.id)));
    setCurrentIdState((prev) => {
      if (prev && ids.includes(prev)) {
        localStorage.removeItem(SESSION_KEY);
        return null;
      }
      return prev;
    });
  }, []);

  const toggleArchive = useCallback(() => {
    setShowArchive((prev) => !prev);
  }, []);

  const branch = useCallback(async (sessionId: string, messageId?: string) => {
    const body: Record<string, string> = {};
    if (messageId) body.from_message_id = messageId;
    const data = await api<BranchInfo>(`/sessions/${sessionId}/branch`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
    return data;
  }, []);

  return {
    sessions, currentId, setCurrentId, showArchive,
    load, create, remove,
    batchArchive, batchUnarchive, batchDelete,
    toggleArchive, branch,
  };
}