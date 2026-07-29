import { useState, useCallback } from 'react';
import { api } from '../utils/api';
import type { TraceStats, TraceSession, TraceEvent } from '../types';

export function useTrace() {
  const [stats, setStats] = useState<TraceStats | null>(null);
  const [sessions, setSessions] = useState<TraceSession[]>([]);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStats = useCallback(async () => {
    try {
      const data = await api<TraceStats>('/api/trace/stats');
      setStats(data);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  const loadSessions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api<TraceSession[] | { sessions: TraceSession[] }>('/api/trace/sessions?limit=50');
      const list = Array.isArray(data) ? data : (data?.sessions || []);
      setSessions(list);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadEvents = useCallback(async (sid: string) => {
    setLoading(true);
    setError(null);
    setSessionId(sid);
    try {
      const data = await api<{ session_id: string; events: TraceEvent[] }>(`/api/trace/sessions/${sid}`);
      setEvents(data.events || []);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  const backToList = useCallback(() => {
    setSessionId(null);
    setEvents([]);
  }, []);

  return { stats, sessions, events, sessionId, loading, error, loadStats, loadSessions, loadEvents, backToList };
}