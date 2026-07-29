import { useState, useCallback } from 'react';
import { api } from '../utils/api';
import type { McpServer } from '../types';

export function useMcp() {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api<{ servers: McpServer[] }>('/mcp/servers');
      setServers(data.servers || []);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  const create = useCallback(async (server: Omit<McpServer, 'enabled' | 'connected' | 'tool_count' | 'tools'>) => {
    await api('/mcp/servers', {
      method: 'POST',
      body: JSON.stringify(server),
    });
    await load();
  }, [load]);

  const update = useCallback(async (name: string, server: Partial<McpServer>) => {
    await api(`/mcp/servers/${name}`, {
      method: 'PUT',
      body: JSON.stringify(server),
    });
    await load();
  }, [load]);

  const remove = useCallback(async (name: string) => {
    await api(`/mcp/servers/${name}`, { method: 'DELETE' });
    setServers(prev => prev.filter(s => s.name !== name));
  }, []);

  const toggle = useCallback(async (name: string) => {
    const data = await api<{ ok: boolean; name: string }>(`/mcp/servers/${name}/toggle`, { method: 'POST' });
    if (data.ok) {
      await load();
    }
  }, [load]);

  return { servers, loading, error, load, create, update, remove, toggle };
}