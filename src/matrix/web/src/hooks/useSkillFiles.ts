import { useState, useCallback } from 'react';
import { api, getToken } from '../utils/api';
import type { SkillFile } from '../types';

export function useSkillFiles() {
  const [knowledgeFiles, setKnowledgeFiles] = useState<SkillFile[]>([]);
  const [scriptFiles, setScriptFiles] = useState<SkillFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadFiles = useCallback(async (skillName: string) => {
    setLoading(true);
    setError(null);
    try {
      const [kResp, sResp] = await Promise.allSettled([
        api<{ files: string[] }>(`/skills/${skillName}/knowledge`),
        api<{ files: string[] }>(`/skills/${skillName}/scripts`),
      ]);
      const kFiles = kResp.status === 'fulfilled' ? (kResp.value as { files: string[] }).files || [] : [];
      const sFiles = sResp.status === 'fulfilled' ? (sResp.value as { files: string[] }).files || [] : [];

      // Load content for each file
      const kWithContent = await Promise.all(
        kFiles.map(async (f) => {
          try {
            const data = await api<SkillFile>(`/skills/${skillName}/knowledge/${f}`);
            return data;
          } catch {
            return { filename: f, content: '' };
          }
        })
      );
      const sWithContent = await Promise.all(
        sFiles.map(async (f) => {
          try {
            const data = await api<SkillFile>(`/skills/${skillName}/scripts/${f}`);
            return data;
          } catch {
            return { filename: f, content: '' };
          }
        })
      );

      setKnowledgeFiles(kWithContent);
      setScriptFiles(sWithContent);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  const saveFile = useCallback(async (skillName: string, type: 'knowledge' | 'scripts', filename: string, content: string) => {
    const token = getToken();
    const resp = await fetch(`/skills/${skillName}/${type}/${filename}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ content }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error((body as { detail?: string }).detail || `HTTP ${resp.status}`);
    }
    await loadFiles(skillName);
  }, [loadFiles]);

  const deleteFile = useCallback(async (skillName: string, type: 'knowledge' | 'scripts', filename: string) => {
    await api(`/skills/${skillName}/${type}/${filename}`, { method: 'DELETE' });
    await loadFiles(skillName);
  }, [loadFiles]);

  const createFile = useCallback(async (skillName: string, type: 'knowledge' | 'scripts', filename: string, content: string) => {
    await saveFile(skillName, type, filename, content);
  }, [saveFile]);

  return { knowledgeFiles, scriptFiles, loading, error, loadFiles, saveFile, deleteFile, createFile };
}