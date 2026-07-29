import { useState, useCallback, useRef } from 'react';
import { api, getToken } from '../utils/api';
import type { SkillFile } from '../types';

export function useSkillFiles() {
  const [knowledgeFiles, setKnowledgeFiles] = useState<SkillFile[]>([]);
  const [scriptFiles, setScriptFiles] = useState<SkillFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Store current file lists so saveFile/deleteFile can reload with them
  const kFileListRef = useRef<string[]>([]);
  const sFileListRef = useRef<string[]>([]);

  const _doLoad = useCallback(async (skillName: string, kFileList: string[], sFileList: string[]) => {
    // Load content for each file
    const kWithContent = await Promise.all(
      kFileList.map(async (f) => {
        try {
          const data = await api<SkillFile>(`/skills/${skillName}/knowledge/${f}`);
          return data;
        } catch {
          return { filename: f, content: '' };
        }
      })
    );
    const sWithContent = await Promise.all(
      sFileList.map(async (f) => {
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
  }, []);

  const loadFiles = useCallback(async (skillName: string, kFileList?: string[], sFileList?: string[]) => {
    setLoading(true);
    setError(null);
    try {
      const [kResp] = await Promise.allSettled([
        api<{ knowledge?: string[] }>(`/skills/${skillName}/knowledge`),
      ]);
      const respData = kResp.status === 'fulfilled' ? (kResp.value as { knowledge?: string[] }) : {};
      const kFiles = Array.isArray(respData) ? respData : (respData.knowledge || kFileList || []);
      const sFiles = sFileList || [];

      kFileListRef.current = kFiles;
      sFileListRef.current = sFiles;

      await _doLoad(skillName, kFiles, sFiles);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [_doLoad]);

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
    await _doLoad(skillName, kFileListRef.current, sFileListRef.current);
  }, [_doLoad]);

  const deleteFile = useCallback(async (skillName: string, type: 'knowledge' | 'scripts', filename: string) => {
    await api(`/skills/${skillName}/${type}/${filename}`, { method: 'DELETE' });
    if (type === 'knowledge') {
      kFileListRef.current = kFileListRef.current.filter(f => f !== filename);
    } else {
      sFileListRef.current = sFileListRef.current.filter(f => f !== filename);
    }
    await _doLoad(skillName, kFileListRef.current, sFileListRef.current);
  }, [_doLoad]);

  const createFile = useCallback(async (skillName: string, type: 'knowledge' | 'scripts', filename: string, content: string) => {
    await saveFile(skillName, type, filename, content);
    if (type === 'knowledge') {
      kFileListRef.current = [...kFileListRef.current, filename];
    } else {
      sFileListRef.current = [...sFileListRef.current, filename];
    }
    await _doLoad(skillName, kFileListRef.current, sFileListRef.current);
  }, [saveFile, _doLoad]);

  return { knowledgeFiles, scriptFiles, loading, error, loadFiles, saveFile, deleteFile, createFile };
}