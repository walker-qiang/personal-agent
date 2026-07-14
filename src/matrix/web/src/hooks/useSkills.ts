import { useState, useCallback } from 'react';
import type { SkillItem } from '../types';
import { getToken } from '../utils/api';

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

export interface UseSkillsReturn {
  skills: SkillItem[];
  load: () => Promise<void>;
  create: (
    name: string,
    description: string,
    prompt: string,
    workflow: string,
    output_format: string,
  ) => Promise<SkillItem | null>;
  update: (name: string, data: Partial<SkillItem>) => Promise<void>;
  remove: (name: string) => Promise<void>;
}

export function useSkills(): UseSkillsReturn {
  const [skills, setSkills] = useState<SkillItem[]>([]);

  const load = useCallback(async () => {
    const res = await fetch('/skills', {
      headers: getAuthHeaders(),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(
        (body as { detail?: string }).detail || 'Failed to load skills',
      );
    }
    const data: SkillItem[] = await res.json();
    setSkills(data);
  }, []);

  const create = useCallback(
    async (
      name: string,
      description: string,
      prompt: string,
      workflow: string,
      output_format: string,
    ): Promise<SkillItem | null> => {
      const res = await fetch('/skills', {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          name,
          description,
          prompt,
          workflow,
          output_format,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
          (body as { detail?: string }).detail || 'Failed to create skill',
        );
      }
      const skill: SkillItem = await res.json();
      setSkills((prev) => [...prev, skill]);
      return skill;
    },
    [],
  );

  const update = useCallback(
    async (name: string, data: Partial<SkillItem>) => {
      const res = await fetch(`/skills/${encodeURIComponent(name)}`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify(data),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(
          (body as { detail?: string }).detail || 'Failed to update skill',
        );
      }
      const updatedSkill: SkillItem = await res.json();
      setSkills((prev) =>
        prev.map((s) => (s.name === name ? updatedSkill : s)),
      );
    },
    [],
  );

  const remove = useCallback(async (name: string) => {
    const res = await fetch(`/skills/${encodeURIComponent(name)}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(
        (body as { detail?: string }).detail || 'Failed to delete skill',
      );
    }
    setSkills((prev) => prev.filter((s) => s.name !== name));
  }, []);

  return { skills, load, create, update, remove };
}