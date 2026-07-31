import React, { useState, useEffect, useRef } from 'react';
import { api } from '../utils/api';

interface Props {
  sessionId?: string;
  onModelChange?: (model: string) => void;
}

interface ModelOption {
  provider: string;
  id: string;
  name: string;
  desc?: string;
}

interface ModelGroup {
  label: string;
  provider: string;
  models: ModelOption[];
}

const ModelSelector: React.FC<Props> = ({ sessionId, onModelChange }) => {
  const [groups, setGroups] = useState<ModelGroup[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadModels();
  }, [sessionId]);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const loadModels = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await api<{
        providers?: { id: string; name: string; models: { id: string; name: string; desc?: string }[] }[];
        current?: { provider: string; model: string };
      }>(`/api/provider${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ''}`);

      const groupsList: ModelGroup[] = [];

      // Chat models from providers
      if (data.providers && data.providers.length > 0) {
        for (const prov of data.providers) {
          if (prov.models && prov.models.length > 0) {
            groupsList.push({
              label: prov.name,
              provider: prov.id,
              models: prov.models.map((m) => ({ ...m, provider: prov.id })),
            });
          }
        }
      }

      setGroups(groupsList);
      // Set selected model from current or first chat model
      if (data.current?.provider && data.current?.model) {
        setSelectedModel(`${data.current.provider}:${data.current.model}`);
      } else if (data.providers && data.providers.length > 0 && data.providers[0].models?.length > 0) {
        setSelectedModel(`${data.providers[0].id}:${data.providers[0].models[0].id}`);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '加载模型失败';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = async (option: ModelOption) => {
    const key = `${option.provider}:${option.id}`;
    if (key === selectedModel) {
      setOpen(false);
      return;
    }
    setSelectedModel(key);
    setOpen(false);
    try {
      await api('/api/provider', {
        method: 'POST',
        body: JSON.stringify({ provider: option.provider, model: option.id, session_id: sessionId }),
      });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : '切换模型失败');
      await loadModels();
      return;
    }
    onModelChange?.(key);
  };

  const selectedOption = groups.flatMap((group) => group.models).find(
    (option) => `${option.provider}:${option.id}` === selectedModel,
  );
  const displayName = selectedOption?.name || (loading ? '加载中...' : '选择模型');

  return (
    <div ref={ref} style={styles.container}>
      <button
        style={styles.trigger}
        onClick={() => setOpen(!open)}
        disabled={loading}
      >
        <span style={styles.triggerText}>{displayName}</span>
        <span style={styles.arrow}>{open ? '\u25B2' : '\u25BC'}</span>
      </button>

      {open && (
        <div style={styles.dropdown}>
          {error && (
            <div style={styles.error}>{error}</div>
          )}
          {groups.map((group) => (
            <div key={group.label} style={styles.group}>
              <div style={styles.groupLabel}>{group.label}</div>
              {group.models.map((model) => (
                <button
                  key={`${model.provider}:${model.id}`}
                  style={{
                    ...styles.option,
                    ...(selectedModel === `${model.provider}:${model.id}` ? styles.optionActive : {}),
                  }}
                  onClick={() => handleSelect(model)}
                >
                  <span>{model.name}</span>
                  {model.desc && <span style={styles.optionDesc}>{model.desc}</span>}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'relative',
    flexShrink: 0,
  },
  trigger: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '8px 10px',
    borderRadius: 'var(--radius)',
    border: '1px solid var(--border)',
    backgroundColor: 'var(--bg)',
    color: 'var(--text)',
    fontSize: 12,
    cursor: 'pointer',
    transition: 'border-color 0.15s',
    fontFamily: 'var(--font)',
    whiteSpace: 'nowrap',
  },
  triggerText: {
    fontWeight: 500,
    fontSize: 12,
  },
  arrow: {
    fontSize: 10,
    color: 'var(--text-secondary)',
  },
  dropdown: {
    position: 'absolute',
    bottom: 'calc(100% + 4px)',
    left: 0,
    minWidth: 200,
    maxHeight: 420,
    overflowY: 'auto',
    backgroundColor: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    boxShadow: '0 -4px 24px rgba(0,0,0,0.12)',
    zIndex: 5000,
    padding: '4px 0',
  },
  error: {
    padding: '10px 14px',
    color: 'var(--error)',
    fontSize: 12,
  },
  group: {
    padding: '2px 0',
  },
  groupLabel: {
    padding: '6px 14px',
    fontSize: 11,
    fontWeight: 600,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  option: {
    width: '100%',
    padding: '10px 14px',
    border: 'none',
    backgroundColor: 'transparent',
    color: 'var(--text-secondary)',
    fontSize: 13,
    cursor: 'pointer',
    textAlign: 'left' as const,
    transition: 'background 0.1s',
    fontFamily: 'var(--font)',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  optionActive: {
    backgroundColor: 'rgba(99,102,241,0.08)',
    color: 'var(--accent)',
  },
  optionDesc: {
    marginLeft: 'auto',
    color: 'var(--text-secondary)',
    fontSize: 11,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap' as const,
  },
};

export default ModelSelector;
