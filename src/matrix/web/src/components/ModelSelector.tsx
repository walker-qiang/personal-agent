import React, { useState, useEffect, useRef } from 'react';
import { api } from '../utils/api';

interface Props {
  onModelChange?: (model: string) => void;
}

interface ModelGroup {
  label: string;
  models: string[];
}

const ModelSelector: React.FC<Props> = ({ onModelChange }) => {
  const [groups, setGroups] = useState<ModelGroup[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadModels();
  }, []);

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
        image_models?: { provider: string; name: string; models: { id: string }[] }[];
        video_models?: { provider: string; name: string; models: { id: string }[] }[];
        current?: { provider: string; model: string };
      }>('/api/provider');

      const groupsList: ModelGroup[] = [];

      // Chat models from providers
      if (data.providers && data.providers.length > 0) {
        for (const prov of data.providers) {
          if (prov.models && prov.models.length > 0) {
            groupsList.push({
              label: prov.name,
              models: prov.models.map((m) => m.id),
            });
          }
        }
      }
      // Image models
      if (data.image_models && data.image_models.length > 0) {
        const allImageModels: string[] = [];
        for (const img of data.image_models) {
          if (img.models && img.models.length > 0) {
            allImageModels.push(...img.models.map((m) => m.id));
          }
        }
        if (allImageModels.length > 0) {
          groupsList.push({ label: '图片模型', models: allImageModels });
        }
      }
      // Video models
      if (data.video_models && data.video_models.length > 0) {
        const allVideoModels: string[] = [];
        for (const vid of data.video_models) {
          if (vid.models && vid.models.length > 0) {
            allVideoModels.push(...vid.models.map((m) => m.id));
          }
        }
        if (allVideoModels.length > 0) {
          groupsList.push({ label: '视频模型', models: allVideoModels });
        }
      }

      setGroups(groupsList);
      // Set selected model from current or first chat model
      if (data.current?.model) {
        setSelectedModel(data.current.model);
      } else if (data.providers && data.providers.length > 0 && data.providers[0].models?.length > 0) {
        setSelectedModel(data.providers[0].models[0].id);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '加载模型失败';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSelect = async (model: string) => {
    setSelectedModel(model);
    setOpen(false);
    try {
      await api('/api/provider', {
        method: 'POST',
        body: JSON.stringify({ model }),
      });
    } catch {
      // Silently ignore provider switch errors
    }
    onModelChange?.(model);
  };

  const displayName = selectedModel || (loading ? '加载中...' : '选择模型');

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
                  key={model}
                  style={{
                    ...styles.option,
                    ...(model === selectedModel ? styles.optionActive : {}),
                  }}
                  onClick={() => handleSelect(model)}
                >
                  {model}
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
};

export default ModelSelector;