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
        models?: string[];
        image_models?: { id: string }[];
        video_models?: { id: string }[];
      }>('/api/provider');

      const groupsList: ModelGroup[] = [];

      if (data.models && data.models.length > 0) {
        groupsList.push({ label: '对话模型', models: data.models });
      }
      if (data.image_models && data.image_models.length > 0) {
        groupsList.push({
          label: '图片模型',
          models: data.image_models.map((m) => m.id),
        });
      }
      if (data.video_models && data.video_models.length > 0) {
        groupsList.push({
          label: '视频模型',
          models: data.video_models.map((m) => m.id),
        });
      }

      setGroups(groupsList);
      if (data.models && data.models.length > 0) {
        setSelectedModel(data.models[0]);
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
    display: 'inline-block',
  },
  trigger: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 14px',
    borderRadius: 8,
    border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)',
    color: 'var(--text, #e0e0e0)',
    fontSize: 13,
    cursor: 'pointer',
    transition: 'border-color 0.15s',
  },
  triggerText: {
    fontWeight: 500,
  },
  arrow: {
    fontSize: 10,
    color: 'var(--muted, #888)',
  },
  dropdown: {
    position: 'absolute',
    top: 'calc(100% + 6px)',
    left: 0,
    minWidth: 220,
    maxHeight: 320,
    overflowY: 'auto',
    backgroundColor: 'var(--bg2, #222)',
    border: '1px solid var(--rule, #333)',
    borderRadius: 10,
    boxShadow: '0 6px 24px rgba(0,0,0,0.5)',
    zIndex: 5000,
    padding: '6px 0',
  },
  error: {
    padding: '10px 14px',
    color: '#ff5050',
    fontSize: 12,
  },
  group: {
    padding: '4px 0',
  },
  groupLabel: {
    padding: '6px 14px',
    fontSize: 11,
    fontWeight: 600,
    color: 'var(--muted, #888)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  option: {
    display: 'block',
    width: '100%',
    padding: '8px 14px',
    border: 'none',
    backgroundColor: 'transparent',
    color: 'var(--text, #e0e0e0)',
    fontSize: 13,
    cursor: 'pointer',
    textAlign: 'left' as const,
    transition: 'background-color 0.1s',
  },
  optionActive: {
    backgroundColor: 'var(--accent, #5b8def)',
    color: '#fff',
  },
};

export default ModelSelector;