import React, { useState, useEffect } from 'react';
import { useMcp } from '../hooks/useMcp';
import type { McpServer } from '../types';

interface Props {
  onClose: () => void;
}

export const McpPanel: React.FC<Props> = ({ onClose }) => {
  const { servers, loading, error, load, create, update, remove, toggle } = useMcp();
  const [editing, setEditing] = useState<McpServer | null>(null);
  const [creating, setCreating] = useState(false);

  useEffect(() => { load(); }, [load]);

  const handleSave = async (data: Record<string, string>) => {
    const envLines = (data.envStr || '').split('\n').filter(Boolean);
    const env: Record<string, string> = {};
    envLines.forEach((line: string) => {
      const idx = line.indexOf('=');
      if (idx > 0) env[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
    });

    const server = {
      name: data.name,
      transport: data.transport as 'stdio' | 'http',
      ...(data.transport === 'stdio'
        ? { command: data.command, args: (data.args || '').split(/\s+/).filter(Boolean), env: Object.keys(env).length > 0 ? env : undefined }
        : { url: data.url }),
      ...(data.timeout ? { timeout: parseInt(data.timeout, 10) } : {}),
    };

    if (creating) {
      await create(server as any);
      setCreating(false);
    } else if (editing) {
      await update(editing.name, server);
      setEditing(null);
    }
  };

  const handleDelete = async (name: string) => {
    if (confirm(`确定删除 MCP 服务 "${name}"？`)) {
      await remove(name);
    }
  };

  const overlayStyle: React.CSSProperties = {
    position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  };
  const modalStyle: React.CSSProperties = {
    background: 'var(--surface)', borderRadius: 12, width: 600, maxHeight: '80vh',
    overflow: 'hidden', display: 'flex', flexDirection: 'column',
    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
  };
  const headerStyle: React.CSSProperties = {
    padding: '16px 20px', borderBottom: '1px solid var(--border)',
    fontWeight: 600, fontSize: 16, display: 'flex', justifyContent: 'space-between',
    alignItems: 'center',
  };
  const bodyStyle: React.CSSProperties = {
    padding: 20, overflowY: 'auto', flex: 1,
  };

  const showForm = creating || editing !== null;

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={e => e.stopPropagation()}>
        <div style={headerStyle}>
          <span>MCP 服务管理</span>
          <div style={{ display: 'flex', gap: 8 }}>
            {!showForm && (
              <button onClick={() => setCreating(true)} style={btnStyle}>+ 添加</button>
            )}
            <button onClick={onClose} style={{ ...btnStyle, background: 'transparent' }}>✕</button>
          </div>
        </div>
        <div style={bodyStyle}>
          {error && <div style={{ color: 'var(--error)', marginBottom: 12 }}>{error}</div>}

          {showForm ? (
            <McpForm
              initial={editing}
              onSave={handleSave}
              onCancel={() => { setEditing(null); setCreating(false); }}
            />
          ) : loading ? (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-secondary)' }}>加载中...</div>
          ) : servers.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-secondary)' }}>
              暂无 MCP 服务，点击"+ 添加"创建
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {servers.map(s => (
                <McpServerItem
                  key={s.name}
                  server={s}
                  onEdit={() => setEditing(s)}
                  onDelete={() => handleDelete(s.name)}
                  onToggle={() => toggle(s.name)}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const McpServerItem: React.FC<{
  server: McpServer;
  onEdit: () => void;
  onDelete: () => void;
  onToggle: () => void;
}> = ({ server, onEdit, onDelete, onToggle }) => {
  const itemStyle: React.CSSProperties = {
    display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px',
    background: 'var(--bg2)', borderRadius: 8, border: '1px solid var(--border)',
  };
  const dotStyle: React.CSSProperties = {
    width: 8, height: 8, borderRadius: '50%',
    background: server.connected ? '#22c55e' : '#6b7280',
    flexShrink: 0,
  };
  return (
    <div style={itemStyle}>
      <div style={dotStyle} />
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 500, fontSize: 14 }}>{server.name}</div>
        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          {server.transport === 'stdio' ? `stdio · ${server.command}` : `http · ${server.url}`}
          {server.connected && server.tool_count !== undefined && ` · ${server.tool_count} 工具`}
        </div>
      </div>
      <button onClick={onToggle} style={{ ...btnStyle, fontSize: 12, padding: '4px 10px' }}>
        {server.connected ? '断开' : '连接'}
      </button>
      <button onClick={onEdit} style={{ ...btnStyle, background: 'transparent', fontSize: 12 }}>编辑</button>
      <button onClick={onDelete} style={{ ...btnStyle, background: 'transparent', color: 'var(--error)', fontSize: 12 }}>删除</button>
    </div>
  );
};

const McpForm: React.FC<{
  initial: McpServer | null;
  onSave: (data: Record<string, string>) => void;
  onCancel: () => void;
}> = ({ initial, onSave, onCancel }) => {
  const [transport, setTransport] = useState(initial?.transport || 'stdio');
  const [name, setName] = useState(initial?.name || '');
  const [command, setCommand] = useState(initial?.command || '');
  const [args, setArgs] = useState(initial?.args?.join(' ') || '');
  const [url, setUrl] = useState(initial?.url || '');
  const [envStr, setEnvStr] = useState(
    initial?.env ? Object.entries(initial.env).map(([k, v]) => `${k}=${v}`).join('\n') : ''
  );
  const [timeout, setTimeout_] = useState(initial?.timeout?.toString() || '');

  const handleSubmit = () => {
    if (!name.trim()) return;
    onSave({ name: name.trim(), transport, command, args, url, envStr, timeout });
  };

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '8px 10px', borderRadius: 6,
    border: '1px solid var(--border)', background: 'var(--bg)',
    color: 'var(--text)', fontSize: 13, boxSizing: 'border-box',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div>
        <label style={labelStyle}>传输方式</label>
        <div style={{ display: 'flex', gap: 8 }}>
          {(['stdio', 'http'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTransport(t)}
              style={{
                ...btnStyle,
                background: transport === t ? 'var(--accent)' : 'var(--bg2)',
                color: transport === t ? '#fff' : 'var(--text)',
                fontSize: 12, padding: '6px 14px',
              }}
            >
              {t}
            </button>
          ))}
        </div>
      </div>
      <div>
        <label style={labelStyle}>名称</label>
        <input style={inputStyle} value={name} onChange={e => setName(e.target.value)} placeholder="服务名称" />
      </div>
      {transport === 'stdio' ? (
        <>
          <div>
            <label style={labelStyle}>命令</label>
            <input style={inputStyle} value={command} onChange={e => setCommand(e.target.value)} placeholder="例如: npx" />
          </div>
          <div>
            <label style={labelStyle}>参数</label>
            <input style={inputStyle} value={args} onChange={e => setArgs(e.target.value)} placeholder="空格分隔，例如: -y @playwright/mcp" />
          </div>
          <div>
            <label style={labelStyle}>环境变量（每行 KEY=VALUE）</label>
            <textarea
              style={{ ...inputStyle, minHeight: 60, resize: 'vertical' }}
              value={envStr}
              onChange={e => setEnvStr(e.target.value)}
              placeholder="KEY=VALUE"
            />
          </div>
        </>
      ) : (
        <div>
          <label style={labelStyle}>URL</label>
          <input style={inputStyle} value={url} onChange={e => setUrl(e.target.value)} placeholder="http://..." />
        </div>
      )}
      <div>
        <label style={labelStyle}>超时（秒）</label>
        <input style={inputStyle} value={timeout} onChange={e => setTimeout_(e.target.value)} placeholder="默认超时" />
      </div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 4 }}>
        <button onClick={onCancel} style={{ ...btnStyle, background: 'var(--bg2)' }}>取消</button>
        <button onClick={handleSubmit} style={btnStyle}>{initial ? '保存' : '创建'}</button>
      </div>
    </div>
  );
};

const labelStyle: React.CSSProperties = {
  display: 'block', fontSize: 12, color: 'var(--text-secondary)',
  marginBottom: 4, fontWeight: 500,
};

const btnStyle: React.CSSProperties = {
  padding: '6px 14px', borderRadius: 6, border: 'none',
  background: 'var(--accent)', color: '#fff', cursor: 'pointer',
  fontSize: 13, fontWeight: 500,
};