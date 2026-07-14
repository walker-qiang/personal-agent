import React from 'react';

interface ConfirmAction {
  agent: string;
  tool: string;
  args: Record<string, unknown>;
  summary: string;
}

interface ConfirmDialogProps {
  actions: ConfirmAction[];
  onApprove: () => void;
  onSkip: () => void;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({ actions, onApprove, onSkip }) => {
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(0,0,0,0.6)',
      backdropFilter: 'blur(4px)',
    }}>
      <div style={{
        background: 'var(--bg2)', borderRadius: 'var(--radius)',
        border: '1px solid var(--rule)', padding: 24,
        maxWidth: 480, width: '90%',
        boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
      }}>
        <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8, color: 'var(--text)' }}>
          确认执行操作
        </div>
        <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 16 }}>
          以下操作涉及写入或修改，请确认是否继续：
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 20 }}>
          {actions.map((action, i) => (
            <div key={i} style={{
              padding: '10px 12px',
              background: 'var(--bg)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--rule)',
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)' }}>
                {action.summary}
              </div>
              {Object.keys(action.args).length > 0 && (
                <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 4, fontFamily: 'monospace' }}>
                  {JSON.stringify(action.args)}
                </div>
              )}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            onClick={onSkip}
            style={{
              padding: '8px 16px', borderRadius: 'var(--radius-sm)',
              background: 'transparent', color: 'var(--muted)',
              border: '1px solid var(--rule)', fontSize: 13,
            }}
          >
            跳过
          </button>
          <button
            onClick={onApprove}
            style={{
              padding: '8px 16px', borderRadius: 'var(--radius-sm)',
              background: 'var(--accent)', color: '#fff',
              border: 'none', fontSize: 13, fontWeight: 600,
            }}
          >
            确认执行
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDialog;