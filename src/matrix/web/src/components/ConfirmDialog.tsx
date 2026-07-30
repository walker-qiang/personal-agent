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
      background: 'rgba(0,0,0,0.25)',
    }}>
      <div style={{
        background: 'var(--surface)', borderRadius: 12,
        padding: 24, maxWidth: 480, width: '90%',
        boxShadow: '0 10px 40px rgba(0,0,0,0.15)',
      }}>
        <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8, color: 'var(--text)' }}>
          确认执行操作
        </div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
          以下操作涉及写入或修改，请确认是否继续：
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 20 }}>
          {actions.map((action, i) => (
            <div key={i} style={{
              padding: '10px 12px',
              background: 'var(--bg)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
            }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)' }}>
                {action.summary}
              </div>
              {Object.keys(action.args).length > 0 && (
                <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
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
              padding: '8px 16px', borderRadius: 6,
              background: 'var(--border)', color: 'var(--text-secondary)',
              border: 'none', fontSize: 13, cursor: 'pointer', fontFamily: 'var(--font)',
            }}
          >
            跳过
          </button>
          <button
            onClick={onApprove}
            style={{
              padding: '8px 16px', borderRadius: 6,
              background: 'var(--accent)', color: '#fff',
              border: 'none', fontSize: 13, fontWeight: 600, cursor: 'pointer', fontFamily: 'var(--font)',
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