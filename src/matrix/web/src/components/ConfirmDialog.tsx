import React from 'react';

interface ConfirmAction {
  agent?: string;
  tool?: string;
  name?: string;
  args: Record<string, unknown>;
  summary?: string;
  reason?: string;
  risk?: string;
  approval_id?: string;
  operation_id?: string;
}

interface ConfirmDialogProps {
  actions: ConfirmAction[];
  onApprove: () => void;
  onSkip: () => void;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({ actions, onApprove, onSkip }) => {
  const isWriteback = actions.some((action) => (action.name || action.tool) === 'writeback.execute_plan');

  const planFor = (action: ConfirmAction): Record<string, any> | null => {
    const plan = action.args?.plan;
    return plan && typeof plan === 'object' ? plan as Record<string, any> : null;
  };

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
          {isWriteback ? '确认写回个人数据' : '确认执行操作'}
        </div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
          {isWriteback ? '以下是即将落盘、提交并推送的不可变写回计划，请核对后确认：' : '以下操作涉及写入或修改，请确认是否继续：'}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 20 }}>
          {actions.map((action, i) => (
            <div key={i} style={{
              padding: '10px 12px',
              background: 'var(--bg)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
            }}>
              {planFor(action) ? (
                <WritebackPlanPreview plan={planFor(action)!} />
              ) : (
                <>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)' }}>
                    {action.summary || action.reason || action.name || action.tool || '待确认操作'}
                  </div>
                  {Object.keys(action.args || {}).length > 0 && (
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
                      {JSON.stringify(action.args)}
                    </div>
                  )}
                </>
              )}
              {!planFor(action) && action.risk && (
                <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
                  风险：{action.risk}
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

function WritebackPlanPreview({ plan }: { plan: Record<string, any> }) {
  const preview = (plan.preview || {}) as Record<string, any>;
  const precondition = (plan.precondition || {}) as Record<string, any>;
  const reversible = (plan.reversibility || {}) as Record<string, any>;
  const paths = Array.isArray(preview.affected_paths) ? preview.affected_paths : [];
  return (
    <div>
      <div style={previewStyles.title}>{preview.summary || plan.operation || 'durable 写回'}</div>
      <div style={previewStyles.grid}>
        <span style={previewStyles.label}>写入文件</span>
        <span>{paths.length ? paths.join('、') : '由计划确定'}</span>
        <span style={previewStyles.label}>Git 动作</span>
        <span>{preview.git_action || 'commit and push'}</span>
        <span style={previewStyles.label}>缓存范围</span>
        <span>{preview.cache_scope || '未指定'}</span>
        <span style={previewStyles.label}>可逆方式</span>
        <span>{reversible.operation || reversible.kind || '需人工修正'}</span>
        <span style={previewStyles.label}>计划版本</span>
        <span>{plan.plan_id || '未指定'} · {plan.plan_hash || '无 hash'}</span>
        <span style={previewStyles.label}>提交基线</span>
        <span>{precondition.asset_repo_commit || '未指定'}</span>
      </div>
      <div style={previewStyles.notice}>
        风险：{plan.risk || '未指定'} · 过期时间：{plan.expires_at || '未指定'}
      </div>
    </div>
  );
}

const previewStyles: Record<string, React.CSSProperties> = {
  title: { fontSize: 13, fontWeight: 600, color: 'var(--accent)', marginBottom: 10 },
  grid: { display: 'grid', gridTemplateColumns: '88px 1fr', gap: '6px 10px', fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.45 },
  label: { color: 'var(--text-secondary)', opacity: 0.72 },
  notice: { marginTop: 10, paddingTop: 8, borderTop: '1px solid var(--border)', fontSize: 11, color: 'var(--text-secondary)' },
};

export default ConfirmDialog;
