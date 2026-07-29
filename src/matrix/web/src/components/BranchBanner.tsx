import React from 'react';

interface Props {
  onClose: () => void;
}

export const BranchBanner: React.FC<Props> = ({ onClose }) => {
  return (
    <div style={{
      padding: '10px 16px', margin: '8px 0', borderRadius: 8,
      background: 'rgba(99, 102, 241, 0.12)', border: '1px solid rgba(99, 102, 241, 0.3)',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      fontSize: 13, color: 'var(--text)',
    }}>
      <span>↷ 已从分叉点开始新对话，旧分支保留在历史中</span>
      <button onClick={onClose} style={{
        background: 'transparent', border: 'none', color: 'var(--text-secondary)',
        cursor: 'pointer', fontSize: 14, padding: '2px 6px',
      }}>
        ✕
      </button>
    </div>
  );
};