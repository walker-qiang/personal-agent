import React, { useEffect } from 'react';
import { useTrace } from '../hooks/useTrace';

interface Props {
  onClose: () => void;
}

export const TracePanel: React.FC<Props> = ({ onClose }) => {
  const { stats, sessions, events, sessionId, loading, error, loadStats, loadSessions, loadEvents, backToList } = useTrace();

  useEffect(() => {
    loadStats();
    loadSessions();
  }, [loadStats, loadSessions]);

  const drawerStyle: React.CSSProperties = {
    position: 'fixed', top: 0, right: 0, bottom: 0, width: 420,
    background: 'var(--surface)', zIndex: 1000,
    display: 'flex', flexDirection: 'column',
    boxShadow: '-4px 0 16px rgba(0,0,0,0.3)',
    animation: 'slideInRight 0.2s ease',
  };
  const headerStyle: React.CSSProperties = {
    padding: '14px 16px', borderBottom: '1px solid var(--border)',
    fontWeight: 600, fontSize: 15, display: 'flex', justifyContent: 'space-between',
    alignItems: 'center',
  };
  const bodyStyle: React.CSSProperties = {
    flex: 1, overflowY: 'auto', padding: 16,
    display: 'flex', flexDirection: 'column', gap: 12,
  };

  return (
    <>
      <div style={{ position: 'fixed', inset: 0, zIndex: 999 }} onClick={onClose} />
      <div style={drawerStyle}>
        <div style={headerStyle}>
          <span>{sessionId ? 'Trace 事件详情' : 'Trace 追踪'}</span>
          <div style={{ display: 'flex', gap: 8 }}>
            {sessionId && <button onClick={backToList} style={btnStyle}>返回</button>}
            <button onClick={onClose} style={{ ...btnStyle, background: 'transparent' }}>✕</button>
          </div>
        </div>
        <div style={bodyStyle}>
          {error && <div style={{ color: 'var(--error)', fontSize: 13 }}>{error}</div>}

          {!sessionId && stats && (
            <div style={{ display: 'flex', gap: 10 }}>
              <StatCard label="Events" value={stats.total_events} color="var(--accent)" />
              <StatCard label="Sessions" value={stats.total_sessions} color="#22c55e" />
              <StatCard label="Errors" value={stats.total_errors} color="var(--error)" />
            </div>
          )}

          {loading && <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: 20 }}>加载中...</div>}

          {!sessionId && !loading && sessions.length === 0 && (
            <div style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: 20 }}>暂无 Trace 数据</div>
          )}

          {sessionId ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {events.map((evt, i) => (
                <div key={i} style={eventItemStyle}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 12, color: evt.ok === false ? 'var(--error)' : 'var(--success)' }}>
                      {evt.event_type}
                    </span>
                    {evt.tool_name && <span style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 500 }}>{evt.tool_name}</span>}
                    {evt.agent_id && <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{evt.agent_id}</span>}
                    <span style={{ flex: 1 }} />
                    {evt.elapsed_ms !== undefined && (
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{evt.elapsed_ms}ms</span>
                    )}
                  </div>
                  {evt.error && <div style={{ fontSize: 11, color: 'var(--error)', marginTop: 4 }}>{evt.error}</div>}
                  {evt.result_preview && (
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4, maxHeight: 60, overflow: 'hidden' }}>
                      {evt.result_preview}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            sessions.map(s => (
              <div
                key={s.session_id}
                style={{ ...sessionItemStyle, cursor: 'pointer' }}
                onClick={() => loadEvents(s.session_id)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 13, fontWeight: 500, fontFamily: 'monospace' }}>
                    {s.session_id.slice(0, 12)}...
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                    {new Date(s.started).toLocaleString('zh-CN')}
                  </span>
                </div>
                <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                  <span>{s.total_events} events</span>
                  <span>{s.tool_calls} tool calls</span>
                  {s.errors > 0 && <span style={{ color: 'var(--error)' }}>{s.errors} errors</span>}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </>
  );
};

const StatCard: React.FC<{ label: string; value: number; color: string }> = ({ label, value, color }) => (
  <div style={{
    flex: 1, padding: '12px 10px', borderRadius: 8,
    background: 'var(--bg2)', textAlign: 'center',
    border: `1px solid var(--border)`,
  }}>
    <div style={{ fontSize: 22, fontWeight: 700, color }}>{value}</div>
    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>{label}</div>
  </div>
);

const btnStyle: React.CSSProperties = {
  padding: '4px 12px', borderRadius: 6, border: 'none',
  background: 'var(--accent)', color: '#fff', cursor: 'pointer',
  fontSize: 12, fontWeight: 500,
};

const sessionItemStyle: React.CSSProperties = {
  padding: '10px 12px', borderRadius: 8, background: 'var(--bg2)',
  border: '1px solid var(--border)',
};

const eventItemStyle: React.CSSProperties = {
  padding: '8px 10px', borderRadius: 6, background: 'var(--bg2)',
  border: '1px solid var(--border)', fontSize: 12,
};