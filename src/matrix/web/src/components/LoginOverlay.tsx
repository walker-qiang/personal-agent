import React, { useState, FormEvent, KeyboardEvent } from 'react';

interface Props {
  onLogin: (username: string, password: string) => Promise<void>;
  onRegister: (username: string, password: string) => Promise<void>;
  error?: string;
}

const LoginOverlay: React.FC<Props> = ({ onLogin, onRegister, error }) => {
  const [tab, setTab] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState('');

  const displayError = error || localError;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setLocalError('用户名和密码不能为空');
      return;
    }
    setLoading(true);
    setLocalError('');
    try {
      if (tab === 'login') {
        await onLogin(username.trim(), password);
      } else {
        await onRegister(username.trim(), password);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : '操作失败';
      setLocalError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  const switchTab = (newTab: 'login' | 'register') => {
    setTab(newTab);
    setLocalError('');
  };

  return (
    <div style={styles.overlay}>
      <div style={styles.card}>
        <h1 style={styles.title}>Personal Agent</h1>

        <div style={styles.tabs}>
          <button
            style={{ ...styles.tab, ...(tab === 'login' ? styles.tabActive : {}) }}
            onClick={() => switchTab('login')}
          >
            登录
          </button>
          <button
            style={{ ...styles.tab, ...(tab === 'register' ? styles.tabActive : {}) }}
            onClick={() => switchTab('register')}
          >
            注册
          </button>
        </div>

        <form onSubmit={handleSubmit} style={styles.form}>
          <input
            style={styles.input}
            type="text"
            placeholder="用户名"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={handleKeyDown}
            autoFocus
          />
          <input
            style={styles.input}
            type="password"
            placeholder="密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={handleKeyDown}
          />

          {displayError && (
            <div style={styles.error}>{displayError}</div>
          )}

          <button
            style={{ ...styles.submitBtn, ...(loading ? styles.submitBtnDisabled : {}) }}
            type="submit"
            disabled={loading}
          >
            {loading ? '处理中...' : tab === 'login' ? '登录' : '注册'}
          </button>
        </form>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    inset: 0,
    zIndex: 9999,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.75)',
    backdropFilter: 'blur(8px)',
  },
  card: {
    width: 380,
    padding: '40px 32px',
    borderRadius: 16,
    backgroundColor: 'var(--bg, #1a1a2e)',
    border: '1px solid var(--rule, #333)',
    boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  },
  title: {
    margin: '0 0 24px',
    fontSize: 24,
    fontWeight: 700,
    textAlign: 'center',
    color: 'var(--accent, #5b8def)',
  },
  tabs: {
    display: 'flex',
    marginBottom: 24,
    borderRadius: 8,
    overflow: 'hidden',
    border: '1px solid var(--rule, #333)',
  },
  tab: {
    flex: 1,
    padding: '10px 0',
    border: 'none',
    background: 'transparent',
    color: 'var(--muted, #888)',
    fontSize: 14,
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  tabActive: {
    backgroundColor: 'var(--accent, #5b8def)',
    color: '#fff',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  input: {
    padding: '12px 14px',
    borderRadius: 8,
    border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)',
    color: 'var(--text, #e0e0e0)',
    fontSize: 14,
    outline: 'none',
    transition: 'border-color 0.2s',
  },
  error: {
    padding: '10px 14px',
    borderRadius: 8,
    backgroundColor: 'rgba(255, 80, 80, 0.15)',
    color: '#ff5050',
    fontSize: 13,
    lineHeight: 1.5,
  },
  submitBtn: {
    padding: '12px 0',
    borderRadius: 8,
    border: 'none',
    backgroundColor: 'var(--accent, #5b8def)',
    color: '#fff',
    fontSize: 15,
    fontWeight: 600,
    cursor: 'pointer',
    marginTop: 4,
    transition: 'opacity 0.2s',
  },
  submitBtnDisabled: {
    opacity: 0.6,
    cursor: 'not-allowed',
  },
};

export default LoginOverlay;