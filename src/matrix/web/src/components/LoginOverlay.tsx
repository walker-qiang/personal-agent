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
  const [passwordConfirm, setPasswordConfirm] = useState('');
  const [loading, setLoading] = useState(false);
  const [localError, setLocalError] = useState('');

  const displayError = error || localError;

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setLocalError('用户名和密码不能为空');
      return;
    }
    if (tab === 'register') {
      if (password.length < 4) {
        setLocalError('密码至少4位');
        return;
      }
      if (password !== passwordConfirm) {
        setLocalError('两次输入的密码不一致');
        return;
      }
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
        <h1 style={styles.title}>Matrix</h1>

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
          {tab === 'register' && (
            <input
              style={styles.input}
              type="password"
              placeholder="确认密码"
              value={passwordConfirm}
              onChange={(e) => setPasswordConfirm(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          )}

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
    backgroundColor: 'var(--bg)',
  },
  card: {
    width: 360,
    maxWidth: '90vw',
    padding: '40px 36px',
    borderRadius: 16,
    backgroundColor: 'var(--surface)',
    border: '1px solid var(--border)',
    boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
    textAlign: 'center',
  },
  title: {
    margin: '0 0 16px',
    fontSize: 28,
    fontWeight: 800,
    color: 'var(--accent)',
  },
  tabs: {
    display: 'flex',
    marginBottom: 20,
    borderBottom: '2px solid var(--border)',
  },
  tab: {
    flex: 1,
    padding: '8px 0',
    border: 'none',
    borderBottom: '2px solid transparent',
    marginBottom: -2,
    background: 'transparent',
    color: '#6b7280',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'color 0.15s, border-color 0.15s',
    fontFamily: 'var(--font)',
  },
  tabActive: {
    color: 'var(--accent)',
    borderBottomColor: 'var(--accent)',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  input: {
    padding: '12px 16px',
    borderRadius: 10,
    border: '1px solid var(--border)',
    backgroundColor: 'var(--bg)',
    color: 'var(--text)',
    fontSize: 15,
    outline: 'none',
    fontFamily: 'var(--font)',
    boxSizing: 'border-box',
    width: '100%',
  },
  error: {
    marginTop: 4,
    padding: '8px',
    borderRadius: 8,
    backgroundColor: 'rgba(239,68,68,0.1)',
    color: 'var(--error)',
    fontSize: 13,
  },
  submitBtn: {
    padding: '12px 0',
    marginTop: 16,
    borderRadius: 10,
    border: 'none',
    backgroundColor: 'var(--accent)',
    color: '#fff',
    fontSize: 15,
    fontWeight: 600,
    cursor: 'pointer',
    width: '100%',
    fontFamily: 'var(--font)',
  },
  submitBtnDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
};

export default LoginOverlay;
