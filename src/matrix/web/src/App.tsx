import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './hooks/useAuth';
import { useChat } from './hooks/useChat';
import { useSessions } from './hooks/useSessions';
import { useSkills } from './hooks/useSkills';
import LoginOverlay from './components/LoginOverlay';
import SessionList from './components/SessionList';
import MessageBubble from './components/MessageBubble';
import SkillPanel from './components/SkillPanel';
import SkillEditor from './components/SkillEditor';
import ModelSelector from './components/ModelSelector';
import QuickQuestions from './components/QuickQuestions';
import StatusBar from './components/StatusBar';
import RightPanel from './components/RightPanel';
import FileUpload from './components/FileUpload';
import ConfirmDialog from './components/ConfirmDialog';
import type { SkillItem, FileInfo } from './types';

const App: React.FC = () => {
  const { authenticated, username, login, register, logout, error: authError } = useAuth();
  const { messages, send, sending, switchSession, confirmRequired, confirmActions, confirm, dismissConfirm } = useChat();
  const { sessions, currentId, setCurrentId, load: loadSessions, create: createSession, remove: removeSession } = useSessions();
  const { skills, load: loadSkills, create: createSkill, update: updateSkill, remove: removeSkill } = useSkills();

  const [input, setInput] = useState('');
  const [editingSkill, setEditingSkill] = useState<SkillItem | null>(null);
  const [showSkillEditor, setShowSkillEditor] = useState(false);
  const [file, setFile] = useState<FileInfo | null>(null);
  const [status, setStatus] = useState<'idle' | 'thinking' | 'executing' | 'generating'>('idle');
  const [statusText, setStatusText] = useState('就绪');
  const [rightPanel, setRightPanel] = useState({ todos: [] as string[], artifacts: [] as string[], refs: [] as string[] });

  const chatRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Load sessions and skills on mount
  useEffect(() => {
    if (authenticated) {
      loadSessions();
      loadSkills();
    }
  }, [authenticated, loadSessions, loadSkills]);

  // Auto-scroll
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // Update status based on sending state
  useEffect(() => {
    if (sending) {
      setStatus('executing');
      setStatusText('处理中...');
    } else {
      setStatus('idle');
      setStatusText('就绪');
    }
  }, [sending]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;

    let sid = currentId;
    if (!sid) {
      // Create a new session if none selected
      try {
        const res = await fetch('/sessions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('mx_token')}`,
          },
        });
        const data = await res.json();
        sid = data.id || data.session_id;
        setCurrentId(sid);
        loadSessions();
      } catch {
        // Fallback: generate a random session id
        sid = 's-' + Date.now();
        setCurrentId(sid);
      }
    }

    setInput('');
    setFile(null);
    send(text, sid || '', file?.file_id);
  }, [input, sending, currentId, file, send, setCurrentId, loadSessions]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleSkillSend = useCallback((prompt: string) => {
    setInput(prompt);
    inputRef.current?.focus();
  }, []);

  const handleQuickSend = useCallback((question: string) => {
    setInput(question);
    // Auto-send after a brief delay
    setTimeout(() => {
      handleSend();
    }, 50);
  }, [handleSend]);

  const handleNewSession = useCallback(async () => {
    switchSession(null);
    setCurrentId(null);
    setRightPanel({ todos: [], artifacts: [], refs: [] });
  }, [switchSession, setCurrentId]);

  const handleSelectSession = useCallback((id: string) => {
    setCurrentId(id);
    switchSession(id);
    setRightPanel({ todos: [], artifacts: [], refs: [] });
  }, [setCurrentId, switchSession]);

  if (!authenticated) {
    return <LoginOverlay onLogin={login} onRegister={register} error={authError || undefined} />;
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Left sidebar */}
      <SessionList
        sessions={sessions}
        currentId={currentId || ''}
        onSelect={handleSelectSession}
        onCreate={handleNewSession}
        onDelete={removeSession}
      />

      {/* Main chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Top bar */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '8px 16px', borderBottom: '1px solid var(--rule)',
          background: 'var(--bg2)', height: 48, flexShrink: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontWeight: 700, fontSize: 16, color: 'var(--accent)' }}>Project Matrix</span>
            <span style={{ fontSize: 12, color: 'var(--muted)' }}>{username}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ModelSelector />
            <button
              onClick={logout}
              style={{
                padding: '4px 12px', borderRadius: 'var(--radius-sm)',
                background: 'transparent', color: 'var(--muted)', fontSize: 12,
                border: '1px solid var(--rule)',
              }}
            >
              退出
            </button>
          </div>
        </div>

        {/* Messages */}
        <div
          ref={chatRef}
          style={{
            flex: 1, overflowY: 'auto', padding: '16px 24px',
            display: 'flex', flexDirection: 'column', gap: 12,
          }}
        >
          {messages.length === 0 && (
            <div style={{
              flex: 1, display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center',
              color: 'var(--muted)', gap: 16,
            }}>
              <div style={{ fontSize: 48, opacity: 0.3 }}>◈</div>
              <div style={{ fontSize: 16 }}>Project Matrix 已就绪</div>
              <div style={{ fontSize: 13 }}>选择一个技能或输入问题开始</div>
              <QuickQuestions onSend={handleQuickSend} />
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
        </div>

        {/* Input area */}
        <div style={{
          borderTop: '1px solid var(--rule)', padding: '12px 16px',
          background: 'var(--bg2)', flexShrink: 0,
        }}>
          {file && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '4px 8px', marginBottom: 8,
              background: 'var(--bg)', borderRadius: 'var(--radius-sm)',
              fontSize: 12, color: 'var(--muted)',
            }}>
              <span>📎 {file.filename} ({(file.size / 1024).toFixed(1)}KB)</span>
              <button
                onClick={() => setFile(null)}
                style={{ background: 'none', color: 'var(--muted)', fontSize: 14, padding: 0 }}
              >
                ✕
              </button>
            </div>
          )}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <FileUpload onFileSelected={setFile} />
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的问题..."
              disabled={sending}
              style={{
                flex: 1, background: 'var(--bg)', border: '1px solid var(--rule)',
                borderRadius: 'var(--radius)', padding: '10px 14px', fontSize: 14,
                color: 'var(--text)',
              }}
            />
            <button
              onClick={handleSend}
              disabled={sending || !input.trim()}
              style={{
                padding: '10px 20px', borderRadius: 'var(--radius)',
                background: sending || !input.trim() ? 'var(--rule)' : 'var(--accent)',
                color: '#fff', fontWeight: 600, fontSize: 14,
                opacity: sending || !input.trim() ? 0.5 : 1,
              }}
            >
              {sending ? '...' : '发送'}
            </button>
          </div>
        </div>

        {/* Status bar */}
        <StatusBar status={status} text={statusText} />
      </div>

      {/* Right panels */}
      <div style={{
        width: 280, borderLeft: '1px solid var(--rule)',
        background: 'var(--bg2)', overflowY: 'auto', flexShrink: 0,
        display: 'flex', flexDirection: 'column',
      }}>
        <RightPanel todos={rightPanel.todos} artifacts={rightPanel.artifacts} refs={rightPanel.refs} />
        <SkillPanel
          skills={skills}
          onSend={handleSkillSend}
          onEdit={(skill) => { setEditingSkill(skill); setShowSkillEditor(true); }}
          onDelete={removeSkill}
          onCreate={() => { setEditingSkill(null); setShowSkillEditor(true); }}
        />
      </div>

      {/* Skill Editor Modal */}
      {showSkillEditor && (
        <SkillEditor
          skill={editingSkill || undefined}
          onSave={async (data) => {
            if (editingSkill) {
              await updateSkill(editingSkill.name, data);
            } else {
              await createSkill(data.name || '', data.description || '', data.prompt || '', data.workflow || '', data.output_format || '');
            }
            setShowSkillEditor(false);
            setEditingSkill(null);
          }}
          onClose={() => { setShowSkillEditor(false); setEditingSkill(null); }}
        />
      )}

      {/* HITL Confirm Dialog */}
      {confirmRequired && (
        <ConfirmDialog
          actions={confirmActions}
          onApprove={() => confirm('approve')}
          onSkip={() => confirm('skip')}
        />
      )}
    </div>
  );
};

export default App;