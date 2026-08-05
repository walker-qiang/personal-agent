import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useAuth } from './hooks/useAuth';
import { useChat } from './hooks/useChat';
import { useSessions } from './hooks/useSessions';
import { useSkills } from './hooks/useSkills';
import { useKeyboard } from './hooks/useKeyboard';
import { useResizable } from './hooks/useResizable';
import LoginOverlay from './components/LoginOverlay';
import SessionList from './components/SessionList';
import MessageBubble from './components/MessageBubble';
import SkillEditor from './components/SkillEditor';
import SkillListModal from './components/SkillListModal';
import ModelSelector from './components/ModelSelector';
import RightPanel from './components/RightPanel';
import FileUpload from './components/FileUpload';
import ConfirmDialog from './components/ConfirmDialog';
import { McpPanel } from './components/McpPanel';
import { TracePanel } from './components/TracePanel';
import { BranchBanner } from './components/BranchBanner';
import type { SkillItem, FileInfo } from './types';
import { api } from './utils/api';
import { genId } from './utils/format';

const App: React.FC = () => {
  const { authenticated, username, login, register, logout, error: authError } = useAuth();
  const { messages, send, stop, sending, switchSession, confirmRequired, confirmActions, confirm, dismissConfirm, rightPanel } = useChat();
  const {
    sessions, currentId, setCurrentId, showArchive,
    load: loadSessions, remove: removeSession,
    batchArchive, batchUnarchive, batchDelete, toggleArchive, branch,
  } = useSessions();
  const { skills, load: loadSkills, create: createSkill, update: updateSkill } = useSkills();

  const [input, setInput] = useState('');
  const [editingSkill, setEditingSkill] = useState<SkillItem | null>(null);
  const [showSkillEditor, setShowSkillEditor] = useState(false);
  const [file, setFile] = useState<FileInfo | null>(null);
  const [showMcp, setShowMcp] = useState(false);
  const [showTrace, setShowTrace] = useState(false);
  const [rpanelOpen, setRpanelOpen] = useState(false);
  const [showSkillList, setShowSkillList] = useState(false);
  const [showBranchBanner, setShowBranchBanner] = useState(false);
  const [branchCount, setBranchCount] = useState(0);

  const chatRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Resizable panels — match original HTML defaults
  const sidebar = useResizable({ minWidth: 160, maxWidth: 400, defaultWidth: 220, storageKey: 'mx-sidebar-w', direction: 'right' });
  const rpanel = useResizable({ minWidth: 160, maxWidth: 500, defaultWidth: 240, storageKey: 'mx-rpanel-w', direction: 'left' });

  // Keyboard shortcuts
  useKeyboard([
    { key: 'n', ctrl: true, handler: () => handleNewSession() },
    { key: 'k', ctrl: true, handler: () => inputRef.current?.focus() },
  ]);

  // Load sessions and skills on mount
  useEffect(() => {
    if (authenticated) {
      loadSessions();
      loadSkills();
      if (currentId) {
        switchSession(currentId);
      }
    }
  }, [authenticated, currentId, loadSessions, loadSkills, switchSession]);

  // Refresh the sidebar after a stream completes so implicitly-created
  // sessions appear immediately, matching the legacy frontend.
  useEffect(() => {
    if (authenticated && !sending) {
      loadSessions();
    }
  }, [authenticated, sending, loadSessions]);

  // Load branch info when session changes
  useEffect(() => {
    if (!currentId) {
      setBranchCount(0);
      return;
    }
    api<{ branches?: unknown[] }>(`/sessions/${currentId}/branches`)
      .then(data => setBranchCount((data.branches || []).length))
      .catch(() => setBranchCount(0));
  }, [currentId]);

  // Auto-scroll
  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [messages]);

  // The legacy frontend opens the right information panel while a request is
  // running and closes it once the stream finishes.
  useEffect(() => {
    setRpanelOpen(sending);
  }, [sending]);



  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;

    let sid = currentId;
    if (!sid) {
      // Sessions are created implicitly on the server when the first
      // chat message is sent. Generate a client-side ID.
      sid = 'web-' + genId().slice(0, 8);
      setCurrentId(sid);
      loadSessions();
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

  const handleQuickSend = useCallback((question: string) => {
    let sid = currentId;
    if (!sid) {
      sid = 'web-' + genId().slice(0, 8);
      setCurrentId(sid);
      loadSessions();
    }
    send(question, sid || '');
  }, [currentId, send, setCurrentId, loadSessions]);

  const handleNewSession = useCallback(async () => {
    switchSession(null);
    setCurrentId(null);
  }, [switchSession, setCurrentId]);

  const handleSelectSession = useCallback((id: string) => {
    setCurrentId(id);
    switchSession(id);
  }, [setCurrentId, switchSession]);

  const handleBranch = useCallback(async (sessionId: string, messageId?: string) => {
    if (!messageId) return;
    if (!window.confirm('从此处分叉新对话？\n旧分支将保留在历史中。')) return;
    try {
      await branch(sessionId, messageId);
      switchSession(null);
      setShowBranchBanner(true);
      await loadSessions();
    } catch (e) {
      console.error('Branch failed:', e);
    }
  }, [branch, switchSession, loadSessions]);

  const handleMessageBranch = useCallback((messageId: string) => {
    if (currentId) {
      handleBranch(currentId, messageId);
    }
  }, [currentId, handleBranch]);

  const handleRemoveSession = useCallback(async (id: string) => {
    if (!window.confirm('删除此会话？')) return;
    await removeSession(id);
    if (id === currentId) {
      switchSession(null);
    }
  }, [currentId, removeSession, switchSession]);

  const handleBatchDelete = useCallback(async (ids: string[]) => {
    if (!window.confirm('永久删除 ' + ids.length + ' 个会话？此操作不可恢复。')) return;
    await batchDelete(ids);
    if (currentId && ids.includes(currentId)) {
      switchSession(null);
    }
  }, [batchDelete, currentId, switchSession]);

  const handleBatchArchive = useCallback(async (ids: string[]) => {
    if (!window.confirm('归档 ' + ids.length + ' 个会话？')) return;
    await batchArchive(ids);
  }, [batchArchive]);

  const handleBatchUnarchive = useCallback(async (ids: string[]) => {
    if (!window.confirm('取消归档 ' + ids.length + ' 个会话？')) return;
    await batchUnarchive(ids);
  }, [batchUnarchive]);

  if (!authenticated) {
    return <LoginOverlay onLogin={login} onRegister={register} error={authError || undefined} />;
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Left sidebar */}
      <div style={{ width: sidebar.width, flexShrink: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <SessionList
          sessions={sessions}
          currentId={currentId || ''}
          showArchive={showArchive}
          onSelect={handleSelectSession}
          onCreate={handleNewSession}
          onRefresh={loadSessions}
          onDelete={handleRemoveSession}
          onBatchArchive={handleBatchArchive}
          onBatchUnarchive={handleBatchUnarchive}
          onBatchDelete={handleBatchDelete}
          onToggleArchive={toggleArchive}
          username={username}
          onLogout={logout}
          onQuickSend={handleQuickSend}
          onOpenSkillEditor={() => setShowSkillList(true)}
          onOpenMcp={() => setShowMcp(true)}
        />
      </div>
      {/* Sidebar resize handle — matches original HTML resize-handle */}
      <div
        onMouseDown={sidebar.onMouseDown}
        style={{
          width: 5, cursor: 'col-resize', flexShrink: 0,
          background: 'transparent', transition: 'background 0.12s',
          zIndex: 10, position: 'relative',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--accent)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
      />

      {/* Main chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Top bar — matches original HTML header exactly: Matrix + Trace only */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 20px', borderBottom: '1px solid var(--border)',
          background: 'var(--surface)', flexShrink: 0,
          boxShadow: 'var(--shadow-sm)',
        }}>
          <span style={{
            fontWeight: 600, fontSize: 16,
            background: 'linear-gradient(135deg, var(--accent), #a78bfa)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>Matrix</span>
          <button
            onClick={() => setShowTrace(true)}
            style={{
              padding: '4px 12px', borderRadius: 'var(--radius)',
              background: 'var(--surface)', color: 'var(--text-secondary)', fontSize: 12,
              border: '1px solid var(--border)', cursor: 'pointer',
              fontFamily: 'var(--font)', display: 'flex', alignItems: 'center', gap: 5,
              transition: 'background 0.12s, border-color 0.12s, color 0.12s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(99,102,241,0.08)';
              e.currentTarget.style.borderColor = 'var(--accent)';
              e.currentTarget.style.color = 'var(--accent)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'var(--surface)';
              e.currentTarget.style.borderColor = 'var(--border)';
              e.currentTarget.style.color = 'var(--text-secondary)';
            }}
          >
            Trace
          </button>
        </div>

        {/* Messages — matches original HTML main area */}
        <div
          ref={chatRef}
          style={{
            flex: 1, overflowY: 'auto', padding: '24px 0',
            display: 'flex', flexDirection: 'column',
            minHeight: 0,
          }}
        >
          <div style={{
            maxWidth: 860, margin: '0 auto', width: '100%',
            padding: '0 20px', display: 'flex', flexDirection: 'column', gap: 20,
          }}>
            {branchCount > 0 && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '8px 12px', margin: '0 0 8px',
                fontSize: 12, color: 'var(--text-secondary)',
                borderBottom: '1px solid var(--border)',
              }}>
                <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{branchCount} 个分叉点</span>
                <span>此会话有多个对话分支，悬停消息可从此处分叉</span>
                <span>旧分支保留在历史中</span>
              </div>
            )}
            {showBranchBanner && <BranchBanner onClose={() => setShowBranchBanner(false)} />}
            {messages.length === 0 && !showBranchBanner && (
              <div style={{ flex: 1 }} />
            )}
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} onBranch={handleMessageBranch} />
            ))}
          </div>
        </div>

        {/* Input area — matches original HTML footer */}
        <div style={{
          borderTop: '1px solid var(--border)', padding: '0 20px 12px 20px',
          background: 'var(--surface)', flexShrink: 0,
        }}>
          {file && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              padding: '6px 10px', margin: '0 auto 6px',
              maxWidth: 860,
              background: 'var(--bg2)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius)', fontSize: 12, color: 'var(--text-secondary)',
            }}>
              <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {file.filename} ({(file.size / 1024).toFixed(1)}KB)
              </span>
              <button
                onClick={() => setFile(null)}
                style={{ background: 'none', color: 'var(--error)', fontSize: 14, padding: '2px 6px', cursor: 'pointer', border: 'none' }}
              >
                ✕
              </button>
            </div>
          )}
          <div style={{ display: 'flex', gap: 10, maxWidth: 860, margin: '0 auto', alignItems: 'center' }}>
            <FileUpload onFileSelected={setFile} />
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入问题..."
              style={{
                flex: 1, background: 'var(--bg)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius)', padding: '10px 14px', fontSize: 14,
                color: 'var(--text)', outline: 'none',
              }}
            />
            <ModelSelector sessionId={currentId || undefined} />
            {sending ? (
              <button
                onClick={stop}
                style={{
                  padding: '10px 20px', borderRadius: 'var(--radius)',
                  background: 'var(--error)', color: '#fff', fontWeight: 600, fontSize: 14,
                  border: 'none', cursor: 'pointer', whiteSpace: 'nowrap',
                }}
              >
                停止
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!input.trim()}
                style={{
                  padding: '10px 20px', borderRadius: 'var(--radius)',
                  background: !input.trim() ? 'var(--border)' : 'var(--accent)',
                  color: '#fff', fontWeight: 600, fontSize: 14, border: 'none',
                  cursor: !input.trim() ? 'not-allowed' : 'pointer',
                  opacity: !input.trim() ? 0.5 : 1,
                  whiteSpace: 'nowrap',
                }}
              >
                发送
              </button>
            )}
          </div>
        </div>

        </div>

      {/* Right panel resize handle — matches original HTML */}
      <div
        onMouseDown={rpanel.onMouseDown}
        style={{
          width: rpanelOpen ? 5 : 0, cursor: rpanelOpen ? 'col-resize' : 'default',
          flexShrink: 0, background: 'transparent', transition: 'background 0.12s',
          zIndex: 10, position: 'relative',
        }}
        onMouseEnter={(e) => { if (rpanelOpen) e.currentTarget.style.background = 'var(--accent)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
      />

      {/* Right panel — matches original HTML rpanel (hidden by default) */}
      <div style={{
        width: rpanelOpen ? rpanel.width : 0,
        borderLeft: rpanelOpen ? '1px solid var(--border)' : '1px solid transparent',
        background: 'var(--surface)',
        overflowY: rpanelOpen ? 'auto' : 'hidden',
        overflowX: 'hidden', flexShrink: 0,
        display: 'flex', flexDirection: 'column',
        padding: rpanelOpen ? '16px 12px' : 0,
        gap: rpanelOpen ? 16 : 0,
        transition: 'width 0.25s ease, padding 0.25s ease, border-color 0.25s ease',
      }}>
        {rpanelOpen && (
          <RightPanel todos={rightPanel.todos} artifacts={rightPanel.artifacts} refs={rightPanel.refs} />
        )}
      </div>

      {/* Skill List Modal */}
      {showSkillList && (
        <SkillListModal
          skills={skills}
          onEdit={(skill) => { setEditingSkill(skill); setShowSkillEditor(true); }}
          onCreate={() => { setEditingSkill(null); setShowSkillEditor(true); }}
          onClose={() => setShowSkillList(false)}
        />
      )}

      {/* Skill Editor Modal */}
      {showSkillEditor && (
        <SkillEditor
          skill={editingSkill || undefined}
          onSave={async (data) => {
            if (editingSkill) {
              await updateSkill(editingSkill.name, data);
            } else {
              await createSkill(data.name || '', data.title || '', data.description || '', data.workflow || '', data.output_format || '');
            }
            setShowSkillEditor(false);
            setEditingSkill(null);
          }}
          onClose={() => { setShowSkillEditor(false); setEditingSkill(null); }}
        />
      )}

      {/* MCP Panel */}
      {showMcp && <McpPanel onClose={() => setShowMcp(false)} />}

      {/* Trace Panel */}
      {showTrace && <TracePanel onClose={() => setShowTrace(false)} />}

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
