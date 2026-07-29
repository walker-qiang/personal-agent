import React, { useState, useEffect } from 'react';
import type { SkillItem } from '../types';
import { useSkillFiles } from '../hooks/useSkillFiles';

interface Props {
  skill?: SkillItem;
  onSave: (data: Partial<SkillItem>) => void;
  onClose: () => void;
}

const SkillEditor: React.FC<Props> = ({ skill, onSave, onClose }) => {
  const [name, setName] = useState(skill?.name || '');
  const [description, setDescription] = useState(skill?.description || '');
  const [prompt, setPrompt] = useState(skill?.prompt || '');
  const [workflow, setWorkflow] = useState(skill?.workflow || '');
  const [outputFormat, setOutputFormat] = useState(skill?.output_format || '');
  const [error, setError] = useState('');

  const { knowledgeFiles, scriptFiles, loadFiles, saveFile, deleteFile, createFile } = useSkillFiles();
  const [fileEditor, setFileEditor] = useState<{ type: 'knowledge' | 'scripts'; filename: string; content: string } | null>(null);
  const [newFileName, setNewFileName] = useState<{ type: 'knowledge' | 'scripts'; value: string } | null>(null);

  const isEditing = !!skill;

  useEffect(() => {
    if (isEditing && skill) {
      loadFiles(skill.name, skill.knowledge_files || [], skill.script_files || []);
    }
  }, [isEditing, skill, loadFiles]);

  const handleSave = () => {
    if (!name.trim()) { setError('技能名称不能为空'); return; }
    if (!prompt.trim()) { setError('提示词不能为空'); return; }
    onSave({ name: name.trim(), description: description.trim(), prompt: prompt.trim(), workflow: workflow.trim(), output_format: outputFormat.trim() });
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      if (fileEditor) { setFileEditor(null); return; }
      onClose();
    }
  };

  const handleCreateFile = async (type: 'knowledge' | 'scripts') => {
    if (!newFileName || !newFileName.value.trim() || !skill) return;
    await createFile(skill.name, type, newFileName.value.trim(), '');
    setNewFileName(null);
  };

  const handleDeleteFile = async (type: 'knowledge' | 'scripts', filename: string) => {
    if (!skill) return;
    if (confirm(`删除文件 "${filename}"？`)) {
      await deleteFile(skill.name, type, filename);
    }
  };

  const handleSaveFile = async () => {
    if (!fileEditor || !skill) return;
    await saveFile(skill.name, fileEditor.type, fileEditor.filename, fileEditor.content);
    setFileEditor(null);
  };

  return (
    <div style={styles.overlay} onClick={handleOverlayClick} onKeyDown={handleKeyDown} tabIndex={0}>
      <div style={styles.modal} onClick={e => e.stopPropagation()}>
        <div style={styles.header}>
          <h2 style={styles.title}>{isEditing ? '编辑技能' : '新建技能'}</h2>
          <button style={styles.closeBtn} onClick={onClose}>{'\u2715'}</button>
        </div>

        <div style={styles.body}>
          <div style={styles.field}>
            <label style={styles.label}>名称</label>
            <input style={styles.input} type="text" placeholder="技能名称" value={name} onChange={e => setName(e.target.value)} autoFocus />
          </div>
          <div style={styles.field}>
            <label style={styles.label}>描述</label>
            <input style={styles.input} type="text" placeholder="简要描述" value={description} onChange={e => setDescription(e.target.value)} />
          </div>
          <div style={styles.field}>
            <label style={styles.label}>提示词 (prompt)</label>
            <textarea style={styles.textarea} placeholder="技能提示词内容" value={prompt} onChange={e => setPrompt(e.target.value)} rows={4} />
          </div>
          <div style={styles.field}>
            <label style={styles.label}>工作流 (workflow)</label>
            <textarea style={styles.textarea} placeholder="工作流定义" value={workflow} onChange={e => setWorkflow(e.target.value)} rows={3} />
          </div>
          <div style={styles.field}>
            <label style={styles.label}>输出格式 (output_format)</label>
            <input style={styles.input} type="text" placeholder="例如: markdown, json, html" value={outputFormat} onChange={e => setOutputFormat(e.target.value)} />
          </div>

          {isEditing && (
            <>
              <FileSection
                label="Knowledge 文件"
                files={knowledgeFiles}
                type="knowledge"
                newFileName={newFileName?.type === 'knowledge' ? newFileName.value : ''}
                onNewNameChange={v => setNewFileName({ type: 'knowledge', value: v })}
                onCreateFile={() => handleCreateFile('knowledge')}
                onEdit={(f, c) => setFileEditor({ type: 'knowledge', filename: f, content: c })}
                onDelete={f => handleDeleteFile('knowledge', f)}
              />
              <FileSection
                label="Scripts 文件"
                files={scriptFiles}
                type="scripts"
                newFileName={newFileName?.type === 'scripts' ? newFileName.value : ''}
                onNewNameChange={v => setNewFileName({ type: 'scripts', value: v })}
                onCreateFile={() => handleCreateFile('scripts')}
                onEdit={(f, c) => setFileEditor({ type: 'scripts', filename: f, content: c })}
                onDelete={f => handleDeleteFile('scripts', f)}
              />
            </>
          )}

          {error && <div style={styles.error}>{error}</div>}
        </div>

        <div style={styles.footer}>
          <button style={styles.cancelBtn} onClick={onClose}>取消</button>
          <button style={styles.saveBtn} onClick={handleSave}>保存</button>
        </div>
      </div>

      {fileEditor && (
        <div style={styles.overlay} onClick={() => setFileEditor(null)}>
          <div style={{ ...styles.modal, width: 520 }} onClick={e => e.stopPropagation()}>
            <div style={styles.header}>
              <h2 style={styles.title}>编辑 {fileEditor.filename}</h2>
              <button style={styles.closeBtn} onClick={() => setFileEditor(null)}>{'\u2715'}</button>
            </div>
            <div style={styles.body}>
              <textarea
                style={{ ...styles.textarea, minHeight: 200 }}
                value={fileEditor.content}
                onChange={e => setFileEditor({ ...fileEditor, content: e.target.value })}
              />
            </div>
            <div style={styles.footer}>
              <button style={styles.cancelBtn} onClick={() => setFileEditor(null)}>取消</button>
              <button style={styles.saveBtn} onClick={handleSaveFile}>保存</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const FileSection: React.FC<{
  label: string;
  files: { filename: string; content: string }[];
  type: 'knowledge' | 'scripts';
  newFileName: string;
  onNewNameChange: (v: string) => void;
  onCreateFile: () => void;
  onEdit: (filename: string, content: string) => void;
  onDelete: (filename: string) => void;
}> = ({ label, files, newFileName, onNewNameChange, onCreateFile, onEdit, onDelete }) => {
  return (
    <div style={{ marginTop: 8 }}>
      <label style={{ ...styles.label, marginBottom: 6, display: 'block' }}>{label}</label>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {files.map(f => (
          <div key={f.filename} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 8px', background: 'var(--bg2)', borderRadius: 6 }}>
            <span style={{ flex: 1, fontSize: 13, color: 'var(--text)', cursor: 'pointer' }} onClick={() => onEdit(f.filename, f.content)}>
              {f.filename}
            </span>
            <button onClick={() => onDelete(f.filename)} style={styles.fileBtn}>✕</button>
          </div>
        ))}
        <div style={{ display: 'flex', gap: 6 }}>
          <input
            style={{ ...styles.input, flex: 1 }}
            placeholder="新建文件名"
            value={newFileName}
            onChange={e => onNewNameChange(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') onCreateFile(); }}
          />
          <button onClick={onCreateFile} style={styles.fileBtn}>+</button>
        </div>
      </div>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, zIndex: 10000, display: 'flex',
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)', backdropFilter: 'blur(4px)',
  },
  modal: {
    width: 520, maxHeight: '85vh', display: 'flex', flexDirection: 'column',
    borderRadius: 14, backgroundColor: 'var(--bg, #1a1a2e)',
    border: '1px solid var(--rule, #333)', boxShadow: '0 8px 40px rgba(0,0,0,0.6)',
    overflow: 'hidden',
  },
  header: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '18px 20px', borderBottom: '1px solid var(--rule, #333)',
  },
  title: { margin: 0, fontSize: 17, fontWeight: 600, color: 'var(--text, #e0e0e0)' },
  closeBtn: {
    width: 30, height: 30, borderRadius: 6, border: 'none',
    backgroundColor: 'transparent', color: 'var(--muted, #888)',
    fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  body: { flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column', gap: 16 },
  field: { display: 'flex', flexDirection: 'column', gap: 6 },
  label: { fontSize: 13, fontWeight: 500, color: 'var(--muted, #888)' },
  input: {
    padding: '10px 12px', borderRadius: 8, border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)', color: 'var(--text, #e0e0e0)',
    fontSize: 14, outline: 'none',
  },
  textarea: {
    padding: '10px 12px', borderRadius: 8, border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)', color: 'var(--text, #e0e0e0)',
    fontSize: 14, outline: 'none', resize: 'vertical' as const, fontFamily: 'inherit', lineHeight: 1.6,
  },
  error: { padding: '10px 14px', borderRadius: 8, backgroundColor: 'rgba(255, 80, 80, 0.15)', color: '#ff5050', fontSize: 13 },
  footer: { display: 'flex', justifyContent: 'flex-end', gap: 10, padding: '16px 20px', borderTop: '1px solid var(--rule, #333)' },
  cancelBtn: { padding: '8px 20px', borderRadius: 8, border: '1px solid var(--rule, #333)', backgroundColor: 'transparent', color: 'var(--muted, #888)', fontSize: 14, cursor: 'pointer' },
  saveBtn: { padding: '8px 24px', borderRadius: 8, border: 'none', backgroundColor: 'var(--accent, #5b8def)', color: '#fff', fontSize: 14, fontWeight: 600, cursor: 'pointer' },
  fileBtn: {
    padding: '4px 8px', borderRadius: 4, border: '1px solid var(--rule, #333)',
    backgroundColor: 'transparent', color: 'var(--text)', fontSize: 12, cursor: 'pointer',
  },
};

export default SkillEditor;