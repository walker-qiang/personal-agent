import React, { useState } from 'react';
import type { SkillItem } from '../types';
import SkillEditor from './SkillEditor';

interface Props {
  skills: SkillItem[];
  onSend: (prompt: string) => void;
  onEdit: (skill: SkillItem) => void;
  onDelete: (name: string) => void;
  onCreate: () => void;
}

const SkillPanel: React.FC<Props> = ({ skills, onSend, onEdit, onDelete, onCreate }) => {
  const [editorOpen, setEditorOpen] = useState(false);
  const [editingSkill, setEditingSkill] = useState<SkillItem | undefined>(undefined);

  const handleEdit = (skill: SkillItem) => {
    setEditingSkill(skill);
    setEditorOpen(true);
  };

  const handleNew = () => {
    setEditingSkill(undefined);
    setEditorOpen(true);
  };

  const handleSave = (data: Partial<SkillItem>) => {
    if (editingSkill) {
      onEdit({ ...editingSkill, ...data } as SkillItem);
    } else {
      onCreate();
      onEdit(data as SkillItem);
    }
    setEditorOpen(false);
    setEditingSkill(undefined);
  };

  const handleClose = () => {
    setEditorOpen(false);
    setEditingSkill(undefined);
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>技能</span>
        <button style={styles.newBtn} onClick={handleNew} title="新建技能">
          +
        </button>
      </div>

      <div style={styles.list}>
        {skills.map((skill) => (
          <div key={skill.name} style={styles.item}>
            <div style={styles.itemHeader}>
              <span style={styles.itemName}>{skill.name}</span>
              <div style={styles.itemActions}>
                <button
                  style={styles.actionBtn}
                  onClick={() => onSend(skill.prompt)}
                  title="发送"
                >
                  {'\u25B6'}
                </button>
                <button
                  style={styles.actionBtn}
                  onClick={() => handleEdit(skill)}
                  title="编辑"
                >
                  {'\u270E'}
                </button>
                <button
                  style={{ ...styles.actionBtn, ...styles.deleteBtn }}
                  onClick={() => onDelete(skill.name)}
                  title="删除"
                >
                  {'\u2715'}
                </button>
              </div>
            </div>
            <div style={styles.itemDesc}>{skill.description}</div>
          </div>
        ))}
        {skills.length === 0 && (
          <div style={styles.empty}>暂无技能，点击 + 新建</div>
        )}
      </div>

      {editorOpen && (
        <SkillEditor
          skill={editingSkill}
          onSave={handleSave}
          onClose={handleClose}
        />
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: 280,
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: 'var(--bg, #1a1a2e)',
    borderLeft: '1px solid var(--rule, #333)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '16px 16px 12px',
    borderBottom: '1px solid var(--rule, #333)',
  },
  title: {
    fontSize: 16,
    fontWeight: 600,
    color: 'var(--text, #e0e0e0)',
  },
  newBtn: {
    width: 32,
    height: 32,
    borderRadius: 8,
    border: '1px solid var(--rule, #333)',
    backgroundColor: 'var(--bg2, #222)',
    color: 'var(--accent, #5b8def)',
    fontSize: 20,
    fontWeight: 600,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    lineHeight: 1,
    transition: 'background-color 0.2s',
  },
  list: {
    flex: 1,
    overflowY: 'auto',
    padding: '8px',
  },
  item: {
    padding: '12px 14px',
    borderRadius: 10,
    marginBottom: 6,
    backgroundColor: 'var(--bg2, #222)',
    border: '1px solid var(--rule, #333)',
  },
  itemHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  itemName: {
    fontSize: 14,
    fontWeight: 600,
    color: 'var(--accent, #5b8def)',
  },
  itemActions: {
    display: 'flex',
    gap: 4,
  },
  actionBtn: {
    width: 26,
    height: 26,
    borderRadius: 6,
    border: 'none',
    backgroundColor: 'transparent',
    color: 'var(--muted, #888)',
    fontSize: 12,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'color 0.15s',
  },
  deleteBtn: {
    color: '#ff5050',
  },
  itemDesc: {
    fontSize: 12,
    color: 'var(--muted, #888)',
    lineHeight: 1.5,
  },
  empty: {
    padding: 24,
    textAlign: 'center',
    color: 'var(--muted, #888)',
    fontSize: 13,
  },
};

export default SkillPanel;