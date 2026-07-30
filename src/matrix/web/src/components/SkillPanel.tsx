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
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  title: {
    fontSize: 11,
    fontWeight: 600,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase',
    letterSpacing: '0.8px',
  },
  newBtn: {
    width: 24,
    height: 24,
    borderRadius: 4,
    border: 'none',
    backgroundColor: 'transparent',
    color: 'var(--text-secondary)',
    fontSize: 14,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    opacity: 0.4,
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  item: {
    padding: '6px 8px',
    borderRadius: 6,
    transition: 'background 0.1s',
  },
  itemHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 2,
  },
  itemName: {
    fontSize: 13,
    fontWeight: 400,
    color: 'var(--text-secondary)',
  },
  itemActions: {
    display: 'none',
    gap: 4,
    marginLeft: 'auto',
  },
  actionBtn: {
    width: 24,
    height: 24,
    borderRadius: 4,
    border: 'none',
    backgroundColor: 'transparent',
    color: 'var(--text-secondary)',
    fontSize: 12,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    opacity: 0.5,
  },
  deleteBtn: {
    color: 'var(--error)',
  },
  itemDesc: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
    opacity: 0.7,
  },
  empty: {
    padding: 8,
    textAlign: 'center',
    color: 'var(--text-secondary)',
    fontSize: 12,
    opacity: 0.6,
  },
};

export default SkillPanel;