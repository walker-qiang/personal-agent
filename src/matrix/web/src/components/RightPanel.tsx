import React from 'react';

interface Props {
  todos: string[];
  artifacts: string[];
  refs: string[];
}

const RightPanel: React.FC<Props> = ({ todos, artifacts, refs }) => {
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>信息面板</span>
      </div>

      <div style={styles.content}>
        <Section title="待办事项" items={todos} emptyText="暂无待办" />
        <Section title="任务产物" items={artifacts} emptyText="暂无产物" />
        <Section title="参考信息" items={refs} emptyText="暂无参考" />
      </div>
    </div>
  );
};

interface SectionProps {
  title: string;
  items: string[];
  emptyText: string;
}

const Section: React.FC<SectionProps> = ({ title, items, emptyText }) => (
  <div style={sectionStyles.wrapper}>
    <div style={sectionStyles.title}>{title}</div>
    {items.length > 0 ? (
      <ul style={sectionStyles.list}>
        {items.map((item, idx) => (
          <li key={idx} style={sectionStyles.item}>
            {item}
          </li>
        ))}
      </ul>
    ) : (
      <div style={sectionStyles.empty}>{emptyText}</div>
    )}
  </div>
);

const sectionStyles: Record<string, React.CSSProperties> = {
  wrapper: {
    marginBottom: 16,
  },
  title: {
    fontSize: 12,
    fontWeight: 600,
    color: 'var(--muted, #888)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    marginBottom: 8,
  },
  list: {
    margin: 0,
    padding: '0 0 0 16px',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  item: {
    fontSize: 13,
    color: 'var(--text, #e0e0e0)',
    lineHeight: 1.5,
    wordBreak: 'break-word',
  },
  empty: {
    fontSize: 12,
    color: 'var(--muted, #888)',
    fontStyle: 'italic',
  },
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: 260,
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: 'var(--bg, #1a1a2e)',
    borderLeft: '1px solid var(--rule, #333)',
  },
  header: {
    padding: '16px 16px 12px',
    borderBottom: '1px solid var(--rule, #333)',
  },
  title: {
    fontSize: 16,
    fontWeight: 600,
    color: 'var(--text, #e0e0e0)',
  },
  content: {
    flex: 1,
    overflowY: 'auto',
    padding: '16px',
  },
};

export default RightPanel;