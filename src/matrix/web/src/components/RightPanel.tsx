import React from 'react';

interface Props {
  todos: string[];
  artifacts: string[];
  refs: string[];
}

const RightPanel: React.FC<Props> = ({ todos, artifacts, refs }) => {
  return (
    <div style={styles.container}>
      <div style={styles.content}>
        <Section title="待办" items={todos} emptyText="暂无待办" />
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
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  title: {
    fontSize: 11,
    fontWeight: 600,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase',
    letterSpacing: '0.8px',
  },
  list: {
    margin: 0,
    padding: '0 0 0 16px',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  item: {
    fontSize: 12,
    color: 'var(--text)',
    lineHeight: 1.6,
    opacity: 0.7,
  },
  empty: {
    fontSize: 12,
    color: 'var(--text-secondary)',
    opacity: 0.7,
  },
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
};

export default RightPanel;