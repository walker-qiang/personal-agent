import { useEffect } from 'react';

interface Shortcut {
  key: string;
  ctrl?: boolean;
  meta?: boolean;
  handler: () => void;
}

export function useKeyboard(shortcuts: Shortcut[]) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      for (const s of shortcuts) {
        const ctrl = s.ctrl ?? false;
        const meta = s.meta ?? false;
        const modMatch = (ctrl && (e.ctrlKey || e.metaKey)) || (meta && (e.metaKey || e.ctrlKey));
        const noMod = !ctrl && !meta;

        const matches = noMod
          ? e.key === s.key && !e.ctrlKey && !e.metaKey
          : modMatch && e.key === s.key;

        if (matches) {
          e.preventDefault();
          s.handler();
          return;
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [shortcuts]);
}