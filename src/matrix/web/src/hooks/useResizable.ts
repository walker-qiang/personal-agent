import { useState, useCallback, useRef, useEffect } from 'react';

interface UseResizableOptions {
  minWidth: number;
  maxWidth: number;
  defaultWidth: number;
  storageKey: string;
  direction?: 'left' | 'right';
}

export function useResizable(options: UseResizableOptions) {
  const { minWidth, maxWidth, defaultWidth, storageKey, direction = 'right' } = options;

  const [width, setWidth] = useState(() => {
    const stored = localStorage.getItem(storageKey);
    if (stored) {
      const val = parseInt(stored, 10);
      if (!isNaN(val)) return Math.min(Math.max(val, minWidth), maxWidth);
    }
    return defaultWidth;
  });

  const dragging = useRef(false);
  const startX = useRef(0);
  const startW = useRef(0);
  // Keep latest width in a ref so mouseup doesn't need `width` in deps.
  const widthRef = useRef(width);
  widthRef.current = width;

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    startX.current = e.clientX;
    startW.current = width;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  }, [width]);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const delta = direction === 'left' ? startX.current - e.clientX : e.clientX - startX.current;
      const newWidth = Math.min(Math.max(startW.current + delta, minWidth), maxWidth);
      setWidth(newWidth);
    };

    const onMouseUp = () => {
      if (!dragging.current) return;
      dragging.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      localStorage.setItem(storageKey, String(widthRef.current));
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [minWidth, maxWidth, storageKey, direction]);

  return { width, onMouseDown };
}
