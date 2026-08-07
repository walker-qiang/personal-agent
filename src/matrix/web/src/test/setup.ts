import '@testing-library/jest-dom';
import { beforeEach } from 'vitest';
import { MockEventSource } from './helpers/mockEventSource';

// --- Node v26+ localStorage/sessionStorage polyfill ---
// Node v26 ships an experimental `localStorage` global that is `undefined`
// by default, and neither happy-dom nor jsdom can override it.
// Provide a simple in-memory implementation.
function createStorage(): Storage {
  const store = new Map<string, string>();
  return {
    get length() { return store.size; },
    key(index: number) { return [...store.keys()][index] ?? null; },
    getItem(key: string) { return store.get(key) ?? null; },
    setItem(key: string, value: string) { store.set(key, String(value)); },
    removeItem(key: string) { store.delete(key); },
    clear() { store.clear(); },
  };
}

// Only polyfill if not already available (e.g., from a working DOM env)
if (typeof globalThis.localStorage === 'undefined' || !globalThis.localStorage) {
  (globalThis as Record<string, unknown>).localStorage = createStorage();
}
if (typeof globalThis.sessionStorage === 'undefined' || !globalThis.sessionStorage) {
  (globalThis as Record<string, unknown>).sessionStorage = createStorage();
}

// Polyfill EventSource (not provided by happy-dom/jsdom)
(globalThis as unknown as { EventSource: typeof EventSource }).EventSource =
  MockEventSource as unknown as typeof EventSource;

// Reset all mocks between tests
beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  MockEventSource.clearInstances();
});
