/**
 * Mock EventSource for testing SSE-based hooks.
 *
 * Usage:
 *   const es = MockEventSource.lastInstance();
 *   es.dispatchEvent('token', { data: JSON.stringify('hello') });
 *
 * The mock stores all instances so tests can find the one created by the hook.
 */

type EventListener = (event: MessageEvent) => void;

export class MockEventSource {
  static instances: MockEventSource[] = [];
  static lastUrl: string = '';

  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;

  url: string;
  readyState: number = MockEventSource.OPEN;
  onerror: ((this: EventSource, ev: Event) => unknown) | null = null;
  onopen: ((this: EventSource, ev: Event) => unknown) | null = null;
  onmessage: ((this: EventSource, ev: MessageEvent) => unknown) | null = null;

  private listeners: Map<string, Set<EventListener>> = new Map();
  private closed: boolean = false;

  constructor(url: string) {
    this.url = url;
    MockEventSource.lastUrl = url;
    MockEventSource.instances.push(this);
    // Auto-open asynchronously (mimics real EventSource)
    setTimeout(() => {
      if (!this.closed) {
        this.readyState = MockEventSource.OPEN;
        this.onopen?.call(this as unknown as EventSource, new Event('open'));
      }
    }, 0);
  }

  addEventListener(type: string, listener: EventListener): void {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type)!.add(listener);
  }

  removeEventListener(type: string, listener: EventListener): void {
    this.listeners.get(type)?.delete(listener);
  }

  close(): void {
    this.closed = true;
    this.readyState = MockEventSource.CLOSED;
  }

  /** Dispatch a named SSE event to all registered listeners. */
  dispatchEvent(type: string, payload: { data: string }): boolean {
    const event = new MessageEvent(type, payload);
    const set = this.listeners.get(type);
    if (set) {
      set.forEach(fn => fn(event));
    }
    // Also fire onmessage for 'message' type
    if (type === 'message' && this.onmessage) {
      this.onmessage.call(this as unknown as EventSource, event);
    }
    return true;
  }

  /** Simulate a connection error (fires onerror, sets readyState to CLOSED). */
  simulateConnectionError(): void {
    this.readyState = MockEventSource.CLOSED;
    this.onerror?.call(this as unknown as EventSource, new Event('error'));
  }

  /** Whether close() was called. */
  isClosed(): boolean {
    return this.closed;
  }

  // --- Static helpers ---

  static lastInstance(): MockEventSource {
    const last = MockEventSource.instances[MockEventSource.instances.length - 1];
    if (!last) {
      throw new Error('No MockEventSource instance found. Did the hook call new EventSource()?');
    }
    return last;
  }

  static clearInstances(): void {
    MockEventSource.instances = [];
    MockEventSource.lastUrl = '';
  }

  static instanceCount(): number {
    return MockEventSource.instances.length;
  }
}
