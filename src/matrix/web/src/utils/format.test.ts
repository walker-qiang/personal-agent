import { describe, it, expect } from 'vitest';
import {
  escapeHtml,
  genId,
  sanitizeUrl,
  formatMoney,
  formatPercent,
  formatDuration,
  formatToolResult,
} from './format';

describe('escapeHtml', () => {
  it('escapes special HTML characters', () => {
    expect(escapeHtml('<script>alert("xss")</script>')).toBe(
      '&lt;script&gt;alert("xss")&lt;/script&gt;',
    );
  });

  it('escapes ampersand', () => {
    expect(escapeHtml('a & b')).toBe('a &amp; b');
  });

  it('returns plain text unchanged', () => {
    expect(escapeHtml('hello world')).toBe('hello world');
  });

  it('handles empty string', () => {
    expect(escapeHtml('')).toBe('');
  });
});

describe('genId', () => {
  it('returns a non-empty string', () => {
    const id = genId();
    expect(id).toBeTruthy();
    expect(typeof id).toBe('string');
  });

  it('generates unique IDs (collision check with 1000 iterations)', () => {
    const ids = new Set<string>();
    for (let i = 0; i < 1000; i++) {
      ids.add(genId());
    }
    expect(ids.size).toBe(1000);
  });
});

describe('sanitizeUrl', () => {
  it('accepts http URLs', () => {
    expect(sanitizeUrl('http://example.com/image.png')).toBe('http://example.com/image.png');
  });

  it('accepts https URLs', () => {
    expect(sanitizeUrl('https://example.com/photo.jpg')).toBe('https://example.com/photo.jpg');
  });

  it('accepts data: image URLs', () => {
    const dataUrl = 'data:image/png;base64,iVBORw0KGgo=';
    expect(sanitizeUrl(dataUrl)).toBe(dataUrl);
  });

  it('rejects javascript: URLs', () => {
    expect(sanitizeUrl('javascript:alert(1)')).toBe('');
  });

  it('rejects non-image data: URLs', () => {
    expect(sanitizeUrl('data:text/html,<script>')).toBe('');
  });

  it('returns empty for empty input', () => {
    expect(sanitizeUrl('')).toBe('');
  });

  it('is case-insensitive for scheme detection', () => {
    expect(sanitizeUrl('HTTPS://Example.COM/x.PNG')).toBe('HTTPS://Example.COM/x.PNG');
  });
});

describe('formatMoney', () => {
  it('formats integers with 2 decimal places', () => {
    expect(formatMoney(1000)).toBe('1,000.00');
  });

  it('formats decimals with proper grouping', () => {
    expect(formatMoney(1234567.89)).toBe('1,234,567.89');
  });

  it('handles zero', () => {
    expect(formatMoney(0)).toBe('0.00');
  });

  it('handles negative numbers', () => {
    expect(formatMoney(-1234.5)).toBe('-1,234.50');
  });
});

describe('formatPercent', () => {
  it('converts decimal to percentage', () => {
    expect(formatPercent(0.15)).toBe('15.00%');
  });

  it('handles zero', () => {
    expect(formatPercent(0)).toBe('0.00%');
  });

  it('handles values > 1 (over 100%)', () => {
    expect(formatPercent(1.5)).toBe('150.00%');
  });

  it('handles negative values', () => {
    expect(formatPercent(-0.25)).toBe('-25.00%');
  });
});

describe('formatDuration', () => {
  it('formats milliseconds', () => {
    expect(formatDuration(500)).toBe('500ms');
  });

  it('formats seconds with one decimal', () => {
    expect(formatDuration(1500)).toBe('1.5s');
  });

  it('formats exact seconds without decimals', () => {
    expect(formatDuration(3000)).toBe('3.0s');
  });

  it('formats minutes', () => {
    expect(formatDuration(65000)).toBe('1m 5s');
  });

  it('formats large durations', () => {
    expect(formatDuration(125000)).toBe('2m 5s');
  });

  it('handles zero', () => {
    expect(formatDuration(0)).toBe('0ms');
  });
});

describe('formatToolResult', () => {
  it('returns placeholder for null', () => {
    expect(formatToolResult('test', null)).toBe('（无结果）');
  });

  it('returns placeholder for undefined', () => {
    expect(formatToolResult('test', undefined)).toBe('（无结果）');
  });

  it('returns string result as-is', () => {
    expect(formatToolResult('test', 'hello')).toBe('hello');
  });

  it('parses JSON string results', () => {
    const json = '{"name":"test","value":42}';
    const result = formatToolResult('test', json);
    expect(result).toContain('字段');
    expect(result).toContain('name');
    expect(result).toContain('test');
    expect(result).toContain('42');
  });

  it('returns empty array placeholder', () => {
    expect(formatToolResult('test', [])).toBe('（空列表）');
  });

  it('formats array of objects as table', () => {
    const data = [{ name: 'Alice', age: 30 }];
    const result = formatToolResult('test', data);
    expect(result).toContain('<table>');
    expect(result).toContain('Alice');
    expect(result).toContain('30');
  });

  it('formats object with image URLs', () => {
    const data = { images: [{ url: 'https://example.com/img.png' }] };
    const result = formatToolResult('test', data);
    expect(result).toContain('<img');
    expect(result).toContain('https://example.com/img.png');
  });

  it('formats object with video URLs', () => {
    const data = { videos: [{ url: 'https://example.com/vid.mp4' }] };
    const result = formatToolResult('test', data);
    expect(result).toContain('<video');
    expect(result).toContain('https://example.com/vid.mp4');
  });

  it('sanitizes malicious URLs in images', () => {
    const data = { images: [{ url: 'javascript:alert(1)' }] };
    const result = formatToolResult('test', data);
    expect(result).not.toContain('javascript');
  });

  it('formats key-value pairs as table', () => {
    const data = { name: 'test', count: 42 };
    const result = formatToolResult('test', data);
    expect(result).toContain('<table>');
    expect(result).toContain('name');
    expect(result).toContain('test');
    expect(result).toContain('42');
  });

  it('skips long prompt values (>50 chars)', () => {
    const longPrompt = 'a'.repeat(60);
    const data = { prompt: longPrompt, name: 'test' };
    const result = formatToolResult('test', data);
    expect(result).not.toContain(longPrompt);
    expect(result).toContain('name');
  });

  it('formats numbers with locale formatting', () => {
    const data = { count: 1234567 };
    const result = formatToolResult('test', data);
    expect(result).toContain('1,234,567');
  });

  it('escapes HTML in string values', () => {
    const data = { name: '<script>alert(1)</script>' };
    const result = formatToolResult('test', data);
    expect(result).not.toContain('<script>');
    expect(result).toContain('&lt;script&gt;');
  });
});
