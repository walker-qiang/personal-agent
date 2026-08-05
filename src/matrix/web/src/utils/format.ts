export function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/** UUID with fallback for non-secure contexts (crypto.randomUUID needs HTTPS/localhost). */
export function genId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return 'id-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
}

export function sanitizeUrl(url: string): string {
  if (!url) return '';
  const lower = url.toLowerCase().trim();
  if (lower.startsWith('http://') || lower.startsWith('https://') || lower.startsWith('data:image/')) {
    return url;
  }
  return '';
}

export function formatMoney(n: number): string {
  return n.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function formatPercent(n: number): string {
  return (n * 100).toFixed(2) + '%';
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

export function formatToolResult(name: string, result: unknown): string {
  if (result === null || result === undefined) return '（无结果）';
  if (typeof result === 'string') {
    try {
      return formatToolResult(name, JSON.parse(result));
    } catch {
      return result;
    }
  }
  if (Array.isArray(result)) {
    if (result.length === 0) return '（空列表）';
    const keys = Object.keys(result[0] || {});
    if (keys.length === 0) return JSON.stringify(result);
    let html = '<table><thead><tr>';
    for (const k of keys) {
      html += `<th>${escapeHtml(k)}</th>`;
    }
    html += '</tr></thead><tbody>';
    for (const row of result) {
      html += '<tr>';
      for (const k of keys) {
        let v = (row as Record<string, unknown>)[k];
        if (v === null || v === undefined) {
          html += '<td>-</td>';
        } else if (typeof v === 'number') {
          const formatted = Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
          html += `<td><span class="stat">${formatted}</span></td>`;
        } else {
          html += `<td>${escapeHtml(String(v))}</td>`;
        }
      }
      html += '</tr>';
    }
    html += '</tbody></table>';
    return html;
  }
  if (typeof result === 'object') {
    const obj = result as Record<string, unknown>;
    // Image generation result
    if (obj.images && Array.isArray(obj.images) && obj.images.length > 0) {
      const imgs = obj.images as Array<{ url: string; task_id?: string }>;
      return imgs.map(img => {
        const url = sanitizeUrl(img.url);
        if (!url) return '';
        return `<img src="${url}" alt="生成的图片" style="max-width:100%;border-radius:8px;margin:8px 0" />`;
      }).join('');
    }
    // Video generation result
    if (obj.videos && Array.isArray(obj.videos) && obj.videos.length > 0) {
      const vids = obj.videos as Array<{ url: string; task_id?: string }>;
      return vids.map(vid => {
        const url = sanitizeUrl(vid.url);
        if (!url) return '';
        return `<video controls preload="metadata" style="max-width:100%;border-radius:8px;margin:8px 0"><source src="${url}"></video>`;
      }).join('');
    }
    // Holdings summary
    if (obj.buckets && Array.isArray(obj.buckets)) {
      return formatToolResult(name, obj.buckets);
    }
    // Key-value pairs
    let html = '<table><thead><tr><th>字段</th><th>值</th></tr></thead><tbody>';
    for (const [k, v] of Object.entries(obj)) {
      if (k === 'prompt' && typeof v === 'string' && v.length > 50) continue;
      let valueHtml: string;
      if (v === null || v === undefined) {
        valueHtml = '-';
      } else if (Array.isArray(v)) {
        // Recursively format nested arrays
        valueHtml = formatToolResult(name, v);
      } else if (typeof v === 'object') {
        // Recursively format nested objects
        valueHtml = formatToolResult(name, v);
      } else if (typeof v === 'number') {
        const formatted = Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2);
        valueHtml = `<span class="stat">${formatted}</span>`;
      } else {
        valueHtml = escapeHtml(String(v));
      }
      html += `<tr><td>${escapeHtml(k)}</td><td>${valueHtml}</td></tr>`;
    }
    html += '</tbody></table>';
    return html;
  }
  return escapeHtml(String(result));
}
