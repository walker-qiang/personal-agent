import { marked } from 'marked';
import DOMPurify from 'dompurify';

// Configure marked.js — matches original HTML frontend settings
marked.setOptions({
  breaks: true,
  gfm: true,
});

// Allow the extra tags/attrs produced by renderMarkdown / formatToolResult,
// on top of DOMPurify's default allow-list.
const PURIFY_CONFIG = {
  ADD_TAGS: ['video', 'source'],
  ADD_ATTR: ['controls', 'preload', 'target', 'rel'],
};

export function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html, PURIFY_CONFIG) as string;
}

export function renderMarkdown(text: string): string {
  if (!text) return '';
  let html = marked.parse(text, { async: false }) as string;

  // Replace video placeholder images with <video> elements
  html = html.replace(
    /<img\s+[^>]*alt="([^"]*(?:视频|video)[^"]*)"[^>]*src="([^"]*)"[^>]*\/?>/gi,
    (_match, alt, src) => {
      if (!src) return _match;
      return `<video controls preload="metadata" style="max-width:100%;border-radius:8px" title="${alt}"><source src="${src}"></video>`;
    },
  );
  html = html.replace(
    /<img\s+[^>]*src="([^"]*\.(?:mp4|webm|mov|avi|mkv)[^"]*)"[^>]*\/?>/gi,
    (_match, src) => {
      return `<video controls preload="metadata" style="max-width:100%;border-radius:8px"><source src="${src}"></video>`;
    },
  );

  return sanitizeHtml(html);
}
