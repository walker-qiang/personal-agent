import { marked } from 'marked';

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

  return html;
}