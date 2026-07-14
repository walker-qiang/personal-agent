import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../server/static',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:7101',
      '/chat': 'http://127.0.0.1:7101',
      '/healthz': 'http://127.0.0.1:7101',
      '/tools': 'http://127.0.0.1:7101',
      '/reset': 'http://127.0.0.1:7101',
      '/sessions': 'http://127.0.0.1:7101',
      '/skills': 'http://127.0.0.1:7101',
    },
  },
});