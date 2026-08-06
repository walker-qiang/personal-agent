/// <reference types="vitest" />
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    // Output to react-app subdirectory to avoid overwriting
    outDir: '../server/static/react-app',
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
      '/mcp': 'http://127.0.0.1:7101',
    },
  },
  test: {
    globals: true,
    environment: 'happy-dom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
