import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

const __dirname = dirname(fileURLToPath(import.meta.url));

// https://vite.dev/config/
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        chat: resolve(__dirname, 'chat.html'),
      },
    },
  },
  plugins: [react()],
  server: {
    proxy: {
      '/proxy-api/v1/chat/completions': {
        target: 'http://localhost:3001',
        // target: 'https://ztrxxhzxdt3bn6-3000.proxy.runpod.net',
        rewrite: (path) => path.replace(/^\/proxy-api/, ''),
      },
      '/proxy-api': {
        target: 'http://localhost:3001',
        rewrite: (path) => path.replace(/^\/proxy-api/, ''),
      },
    },
  },
});
