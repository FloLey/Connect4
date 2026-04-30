// frontend/vite.config.js
/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Needed for Docker port mapping
    port: 5173,
    watch: {
      usePolling: true, // <--- CRITICAL: Enables Hot Reload in Docker/WSL
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.js'],
    css: false,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      // Gate covers everything in src/ that ships to users. Tests + the App /
      // main entry (just provider wiring + constants) are excluded.
      include: ['src/**/*.{js,jsx}'],
      exclude: [
        'src/**/__tests__/**',
        'src/test/**',
        'src/main.jsx',
        'src/App.jsx',
        'src/constants.js',
      ],
      thresholds: {
        lines: 70,
        functions: 70,
        statements: 70,
        branches: 60,
      },
    },
  },
})
