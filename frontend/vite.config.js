// frontend/vite.config.js
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
})
