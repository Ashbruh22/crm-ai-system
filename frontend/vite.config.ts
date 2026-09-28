import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
  // The app calls the ML service directly at VITE_API_URL, and the service
  // allows this origin via ALLOWED_ORIGINS. The old dev proxy pointed at
  // nginx on :80, which is not part of the hosted topology.
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/react/') || id.includes('node_modules/react-dom/') || id.includes('node_modules/react-router-dom/')) return 'vendor';
          if (id.includes('node_modules/recharts/')) return 'charts';
          if (id.includes('node_modules/@tanstack/react-query/')) return 'query';
        }
      }
    }
  }
})
