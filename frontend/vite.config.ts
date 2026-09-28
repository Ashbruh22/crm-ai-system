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
          if (
            id.includes('node_modules/react/') ||
            id.includes('node_modules/react-dom/') ||
            id.includes('node_modules/react-router-dom/')
          ) {
            return 'vendor';
          }
          if (id.includes('node_modules/@tanstack/react-query/')) return 'query';
          // recharts is deliberately NOT named here. Giving it a manual chunk
          // put a <link rel="modulepreload"> for 366 kB of charting in the
          // entry HTML, so every visitor downloaded it before seeing the
          // pipeline -- even though ScoreHistory is lazily imported. Left
          // unnamed, it stays inside that lazy chunk and loads on demand.
        }
      }
    }
  }
})
