import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        // Proxy to Hugging Face Space API; fallback to local dev server if env var not set
        target: process.env.VITE_API_URL || 'https://huggingface.co/spaces/LalithyaKonne/ai-sepsis-detection',
        changeOrigin: true,
        // Rewrite the path if necessary (remove /api prefix)
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
