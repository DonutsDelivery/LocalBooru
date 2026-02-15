import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // Use relative paths for Electron
  server: {
    port: 5210, // LocalBooru dev port (avoid OpenDAW on 5173-5199)
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8790',
        changeOrigin: true,
      },
      '/thumbnails': {
        target: 'http://127.0.0.1:8790',
        changeOrigin: true,
      },
    },
  },
})
