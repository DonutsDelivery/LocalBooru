import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

import { getDevProxyTarget } from './devProxy.js'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  plugins: [react()],
  base: './', // Use relative paths for Electron
  server: {
    port: 5210, // LocalBooru dev port (avoid OpenDAW on 5173-5199)
    strictPort: true,
    proxy: {
      '/health': {
        target: command === 'serve' ? getDevProxyTarget(process.env.VITE_LOCALBOORU_PORT) : undefined,
        changeOrigin: true,
      },
      '/api': {
        target: command === 'serve' ? getDevProxyTarget(process.env.VITE_LOCALBOORU_PORT) : undefined,
        changeOrigin: true,
      },
      '/thumbnails': {
        target: command === 'serve' ? getDevProxyTarget(process.env.VITE_LOCALBOORU_PORT) : undefined,
        changeOrigin: true,
      },
    },
  },
}))
