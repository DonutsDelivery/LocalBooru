import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // Use relative paths for Electron
  server: {
    port: 5174, // Avoid conflict with other dev servers on 5173
  },
})
