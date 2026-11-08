import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      // LicenseNotices imports the repo-root LICENSE via ?raw; Vite checks the id with
      // its ?raw suffix, so a file entry would not match and the repo root is allowed instead.
      allow: ['..'],
    },
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.component.test.jsx'],
    restoreMocks: true,
  },
})
