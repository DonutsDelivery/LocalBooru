import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Debug: Log when main.jsx executes
console.log('[LocalBooru] main.jsx executing...')

try {
  const root = document.getElementById('root')
  console.log('[LocalBooru] Found root element:', root)

  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
  console.log('[LocalBooru] React render called')
} catch (error) {
  console.error('[LocalBooru] Failed to render:', error)
  // Show error on screen if React fails to mount
  document.getElementById('root').innerHTML = `
    <div style="color: white; padding: 20px; font-family: monospace;">
      <h1>Failed to load</h1>
      <pre>${error.stack || error.message}</pre>
    </div>
  `
}
