import { useState, useEffect, useCallback } from 'react'
import './Toast.css'

// Global toast state - allows calling toast() from anywhere (including non-React code)
let toastListeners = []
let toastId = 0

export function toast(message, type = 'info', duration = 4000) {
  const id = ++toastId
  const t = { id, message, type, duration }
  toastListeners.forEach(fn => fn(t))
  return id
}

toast.success = (msg, duration) => toast(msg, 'success', duration)
toast.error = (msg, duration) => toast(msg, 'error', duration ?? 6000)
toast.warning = (msg, duration) => toast(msg, 'warning', duration)
toast.info = (msg, duration) => toast(msg, 'info', duration)

export default function ToastContainer() {
  const [toasts, setToasts] = useState([])

  useEffect(() => {
    const handler = (t) => {
      setToasts(prev => [...prev, { ...t, visible: true }])
      if (t.duration > 0) {
        setTimeout(() => dismissToast(t.id), t.duration)
      }
    }
    toastListeners.push(handler)
    return () => {
      toastListeners = toastListeners.filter(fn => fn !== handler)
    }
  }, [])

  const dismissToast = useCallback((id) => {
    setToasts(prev => prev.map(t => t.id === id ? { ...t, visible: false } : t))
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id))
    }, 300)
  }, [])

  if (toasts.length === 0) return null

  return (
    <div className="toast-container">
      {toasts.map(t => (
        <div
          key={t.id}
          className={`toast toast-${t.type}${t.visible ? ' toast-enter' : ' toast-exit'}`}
          onClick={() => dismissToast(t.id)}
        >
          <span className="toast-icon">
            {t.type === 'success' && '\u2713'}
            {t.type === 'error' && '\u2717'}
            {t.type === 'warning' && '\u26A0'}
            {t.type === 'info' && '\u2139'}
          </span>
          <span className="toast-message">{t.message}</span>
        </div>
      ))}
    </div>
  )
}
