import { useState, useEffect, useCallback, useRef } from 'react'
import { getFamilyModeStatus, lockFamilyMode, unlockFamilyMode, subscribeToLibraryEvents } from '../api'
import './FamilyModeLock.css'

function FamilyModeLock({ onStateChange, onLockChange }) {
  const [status, setStatus] = useState(null)
  const [showPinInput, setShowPinInput] = useState(false)
  const [pin, setPin] = useState('')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const inputRef = useRef(null)
  const onStateChangeRef = useRef(onStateChange)
  const onLockChangeRef = useRef(onLockChange)
  onStateChangeRef.current = onStateChange
  onLockChangeRef.current = onLockChange

  const fetchStatus = useCallback(async () => {
    try {
      const data = await getFamilyModeStatus()
      setStatus(data)
      if (onLockChangeRef.current) onLockChangeRef.current(data.is_locked)
    } catch (e) {
      // Family mode not configured or backend not ready
    }
  }, [])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  // Listen for SSE family_mode events
  useEffect(() => {
    const unsubscribe = subscribeToLibraryEvents((event) => {
      if (event.type === 'family_mode') {
        const isLocked = event.data?.is_locked ?? event.is_locked
        setStatus(prev => prev ? { ...prev, is_locked: isLocked } : prev)
        if (onLockChangeRef.current) onLockChangeRef.current(isLocked)
        if (onStateChangeRef.current) onStateChangeRef.current()
      }
    })
    return unsubscribe
  }, [])

  // Focus input when PIN form shows
  useEffect(() => {
    if (showPinInput && inputRef.current) {
      inputRef.current.focus()
    }
  }, [showPinInput])

  if (!status || !status.enabled) return null

  const handleLock = async () => {
    setLoading(true)
    try {
      await lockFamilyMode()
      setStatus(prev => ({ ...prev, is_locked: true }))
      if (onLockChange) onLockChange(true)
      if (onStateChange) onStateChange()
    } catch (e) {
      console.error('Failed to lock:', e)
    }
    setLoading(false)
  }

  const handleUnlock = async (e) => {
    e?.preventDefault()
    if (!pin) return
    setLoading(true)
    setError(null)
    try {
      await unlockFamilyMode(pin)
      setStatus(prev => ({ ...prev, is_locked: false }))
      setShowPinInput(false)
      setPin('')
      if (onLockChange) onLockChange(false)
      if (onStateChange) onStateChange()
    } catch (e) {
      setError('Wrong PIN')
    }
    setLoading(false)
  }

  const cancelUnlock = () => {
    setShowPinInput(false)
    setPin('')
    setError(null)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') cancelUnlock()
  }

  if (status.is_locked && showPinInput) {
    return (
      <div className="family-mode-lock">
        <form className="family-pin-form" onSubmit={handleUnlock}>
          <input
            ref={inputRef}
            type="password"
            placeholder="PIN"
            value={pin}
            onChange={(e) => { setPin(e.target.value); setError(null) }}
            onKeyDown={handleKeyDown}
            maxLength={20}
          />
          <button type="submit" className="pin-unlock-btn" disabled={loading || !pin}>
            {loading ? '...' : 'Unlock'}
          </button>
          <button type="button" className="pin-cancel-btn" onClick={cancelUnlock}>
            <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        </form>
        {error && <span className="family-pin-error">{error}</span>}
      </div>
    )
  }

  return (
    <div className="family-mode-lock">
      {status.is_locked ? (
        <button
          className="family-lock-btn locked"
          onClick={() => setShowPinInput(true)}
          disabled={loading}
          title="Family mode is active — click to unlock"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
          </svg>
          <span>Family Mode</span>
        </button>
      ) : (
        <button
          className="family-lock-btn unlocked"
          onClick={handleLock}
          disabled={loading}
          title="Click to enable family mode"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
            <path d="M7 11V7a5 5 0 0 1 9.9-1"/>
          </svg>
        </button>
      )}
    </div>
  )
}

export default FamilyModeLock
