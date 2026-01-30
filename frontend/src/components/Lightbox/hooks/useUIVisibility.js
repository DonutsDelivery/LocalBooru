import { useCallback, useState, useRef, useEffect } from 'react'

/**
 * Hook for managing UI visibility (auto-hide) and fullscreen state
 */
export function useUIVisibility(containerRef) {
  const [showUI, setShowUI] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const hideUITimeout = useRef(null)

  // Auto-hide UI after inactivity
  const resetHideTimer = useCallback(() => {
    setShowUI(true)
    if (hideUITimeout.current) {
      clearTimeout(hideUITimeout.current)
    }
    hideUITimeout.current = setTimeout(() => {
      setShowUI(false)
    }, 3000)
  }, [])

  // Start hide timer on mount, clear on unmount
  useEffect(() => {
    resetHideTimer()
    return () => {
      if (hideUITimeout.current) {
        clearTimeout(hideUITimeout.current)
      }
    }
  }, [resetHideTimer])

  // Handle mouse movement to show UI
  const handleMouseMove = useCallback(() => {
    resetHideTimer()
  }, [resetHideTimer])

  // Fullscreen toggle handler
  const handleToggleFullscreen = useCallback(async () => {
    if (!containerRef.current) return

    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen()
      } else {
        await document.exitFullscreen()
      }
    } catch (err) {
      console.error('Fullscreen error:', err)
    }
  }, [containerRef])

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  return {
    showUI,
    isFullscreen,
    resetHideTimer,
    handleMouseMove,
    handleToggleFullscreen
  }
}
