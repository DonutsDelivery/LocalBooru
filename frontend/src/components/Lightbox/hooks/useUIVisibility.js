import { useCallback, useState, useRef, useEffect } from 'react'

export function shouldRevealOnly(isUIVisible) {
  return !isUIVisible
}

// Detect if the native Fullscreen API is available and functional.
// Android WebView often has requestFullscreen defined but it silently fails
// or is blocked by the Tauri WebView wrapper.
function hasNativeFullscreen() {
  const el = document.documentElement
  if (!el.requestFullscreen && !el.webkitRequestFullscreen) return false
  // Android WebView in Tauri: the API exists but doesn't work
  if (/Android/i.test(navigator.userAgent) && window.__TAURI_INTERNALS__) return false
  return true
}

/**
 * Hook for managing UI visibility (auto-hide) and fullscreen state
 */
export function useUIVisibility(containerRef) {
  const [showUI, setShowUI] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const hideUITimeout = useRef(null)
  const usingCssFullscreen = useRef(!hasNativeFullscreen())
  const showUIRef = useRef(true)
  const consumeRevealTapRef = useRef(false)

  // Auto-hide UI after inactivity
  const resetHideTimer = useCallback(() => {
    showUIRef.current = true
    setShowUI(true)
    if (hideUITimeout.current) {
      clearTimeout(hideUITimeout.current)
    }
    hideUITimeout.current = setTimeout(() => {
      showUIRef.current = false
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

  const handleTouchInteractionStart = useCallback(() => {
    consumeRevealTapRef.current = shouldRevealOnly(showUIRef.current)
    resetHideTimer()
  }, [resetHideTimer])

  const consumeRevealTap = useCallback(() => {
    if (!consumeRevealTapRef.current) return false
    consumeRevealTapRef.current = false
    return true
  }, [])

  const cancelRevealTap = useCallback(() => {
    consumeRevealTapRef.current = false
  }, [])

  const toggleCssFullscreen = useCallback(() => {
    setIsFullscreen(prev => {
      const next = !prev
      if (window.AndroidImmersive) {
        next ? window.AndroidImmersive.enter() : window.AndroidImmersive.exit()
      }
      return next
    })
  }, [])

  // Fullscreen toggle handler
  // On platforms where the native Fullscreen API works (desktop browsers),
  // use it so the entire page goes fullscreen including sibling elements.
  // On Android WebView / Tauri Mobile, fall back to CSS-based fullscreen
  // which covers the title bar and uses the full viewport via position/z-index.
  const handleToggleFullscreen = useCallback(async () => {
    if (usingCssFullscreen.current) {
      toggleCssFullscreen()
      return
    }

    try {
      const el = containerRef.current || document.documentElement
      const fsElement = document.fullscreenElement ?? document.webkitFullscreenElement
      if (!fsElement) {
        if (el.requestFullscreen) {
          await el.requestFullscreen()
        } else if (el.webkitRequestFullscreen) {
          el.webkitRequestFullscreen()
        }
      } else {
        if (document.exitFullscreen) {
          await document.exitFullscreen()
        } else if (document.webkitExitFullscreen) {
          document.webkitExitFullscreen()
        }
      }
    } catch (err) {
      console.error('Fullscreen error:', err)
      // If the native API threw, switch to CSS fallback for this session
      usingCssFullscreen.current = true
      toggleCssFullscreen()
    }
  }, [containerRef, toggleCssFullscreen])

  // Exit immersive mode when component unmounts (lightbox closes)
  useEffect(() => {
    return () => {
      if (window.AndroidImmersive && window.AndroidImmersive.isActive()) {
        window.AndroidImmersive.exit()
      }
    }
  }, [])

  // Listen for fullscreen changes (only relevant for native fullscreen)
  useEffect(() => {
    if (usingCssFullscreen.current) return

    const handleFullscreenChange = () => {
      setIsFullscreen(!!(document.fullscreenElement ?? document.webkitFullscreenElement))
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange)
    }
  }, [])

  return {
    showUI,
    isFullscreen,
    resetHideTimer,
    handleMouseMove,
    handleTouchInteractionStart,
    consumeRevealTap,
    cancelRevealTap,
    handleToggleFullscreen
  }
}
