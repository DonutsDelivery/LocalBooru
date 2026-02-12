import { useCallback, useState, useRef, useEffect } from 'react'
import { getVideoPlaybackConfig } from '../../../api'

/**
 * Hook for auto-advancing to next video when current one ends.
 * Shows a countdown overlay with cancel/advance-now actions.
 */
export function useAutoAdvance(mediaRef, { onNav, currentIndex, totalImages, isVideoFile }) {
  const [countdown, setCountdown] = useState(null) // null = not counting, number = seconds left
  const [config, setConfig] = useState(null)
  const countdownTimerRef = useRef(null)
  const configLoadedRef = useRef(false)

  // Load config once
  useEffect(() => {
    if (configLoadedRef.current) return
    configLoadedRef.current = true
    getVideoPlaybackConfig()
      .then(setConfig)
      .catch(() => setConfig({ auto_advance_enabled: false, auto_advance_delay: 5 }))
  }, [])

  const isEnabled = config?.auto_advance_enabled && isVideoFile
  const isLastItem = currentIndex >= totalImages - 1

  // Clear countdown timer
  const clearCountdown = useCallback(() => {
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current)
      countdownTimerRef.current = null
    }
    setCountdown(null)
  }, [])

  // Cancel countdown
  const cancelCountdown = useCallback(() => {
    clearCountdown()
    // Re-enable loop on the video so it loops instead
    if (mediaRef.current) {
      mediaRef.current.loop = true
    }
  }, [clearCountdown, mediaRef])

  // Advance now (skip countdown)
  const advanceNow = useCallback(() => {
    clearCountdown()
    onNav(1)
  }, [clearCountdown, onNav])

  // Start countdown when video ends
  const handleVideoEnded = useCallback(() => {
    if (!isEnabled || isLastItem) return

    const delay = config?.auto_advance_delay || 5
    setCountdown(delay)

    countdownTimerRef.current = setInterval(() => {
      setCountdown(prev => {
        if (prev === null) return null
        if (prev <= 1) {
          clearInterval(countdownTimerRef.current)
          countdownTimerRef.current = null
          // Navigate to next
          onNav(1)
          return null
        }
        return prev - 1
      })
    }, 1000)
  }, [isEnabled, isLastItem, config?.auto_advance_delay, onNav])

  // Pause countdown when tab is hidden
  useEffect(() => {
    if (countdown === null) return

    const handleVisibilityChange = () => {
      if (document.hidden && countdownTimerRef.current) {
        clearInterval(countdownTimerRef.current)
        countdownTimerRef.current = null
      } else if (!document.hidden && countdown !== null) {
        // Resume countdown
        countdownTimerRef.current = setInterval(() => {
          setCountdown(prev => {
            if (prev === null) return null
            if (prev <= 1) {
              clearInterval(countdownTimerRef.current)
              countdownTimerRef.current = null
              onNav(1)
              return null
            }
            return prev - 1
          })
        }, 1000)
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [countdown, onNav])

  // Listen for video ended event
  useEffect(() => {
    const video = mediaRef.current
    if (!video || !isEnabled) return

    // Disable loop when auto-advance is enabled (so ended event fires)
    video.loop = false

    video.addEventListener('ended', handleVideoEnded)
    return () => {
      video.removeEventListener('ended', handleVideoEnded)
    }
  }, [mediaRef, isEnabled, handleVideoEnded])

  // Clear countdown on navigation or unmount
  useEffect(() => {
    return clearCountdown
  }, [currentIndex, clearCountdown])

  return {
    countdown,
    isEnabled,
    config,
    setConfig,
    cancelCountdown,
    advanceNow,
    clearCountdown,
  }
}
