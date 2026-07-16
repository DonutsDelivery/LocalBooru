import { useCallback, useState, useRef, useEffect } from 'react'
import { getVideoPlaybackConfig } from '../../../api'

/**
 * Hook for auto-advancing to next video when current one ends.
 * Shows a countdown overlay with cancel/advance-now actions.
 */
export function useAutoAdvance(mediaRef, {
  onNav,
  currentIndex,
  totalImages,
  isVideoFile,
  streamTransitioningRef,
  getCurrentAbsoluteTime,
  durationRef,
  isStreaming = false,
  isBuffering = false,
  pendingSeek = null,
  bufferedEnd = 0,
}) {
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
    // Ignore spurious 'ended' events during stream transitions (e.g. HLS source
    // being destroyed and re-created when seeking in a transcode stream)
    if (streamTransitioningRef?.current) return
    if (isBuffering || pendingSeek !== null) {
      console.log('[Auto-advance] Ignoring ended event while stream is buffering/seeking', {
        isBuffering,
        pendingSeek,
      })
      return
    }

    const video = mediaRef.current
    const knownDuration = durationRef?.current || video?.duration || 0
    const currentTime = getCurrentAbsoluteTime ? getCurrentAbsoluteTime() : (video?.currentTime || 0)

    if (isStreaming && (!knownDuration || !isFinite(knownDuration))) {
      console.log('[Auto-advance] Ignoring streaming ended event without known source duration', {
        mediaDuration: video?.duration,
        currentTime,
        readyState: video?.readyState,
        networkState: video?.networkState,
      })
      return
    }

    if (isStreaming && knownDuration && isFinite(knownDuration)) {
      const remaining = knownDuration - currentTime
      const bufferedRemaining = bufferedEnd ? knownDuration - bufferedEnd : Infinity

      if (remaining > 0.75 || bufferedRemaining > 0.75 || video?.readyState < 2) {
        console.log('[Auto-advance] Ignoring streaming ended event before source end', {
          currentTime,
          knownDuration,
          remaining,
          bufferedEnd,
          bufferedRemaining,
          mediaDuration: video?.duration,
          readyState: video?.readyState,
          networkState: video?.networkState,
        })
        return
      }
    }

    // HLS streams can emit `ended` when the currently generated playlist window
    // runs out, even though the source video has not reached its real end yet.
    if (knownDuration && isFinite(knownDuration)) {
      const remaining = knownDuration - currentTime
      if (remaining > 2) {
        console.log('[Auto-advance] Ignoring premature ended event', {
          currentTime,
          knownDuration,
          remaining,
          mediaDuration: video?.duration,
        })
        return
      }
    }

    const delay = config?.auto_advance_delay || 5
    clearCountdown()
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
  }, [
    isEnabled,
    isLastItem,
    config?.auto_advance_delay,
    clearCountdown,
    onNav,
    mediaRef,
    durationRef,
    getCurrentAbsoluteTime,
    streamTransitioningRef,
    isStreaming,
    isBuffering,
    pendingSeek,
    bufferedEnd,
  ])

  // Buffering/seeking can arrive just after a spurious HLS `ended` event.
  // Cancel an already-started countdown as soon as playback proves it is not
  // really finished.
  useEffect(() => {
    if (countdown === null) return
    if (streamTransitioningRef?.current || isBuffering || pendingSeek !== null) {
      console.log('[Auto-advance] Cancelling countdown during stream buffering/seeking', {
        isBuffering,
        pendingSeek,
      })
      clearCountdown()
    }
  }, [countdown, streamTransitioningRef, isBuffering, pendingSeek, clearCountdown])

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

  // Keep native media looping disabled. WebKit's GStreamer backend implements
  // the loop attribute with segmented seeks, which can restart large HTTP
  // Range-backed videos whenever the source advances to the next byte range.
  // Loop explicitly only after the real ended event instead.
  useEffect(() => {
    const video = mediaRef.current
    if (!video) return

    video.loop = false

    const handleEnded = () => {
      if (isEnabled) {
        handleVideoEnded()
        return
      }
      video.currentTime = 0
      video.play().catch(() => {})
    }

    const cancelStreamingCountdown = (event) => {
      if (!isStreaming || countdownTimerRef.current === null) return
      console.log(`[Auto-advance] Cancelling countdown on ${event.type}`)
      clearCountdown()
    }

    video.addEventListener('ended', handleEnded)
    video.addEventListener('waiting', cancelStreamingCountdown)
    video.addEventListener('stalled', cancelStreamingCountdown)
    video.addEventListener('seeking', cancelStreamingCountdown)
    return () => {
      video.removeEventListener('ended', handleEnded)
      video.removeEventListener('waiting', cancelStreamingCountdown)
      video.removeEventListener('stalled', cancelStreamingCountdown)
      video.removeEventListener('seeking', cancelStreamingCountdown)
    }
  }, [mediaRef, isEnabled, handleVideoEnded, isStreaming, clearCountdown])

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
