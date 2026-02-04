import { useCallback, useState, useRef, useEffect } from 'react'
import { isTauri, videoControlAPI } from '../../../tauriAPI'

/**
 * Hook for managing video playback through Tauri/GStreamer backend
 *
 * This provides native GStreamer-based video control as an alternative to HTML5 video.
 * It's particularly useful for:
 * - VFR (Variable Frame Rate) videos that HTML5 struggles with
 * - Better seeking accuracy
 * - Frame-by-frame navigation
 * - Hardware-accelerated playback
 *
 * Supports two update modes:
 * 1. Event streaming (preferred): Real-time updates via Tauri events
 * 2. Polling (fallback): Periodic position queries
 */
export function useTauriVideoControls(videoPath, options = {}) {
  const {
    autoInit = false,
    pollInterval = 100, // ms between position updates (fallback mode)
    useEventStream = true, // Use event streaming instead of polling
    onStateChange = null,
    onPositionChange = null,
    onError = null,
    onEndOfStream = null
  } = options

  // Player state
  const [isInitialized, setIsInitialized] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1.0)
  const [isMuted, setIsMuted] = useState(false)
  const [playbackRate, setPlaybackRate] = useState(1.0)
  const [playerState, setPlayerState] = useState('Idle')
  const [error, setError] = useState(null)
  const [frameInfo, setFrameInfo] = useState(null)
  const [seekMode, setSeekModeState] = useState('accurate')

  // Refs
  const pollIntervalRef = useRef(null)
  const isPollingRef = useRef(false)
  const lastPathRef = useRef(null)
  const eventUnsubscribeRef = useRef(null)
  const isEventStreamingRef = useRef(false)

  // Check if Tauri is available
  const isTauriAvailable = isTauri()

  // Initialize the VFR player
  const initialize = useCallback(async () => {
    if (!isTauriAvailable) {
      setError('Tauri video API not available')
      return false
    }

    try {
      await videoControlAPI.initVfrPlayer()
      setIsInitialized(true)
      setError(null)
      return true
    } catch (e) {
      setError(e.message || 'Failed to initialize video player')
      if (onError) onError(e)
      return false
    }
  }, [isTauriAvailable, onError])

  // Cleanup the player
  const cleanup = useCallback(async () => {
    if (!isTauriAvailable || !isInitialized) return

    // Stop updates (both polling and event streaming)
    await stopUpdates()

    try {
      await videoControlAPI.cleanupVfr()
    } catch (e) {
      console.warn('[TauriVideo] cleanup error:', e)
    }

    setIsInitialized(false)
    setIsPlaying(false)
    setIsPaused(false)
    setCurrentTime(0)
    setDuration(0)
    setPlayerState('Idle')
    lastPathRef.current = null
  }, [isTauriAvailable, isInitialized, stopUpdates])

  // Poll for position and state updates
  const startPolling = useCallback(() => {
    if (!isTauriAvailable || isPollingRef.current) return

    isPollingRef.current = true
    pollIntervalRef.current = setInterval(async () => {
      try {
        // Get position
        const pos = await videoControlAPI.getPositionVfr()
        setCurrentTime(pos)
        if (onPositionChange) onPositionChange(pos)

        // Get state
        const state = await videoControlAPI.getStateVfr()
        setPlayerState(state)

        // Update playing/paused state based on player state
        const playing = state === 'Playing'
        const paused = state === 'Paused'
        setIsPlaying(playing)
        setIsPaused(paused)

        if (onStateChange) onStateChange(state)
      } catch (e) {
        console.warn('[TauriVideo] polling error:', e)
      }
    }, pollInterval)
  }, [isTauriAvailable, pollInterval, onPositionChange, onStateChange])

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
      pollIntervalRef.current = null
    }
    isPollingRef.current = false
  }, [])

  // Start event streaming (preferred method)
  const startEventStream = useCallback(async () => {
    if (!isTauriAvailable || isEventStreamingRef.current) return

    try {
      // Start the event stream on the Rust side
      await videoControlAPI.startEventStream(pollInterval)

      // Subscribe to events
      const unsubscribe = await videoControlAPI.subscribeToEvents((event) => {
        switch (event.type) {
          case 'position':
            setCurrentTime(event.position_secs)
            setDuration(event.duration_secs)
            if (onPositionChange) onPositionChange(event.position_secs)
            break

          case 'state_changed':
            setPlayerState(event.state)
            const playing = event.state === 'Playing'
            const paused = event.state === 'Paused'
            setIsPlaying(playing)
            setIsPaused(paused)
            if (onStateChange) onStateChange(event.state)
            break

          case 'buffering':
            // Could add buffering state if needed
            break

          case 'end_of_stream':
            setIsPlaying(false)
            if (onEndOfStream) onEndOfStream()
            break

          case 'error':
            setError(event.message)
            if (onError) onError(new Error(event.message))
            break

          case 'seek_completed':
            setCurrentTime(event.position_secs)
            break

          case 'volume_changed':
            setVolume(event.volume)
            setIsMuted(event.muted)
            break

          case 'rate_changed':
            setPlaybackRate(event.rate)
            break

          case 'frame_info':
            setFrameInfo({
              is_vfr: event.is_vfr,
              average_fps: event.average_fps,
              container_fps: event.container_fps
            })
            break

          default:
            console.log('[TauriVideo] Unknown event:', event)
        }
      })

      eventUnsubscribeRef.current = unsubscribe
      isEventStreamingRef.current = true
    } catch (e) {
      console.error('[TauriVideo] startEventStream error:', e)
      // Fall back to polling if event streaming fails
      startPolling()
    }
  }, [isTauriAvailable, pollInterval, onPositionChange, onStateChange, onEndOfStream, onError, startPolling])

  // Stop event streaming
  const stopEventStream = useCallback(async () => {
    if (eventUnsubscribeRef.current) {
      eventUnsubscribeRef.current()
      eventUnsubscribeRef.current = null
    }
    isEventStreamingRef.current = false

    if (isTauriAvailable) {
      try {
        await videoControlAPI.stopEventStream()
      } catch (e) {
        console.warn('[TauriVideo] stopEventStream error:', e)
      }
    }
  }, [isTauriAvailable])

  // Start updates (event streaming or polling based on option)
  const startUpdates = useCallback(() => {
    if (useEventStream) {
      startEventStream()
    } else {
      startPolling()
    }
  }, [useEventStream, startEventStream, startPolling])

  // Stop updates
  const stopUpdates = useCallback(async () => {
    stopPolling()
    await stopEventStream()
  }, [stopPolling, stopEventStream])

  // Play a video
  const play = useCallback(async (path) => {
    if (!isTauriAvailable) return

    // Initialize if not already
    if (!isInitialized) {
      const success = await initialize()
      if (!success) return
    }

    const videoUri = path || videoPath
    if (!videoUri) {
      setError('No video path provided')
      return
    }

    try {
      setError(null)
      await videoControlAPI.playVfr(videoUri)
      lastPathRef.current = videoUri

      // Get duration after starting playback
      setTimeout(async () => {
        const dur = await videoControlAPI.getDurationVfr()
        setDuration(dur)

        // Get frame info
        const info = await videoControlAPI.getFrameInfoVfr()
        setFrameInfo(info)
      }, 100)

      setIsPlaying(true)
      setIsPaused(false)
      startUpdates()
    } catch (e) {
      setError(e.message || 'Failed to play video')
      if (onError) onError(e)
    }
  }, [isTauriAvailable, isInitialized, initialize, videoPath, startUpdates, onError])

  // Pause playback
  const pause = useCallback(async () => {
    if (!isTauriAvailable || !isInitialized) return

    try {
      await videoControlAPI.pauseVfr()
      setIsPlaying(false)
      setIsPaused(true)
    } catch (e) {
      console.error('[TauriVideo] pause error:', e)
    }
  }, [isTauriAvailable, isInitialized])

  // Resume playback
  const resume = useCallback(async () => {
    if (!isTauriAvailable || !isInitialized) return

    try {
      await videoControlAPI.resumeVfr()
      setIsPlaying(true)
      setIsPaused(false)
    } catch (e) {
      console.error('[TauriVideo] resume error:', e)
    }
  }, [isTauriAvailable, isInitialized])

  // Toggle play/pause
  const togglePlayPause = useCallback(async () => {
    if (isPlaying) {
      await pause()
    } else {
      await resume()
    }
  }, [isPlaying, pause, resume])

  // Stop playback
  const stop = useCallback(async () => {
    if (!isTauriAvailable || !isInitialized) return

    await stopUpdates()

    try {
      await videoControlAPI.stopVfr()
      setIsPlaying(false)
      setIsPaused(false)
      setCurrentTime(0)
    } catch (e) {
      console.error('[TauriVideo] stop error:', e)
    }
  }, [isTauriAvailable, isInitialized, stopUpdates])

  // Seek to position
  const seek = useCallback(async (positionSecs) => {
    if (!isTauriAvailable || !isInitialized) return

    try {
      await videoControlAPI.seekVfr(positionSecs)
      setCurrentTime(positionSecs)
    } catch (e) {
      console.error('[TauriVideo] seek error:', e)
    }
  }, [isTauriAvailable, isInitialized])

  // Seek with specific mode
  const seekWithMode = useCallback(async (positionSecs, mode) => {
    if (!isTauriAvailable || !isInitialized) return

    try {
      await videoControlAPI.seekVfrWithMode(positionSecs, mode)
      setCurrentTime(positionSecs)
    } catch (e) {
      console.error('[TauriVideo] seekWithMode error:', e)
    }
  }, [isTauriAvailable, isInitialized])

  // Seek forward/backward by delta seconds
  const seekDelta = useCallback(async (deltaSecs) => {
    const newTime = Math.max(0, Math.min(duration, currentTime + deltaSecs))
    await seek(newTime)
  }, [currentTime, duration, seek])

  // Step forward one frame
  const stepFrame = useCallback(async () => {
    if (!isTauriAvailable || !isInitialized) return

    try {
      await videoControlAPI.stepFrameVfr()
      // Update position after step
      const pos = await videoControlAPI.getPositionVfr()
      setCurrentTime(pos)
    } catch (e) {
      console.error('[TauriVideo] stepFrame error:', e)
    }
  }, [isTauriAvailable, isInitialized])

  // Set volume
  const setVolumeValue = useCallback(async (value) => {
    if (!isTauriAvailable || !isInitialized) return

    const clampedValue = Math.max(0, Math.min(1, value))
    try {
      await videoControlAPI.setVolumeVfr(clampedValue)
      setVolume(clampedValue)
    } catch (e) {
      console.error('[TauriVideo] setVolume error:', e)
    }
  }, [isTauriAvailable, isInitialized])

  // Toggle mute
  const toggleMute = useCallback(async () => {
    if (!isTauriAvailable || !isInitialized) return

    try {
      await videoControlAPI.setMutedVfr(!isMuted)
      setIsMuted(!isMuted)
    } catch (e) {
      console.error('[TauriVideo] toggleMute error:', e)
    }
  }, [isTauriAvailable, isInitialized, isMuted])

  // Set playback rate
  const setRate = useCallback(async (rate) => {
    if (!isTauriAvailable || !isInitialized) return

    try {
      await videoControlAPI.setRateVfr(rate)
      setPlaybackRate(rate)
    } catch (e) {
      console.error('[TauriVideo] setRate error:', e)
    }
  }, [isTauriAvailable, isInitialized])

  // Increase/decrease playback rate
  const increaseRate = useCallback(async () => {
    const newRate = Math.min(4.0, playbackRate + 0.25)
    await setRate(newRate)
  }, [playbackRate, setRate])

  const decreaseRate = useCallback(async () => {
    const newRate = Math.max(0.25, playbackRate - 0.25)
    await setRate(newRate)
  }, [playbackRate, setRate])

  const resetRate = useCallback(async () => {
    await setRate(1.0)
  }, [setRate])

  // Set seek mode
  const setSeekMode = useCallback(async (mode) => {
    if (!isTauriAvailable || !isInitialized) return

    try {
      await videoControlAPI.setSeekModeVfr(mode)
      setSeekModeState(mode)
    } catch (e) {
      console.error('[TauriVideo] setSeekMode error:', e)
    }
  }, [isTauriAvailable, isInitialized])

  // Analyze video for VFR
  const analyzeVideo = useCallback(async (path) => {
    if (!isTauriAvailable) return null

    try {
      return await videoControlAPI.analyzeVfr(path || videoPath)
    } catch (e) {
      console.error('[TauriVideo] analyzeVideo error:', e)
      return null
    }
  }, [isTauriAvailable, videoPath])

  // Get stream info
  const getStreamInfo = useCallback(async () => {
    if (!isTauriAvailable || !isInitialized) return null

    try {
      return await videoControlAPI.getStreamInfoVfr()
    } catch (e) {
      console.error('[TauriVideo] getStreamInfo error:', e)
      return null
    }
  }, [isTauriAvailable, isInitialized])

  // Auto-initialize on mount if requested
  useEffect(() => {
    if (autoInit && isTauriAvailable && !isInitialized) {
      initialize()
    }
  }, [autoInit, isTauriAvailable, isInitialized, initialize])

  // Auto-play when video path changes (if already initialized and playing)
  useEffect(() => {
    if (isInitialized && videoPath && videoPath !== lastPathRef.current) {
      play(videoPath)
    }
  }, [isInitialized, videoPath, play])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPolling()
      if (eventUnsubscribeRef.current) {
        eventUnsubscribeRef.current()
      }
      videoControlAPI.stopEventStream().catch(() => {})
      if (isInitialized) {
        videoControlAPI.cleanupVfr().catch(() => {})
      }
    }
  }, [isInitialized, stopPolling])

  return {
    // State
    isAvailable: isTauriAvailable,
    isInitialized,
    isPlaying,
    isPaused,
    currentTime,
    duration,
    volume,
    isMuted,
    playbackRate,
    playerState,
    error,
    frameInfo,
    seekMode,

    // Control functions
    initialize,
    cleanup,
    play,
    pause,
    resume,
    togglePlayPause,
    stop,
    seek,
    seekWithMode,
    seekDelta,
    stepFrame,
    setVolume: setVolumeValue,
    toggleMute,
    setRate,
    increaseRate,
    decreaseRate,
    resetRate,
    setSeekMode,
    analyzeVideo,
    getStreamInfo,

    // For manual update control
    startPolling,
    stopPolling,
    startEventStream,
    stopEventStream,
    startUpdates,
    stopUpdates
  }
}

export default useTauriVideoControls
