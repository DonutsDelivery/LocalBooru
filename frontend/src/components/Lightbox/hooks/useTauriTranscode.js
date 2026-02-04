import { useCallback, useState, useRef, useEffect } from 'react'
import { isTauri, videoControlAPI } from '../../../tauriAPI'

/**
 * Hook for managing video transcoding through Tauri/GStreamer backend
 *
 * This provides on-the-fly transcoding for incompatible video formats.
 * Features:
 * - Hardware-accelerated encoding (NVENC, VA-API)
 * - Quality presets (low, medium, high, original)
 * - HLS streaming output for progressive playback
 * - Progress tracking
 */
export function useTauriTranscode() {
  // State
  const [capabilities, setCapabilities] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeStreams, setActiveStreams] = useState({})

  // Refs
  const progressPollRef = useRef({})
  const isMountedRef = useRef(true)

  // Check if Tauri is available
  const isTauriAvailable = isTauri()

  // Load transcoding capabilities
  const loadCapabilities = useCallback(async () => {
    if (!isTauriAvailable) return null

    try {
      const caps = await videoControlAPI.getTranscodeCapabilities()
      if (isMountedRef.current) {
        setCapabilities(caps)
      }
      return caps
    } catch (e) {
      console.error('[TauriTranscode] loadCapabilities error:', e)
      if (isMountedRef.current) {
        setError(e.message || 'Failed to load transcoding capabilities')
      }
      return null
    }
  }, [isTauriAvailable])

  // Check if a video needs transcoding
  const checkNeedsTranscode = useCallback(async (videoPath) => {
    if (!isTauriAvailable) return false

    try {
      return await videoControlAPI.checkTranscodeNeeded(videoPath)
    } catch (e) {
      console.error('[TauriTranscode] checkNeedsTranscode error:', e)
      return false
    }
  }, [isTauriAvailable])

  // Start transcoding a video
  const startTranscode = useCallback(async (sourcePath, quality = 'medium', startPosition = 0) => {
    if (!isTauriAvailable) {
      throw new Error('Tauri video API not available')
    }

    setIsLoading(true)
    setError(null)

    try {
      const result = await videoControlAPI.startTranscode(sourcePath, quality, startPosition)

      if (result && result.stream_id) {
        // Store active stream
        setActiveStreams(prev => ({
          ...prev,
          [result.stream_id]: {
            sourcePath,
            quality,
            startPosition,
            playlistPath: result.playlist_path,
            encoder: result.encoder,
            state: 'starting',
            progress: 0
          }
        }))

        // Start polling for progress
        pollProgress(result.stream_id)

        return result
      }

      throw new Error('Invalid transcode response')
    } catch (e) {
      setError(e.message || 'Failed to start transcoding')
      throw e
    } finally {
      setIsLoading(false)
    }
  }, [isTauriAvailable])

  // Poll for transcoding progress
  const pollProgress = useCallback((streamId) => {
    if (!isTauriAvailable || progressPollRef.current[streamId]) return

    const poll = async () => {
      if (!isMountedRef.current) return

      try {
        const progress = await videoControlAPI.getTranscodeProgress(streamId)

        if (!isMountedRef.current) return

        if (progress) {
          setActiveStreams(prev => ({
            ...prev,
            [streamId]: {
              ...prev[streamId],
              state: progress.state,
              progress: progress.progress_percent,
              segmentsReady: progress.segments_ready,
              encodingSpeed: progress.encoding_speed,
              eta: progress.eta_secs
            }
          }))

          // Stop polling if completed or errored
          if (progress.state === 'Completed' || progress.state === 'Stopped' || progress.state.startsWith('Error')) {
            clearInterval(progressPollRef.current[streamId])
            delete progressPollRef.current[streamId]
            return
          }
        }
      } catch (e) {
        console.warn('[TauriTranscode] progress poll error:', e)
      }
    }

    // Poll every 500ms
    progressPollRef.current[streamId] = setInterval(poll, 500)
    // Initial poll
    poll()
  }, [isTauriAvailable])

  // Check if a stream is ready for playback
  const isStreamReady = useCallback(async (streamId) => {
    if (!isTauriAvailable) return false

    try {
      return await videoControlAPI.isTranscodeReady(streamId)
    } catch (e) {
      console.error('[TauriTranscode] isStreamReady error:', e)
      return false
    }
  }, [isTauriAvailable])

  // Wait for stream to be ready (with timeout)
  const waitForReady = useCallback(async (streamId, timeoutMs = 30000) => {
    const startTime = Date.now()

    while (Date.now() - startTime < timeoutMs) {
      const ready = await isStreamReady(streamId)
      if (ready) return true

      // Wait 200ms before checking again
      await new Promise(resolve => setTimeout(resolve, 200))
    }

    return false
  }, [isStreamReady])

  // Get playlist path for a stream
  const getPlaylistPath = useCallback(async (streamId) => {
    if (!isTauriAvailable) return null

    try {
      return await videoControlAPI.getTranscodePlaylist(streamId)
    } catch (e) {
      console.error('[TauriTranscode] getPlaylistPath error:', e)
      return null
    }
  }, [isTauriAvailable])

  // Stop a transcoding session
  const stopTranscode = useCallback(async (streamId) => {
    if (!isTauriAvailable) return

    // Stop polling
    if (progressPollRef.current[streamId]) {
      clearInterval(progressPollRef.current[streamId])
      delete progressPollRef.current[streamId]
    }

    try {
      await videoControlAPI.stopTranscode(streamId)

      setActiveStreams(prev => {
        const next = { ...prev }
        delete next[streamId]
        return next
      })
    } catch (e) {
      console.error('[TauriTranscode] stopTranscode error:', e)
    }
  }, [isTauriAvailable])

  // Stop all transcoding sessions
  const stopAllTranscode = useCallback(async () => {
    if (!isTauriAvailable) return

    // Stop all polling
    for (const streamId of Object.keys(progressPollRef.current)) {
      clearInterval(progressPollRef.current[streamId])
    }
    progressPollRef.current = {}

    try {
      await videoControlAPI.stopAllTranscode()
      setActiveStreams({})
    } catch (e) {
      console.error('[TauriTranscode] stopAllTranscode error:', e)
    }
  }, [isTauriAvailable])

  // Get stream info by ID
  const getStreamInfo = useCallback((streamId) => {
    return activeStreams[streamId] || null
  }, [activeStreams])

  // Load capabilities on mount
  useEffect(() => {
    if (isTauriAvailable) {
      loadCapabilities()
    }
  }, [isTauriAvailable, loadCapabilities])

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true

    return () => {
      isMountedRef.current = false

      // Stop all polling
      for (const streamId of Object.keys(progressPollRef.current)) {
        clearInterval(progressPollRef.current[streamId])
      }

      // Stop all active transcodes
      videoControlAPI.stopAllTranscode().catch(() => {})
    }
  }, [])

  return {
    // State
    isAvailable: isTauriAvailable,
    capabilities,
    isLoading,
    error,
    activeStreams,

    // Functions
    loadCapabilities,
    checkNeedsTranscode,
    startTranscode,
    isStreamReady,
    waitForReady,
    getPlaylistPath,
    stopTranscode,
    stopAllTranscode,
    getStreamInfo
  }
}

export default useTauriTranscode
