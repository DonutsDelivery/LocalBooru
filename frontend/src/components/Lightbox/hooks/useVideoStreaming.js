import { useCallback, useState, useRef, useEffect } from 'react'
import Hls from 'hls.js'
import {
  getMediaUrl,
  getOpticalFlowConfig,
  playVideoInterpolated,
  stopInterpolatedStream,
  getSVPConfig,
  playVideoSVP,
  stopSVPStream,
  playVideoTranscode,
  stopTranscodeStream
} from '../../../api'
import { isVideo, needsTranscode } from '../utils/helpers'

/**
 * Hook for managing HLS/SVP/OpticalFlow video streaming
 */
export function useVideoStreaming(mediaRef, image, currentQuality) {
  // HLS streaming ref
  const hlsRef = useRef(null)

  // Optical flow interpolation state
  const [opticalFlowConfig, setOpticalFlowConfig] = useState(null)
  const [opticalFlowLoading, setOpticalFlowLoading] = useState(false)
  const [opticalFlowError, setOpticalFlowError] = useState(null)
  const [opticalFlowStreamUrl, setOpticalFlowStreamUrl] = useState(null)

  // SVP interpolation state
  const [svpConfig, setSvpConfig] = useState(null)
  const [svpConfigLoaded, setSvpConfigLoaded] = useState(false)
  const [svpLoading, setSvpLoading] = useState(false)
  const [svpError, setSvpError] = useState(null)
  const [svpStreamUrl, setSvpStreamUrl] = useState(null)
  const [svpTotalDuration, setSvpTotalDuration] = useState(null)  // Known total duration from API
  const [svpBufferedDuration, setSvpBufferedDuration] = useState(0)  // Duration available in HLS manifest
  const [svpPendingSeek, setSvpPendingSeek] = useState(null)  // Target time waiting for buffer
  const [svpStartOffset, setSvpStartOffset] = useState(0)  // Offset when stream started from seek position
  const svpHlsRef = useRef(null)
  const svpStartingRef = useRef(false)  // Synchronous lock to prevent double-starts

  // Transcode stream state (fallback when SVP/OpticalFlow not available)
  const [transcodeStreamUrl, setTranscodeStreamUrl] = useState(null)
  const [transcodeTotalDuration, setTranscodeTotalDuration] = useState(null)
  const [transcodeStartOffset, setTranscodeStartOffset] = useState(0)
  const transcodeHlsRef = useRef(null)
  const streamTransitioningRef = useRef(false)  // Block time updates during stream transitions

  // Source resolution (from original video)
  const [sourceResolution, setSourceResolution] = useState(null)

  // Track which streams were ever started (for cleanup - avoid unnecessary stop calls)
  const hadSvpStreamRef = useRef(false)
  const hadOpticalFlowStreamRef = useRef(false)
  const hadTranscodeStreamRef = useRef(false)
  const transcodeStartingRef = useRef(false)  // Synchronous lock to prevent double-starts in auto-start effect

  // Refs for callbacks (used by auto-start effect)
  const startSVPStreamRef = useRef(null)
  const startInterpolatedStreamRef = useRef(null)

  // Load optical flow config on mount
  useEffect(() => {
    async function loadOpticalFlowConfig() {
      try {
        const config = await getOpticalFlowConfig()
        setOpticalFlowConfig(config)
      } catch (err) {
        console.error('Failed to load optical flow config:', err)
      }
    }
    loadOpticalFlowConfig()
  }, [])

  // Load SVP config on mount
  useEffect(() => {
    async function loadSVPConfig() {
      try {
        const config = await getSVPConfig()
        setSvpConfig(config)
      } catch (err) {
        console.error('Failed to load SVP config:', err)
      } finally {
        setSvpConfigLoaded(true)
      }
    }
    loadSVPConfig()
  }, [])

  // Auto-dismiss SVP error toast after 5 seconds
  useEffect(() => {
    if (svpError) {
      const timer = setTimeout(() => {
        setSvpError(null)
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [svpError])

  // Auto-dismiss optical flow error toast after 5 seconds
  useEffect(() => {
    if (opticalFlowError) {
      const timer = setTimeout(() => {
        setOpticalFlowError(null)
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [opticalFlowError])

  // Helper to get the current absolute playback time (accounting for stream offsets)
  const getCurrentAbsoluteTime = useCallback(() => {
    if (!mediaRef.current) return 0
    const hlsTime = mediaRef.current.currentTime
    if (svpStreamUrl) {
      return hlsTime + svpStartOffset
    } else if (transcodeStreamUrl) {
      return hlsTime + transcodeStartOffset
    }
    return hlsTime
  }, [mediaRef, svpStreamUrl, svpStartOffset, transcodeStreamUrl, transcodeStartOffset])

  // Restart SVP stream from a specific position (for seeking beyond buffered content)
  const restartSVPFromPosition = useCallback(async (targetTime) => {
    if (!image || !svpConfig?.enabled) return

    console.log(`[SVP] Restarting stream from ${targetTime.toFixed(1)}s`)

    // Block time updates during stream transition to prevent 0:00 flash
    streamTransitioningRef.current = true
    // Show loading indicator
    setSvpLoading(true)
    setSvpPendingSeek(null)

    // Destroy current HLS instance
    if (svpHlsRef.current) {
      svpHlsRef.current.destroy()
      svpHlsRef.current = null
    }

    // Clear current stream state
    setSvpStreamUrl(null)
    setSvpBufferedDuration(0)

    try {
      // Start new stream from target position
      const result = await playVideoSVP(image.file_path, targetTime)

      if (result.success && result.stream_url) {
        // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset immediately
        setSvpStartOffset(targetTime)
        if (result.duration) {
          setSvpTotalDuration(result.duration)
        }
        if (result.source_resolution) {
          setSourceResolution(result.source_resolution)
        }
        // Set stream URL last - this triggers HLS setup
        setSvpStreamUrl(result.stream_url)
      } else {
        streamTransitioningRef.current = false
        setSvpError(result.error || 'Failed to restart SVP stream')
        setSvpLoading(false)
      }
    } catch (err) {
      console.error('SVP restart error:', err)
      streamTransitioningRef.current = false
      setSvpError(err.message || 'Failed to restart SVP stream')
      setSvpLoading(false)
    }
  }, [image, svpConfig])

  // Restart transcode stream from a specific position (for seeking beyond buffered content)
  const restartTranscodeFromPosition = useCallback(async (targetTime) => {
    if (!image) return

    console.log(`[Transcode] Restarting stream from ${targetTime.toFixed(1)}s`)

    // Block time updates during stream transition to prevent 0:00 flash
    streamTransitioningRef.current = true

    // Destroy current HLS instance
    if (transcodeHlsRef.current) {
      transcodeHlsRef.current.destroy()
      transcodeHlsRef.current = null
    }

    // Clear current stream state
    setTranscodeStreamUrl(null)

    try {
      // Start new stream from target position
      const result = await playVideoTranscode(image.file_path, targetTime, currentQuality)

      if (result.success && result.stream_url) {
        // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset immediately
        setTranscodeStartOffset(targetTime)
        if (result.duration) {
          setTranscodeTotalDuration(result.duration)
        }
        if (result.source_resolution) {
          setSourceResolution(result.source_resolution)
        }
        // Set stream URL last - this triggers HLS setup
        setTranscodeStreamUrl(result.stream_url)
      } else {
        console.error('Failed to restart transcode stream:', result.error)
        streamTransitioningRef.current = false
      }
    } catch (err) {
      console.error('Transcode restart error:', err)
      streamTransitioningRef.current = false
    }
  }, [image, currentQuality])

  // Start optical flow interpolation for video (called automatically when enabled)
  const startInterpolatedStream = useCallback(async (startPosition = null) => {
    if (!image || !opticalFlowConfig?.enabled || !isVideo(image.filename)) return
    if (opticalFlowStreamUrl || opticalFlowLoading) return // Already active or starting

    // Get current playback position before stopping other streams
    const playbackPosition = startPosition ?? getCurrentAbsoluteTime()

    // Stop any existing SVP stream
    if (svpStreamUrl) {
      setSvpStreamUrl(null)
      await stopSVPStream()
    }

    // Stop any existing transcode stream
    if (transcodeStreamUrl) {
      setTranscodeStreamUrl(null)
      await stopTranscodeStream()
    }

    setOpticalFlowLoading(true)
    setOpticalFlowError(null)

    try {
      const result = await playVideoInterpolated(image.file_path, playbackPosition, currentQuality)

      if (result.success && result.stream_url) {
        hadOpticalFlowStreamRef.current = true
        setOpticalFlowStreamUrl(result.stream_url)
        if (result.source_resolution) setSourceResolution(result.source_resolution)
      } else {
        setOpticalFlowError(result.error || 'Failed to start interpolated playback')
      }
    } catch (err) {
      console.error('Optical flow error:', err)
      setOpticalFlowError(err.message || 'Failed to start interpolated playback')
    }

    setOpticalFlowLoading(false)
  }, [image, opticalFlowConfig, opticalFlowStreamUrl, opticalFlowLoading, svpStreamUrl, transcodeStreamUrl, currentQuality, getCurrentAbsoluteTime])

  // Start SVP interpolation for video (called manually via button or auto-start)
  // Optional startPosition parameter - if not provided, defaults to 0 for new videos or current position for mode switches
  const startSVPStream = useCallback(async (startPosition = null) => {
    console.log('[startSVPStream] Called', { image: image?.id, enabled: svpConfig?.enabled, isVideo: isVideo(image?.filename), startPosition })
    if (!image || !svpConfig?.enabled || !isVideo(image.filename)) {
      console.log('[startSVPStream] Early return: missing image/config/not video')
      return
    }
    if (svpStreamUrl || svpLoading) {
      console.log('[startSVPStream] Early return: already active or loading', { svpStreamUrl, svpLoading })
      return
    }

    // Use ref for synchronous lock (state updates are batched/async)
    if (svpStartingRef.current) {
      console.log('[startSVPStream] Early return: ref lock active')
      return
    }
    svpStartingRef.current = true

    // Get current playback position before stopping other streams
    // Use provided position or fall back to current absolute time
    const playbackPosition = startPosition ?? getCurrentAbsoluteTime()

    // Stop any existing optical flow stream
    if (opticalFlowStreamUrl) {
      setOpticalFlowStreamUrl(null)
      await stopInterpolatedStream()
    }

    // Stop any existing transcode stream
    if (transcodeStreamUrl) {
      setTranscodeStreamUrl(null)
      await stopTranscodeStream()
    }

    setSvpLoading(true)
    setSvpError(null)

    console.log('[startSVPStream] Calling API with path:', image.file_path, 'position:', playbackPosition, 'quality:', currentQuality)
    try {
      const result = await playVideoSVP(image.file_path, playbackPosition, currentQuality)
      console.log('[startSVPStream] API result:', result)

      if (result.success && result.stream_url) {
        hadSvpStreamRef.current = true
        // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset
        setSvpStartOffset(playbackPosition)
        // Store the known total duration from API for proper timeline display
        if (result.duration) {
          setSvpTotalDuration(result.duration)
        }
        // Set source resolution from API (original video dimensions, not stream dimensions)
        if (result.source_resolution) {
          setSourceResolution(result.source_resolution)
        }
        // Set stream URL last - this triggers HLS setup
        setSvpStreamUrl(result.stream_url)
      } else {
        setSvpError(result.error || 'Failed to start SVP playback')
        setSvpLoading(false)  // Clear loading state on API failure
      }
    } catch (err) {
      console.error('SVP error:', err)
      setSvpError(err.message || 'Failed to start SVP playback')
      setSvpLoading(false)  // Only set loading false on error here
    } finally {
      svpStartingRef.current = false
    }
    // Note: svpLoading stays true until MANIFEST_PARSED fires in the useEffect
  }, [image, svpConfig, svpStreamUrl, svpLoading, opticalFlowStreamUrl, transcodeStreamUrl, currentQuality, getCurrentAbsoluteTime])

  // Update refs after callbacks are defined (used by auto-start effect)
  startSVPStreamRef.current = startSVPStream
  startInterpolatedStreamRef.current = startInterpolatedStream

  // Stop SVP stream
  const stopSVP = useCallback(async () => {
    if (svpHlsRef.current) {
      svpHlsRef.current.destroy()
      svpHlsRef.current = null
    }
    setSvpStreamUrl(null)
    setSvpTotalDuration(null)
    setSvpBufferedDuration(0)
    setSvpPendingSeek(null)
    setSvpError(null)
    await stopSVPStream()
  }, [])

  // Use a ref for currentQuality so the auto-start effect doesn't re-fire on quality changes.
  // Quality changes are handled by handleQualityChange — auto-start only handles new image opens.
  const currentQualityRef = useRef(currentQuality)
  currentQualityRef.current = currentQuality

  // Auto-start interpolated stream when video opens
  // Priority: SVP (if enabled) > Optical Flow (if enabled) > Transcode (if quality != original)
  // NOTE: currentQuality is intentionally NOT a dependency — quality changes are handled by handleQualityChange
  useEffect(() => {
    // Wait until both configs have loaded from the API before auto-starting.
    // Without this gate, the effect fires 3 times during startup:
    //   1. initial mount (both configs null)
    //   2. svpConfig loads (undefined→false)
    //   3. opticalFlowConfig loads (undefined→false)
    // Each fire starts a transcode that kills the previous one via stop_all_transcode_streams().
    if (svpConfig === null || opticalFlowConfig === null) return

    if (image && isVideo(image.filename)) {
      const quality = currentQualityRef.current
      console.log('[Auto-start] Checking...', {
        svpEnabled: svpConfig?.enabled,
        opticalFlowEnabled: opticalFlowConfig?.enabled,
        currentQuality: quality
      })
      // Prefer SVP if enabled
      if (svpConfig?.enabled) {
        console.log('[Auto-start] Starting SVP stream...')
        startSVPStreamRef.current()
      }
      // Fall back to optical flow if enabled
      else if (opticalFlowConfig?.enabled) {
        console.log('[Auto-start] Starting OpticalFlow stream...')
        startInterpolatedStreamRef.current()
      }
      // If quality is not original, use transcode
      else if ((!transcodeStreamUrl) && quality !== 'original') {
        // Use ref lock to prevent concurrent transcode starts (effect re-fires as configs load)
        if (transcodeStartingRef.current) return
        transcodeStartingRef.current = true
        console.log('[Auto-start] Starting transcode stream for quality:', quality)
        // Start transcode stream (position 0 for new video - reset effect already set currentTime to 0)
        playVideoTranscode(image.file_path, 0, quality).then(result => {
          if (result.success) {
            hadTranscodeStreamRef.current = true
            // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset
            setTranscodeStartOffset(0)
            if (result.duration) setTranscodeTotalDuration(result.duration)
            if (result.source_resolution) setSourceResolution(result.source_resolution)
            setTranscodeStreamUrl(result.stream_url)
          } else {
            console.error('[Auto-start] Transcode failed:', result.error)
          }
        }).catch(err => {
          console.error('[Auto-start] Transcode error:', err)
        }).finally(() => {
          transcodeStartingRef.current = false
        })
      }
      // Otherwise play direct (original quality, no interpolation)
    }
  }, [image?.id, svpConfig?.enabled, opticalFlowConfig?.enabled])

  // Setup HLS player when optical flow stream is active
  useEffect(() => {
    if (!opticalFlowStreamUrl || !mediaRef.current) return

    const video = mediaRef.current

    if (Hls.isSupported()) {
      // Pause video during transition to prevent playing old buffered content
      video.pause()

      // Cleanup previous instance - destroy() is synchronous and handles cleanup
      if (hlsRef.current) {
        hlsRef.current.destroy()
        hlsRef.current = null
      }

      // Remove direct video src (if any) - HLS will use MediaSource instead
      video.removeAttribute('src')

      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
        backBufferLength: 30
      })

      // Use getMediaUrl to handle dev mode (different ports for frontend/backend)
      hls.loadSource(getMediaUrl(opticalFlowStreamUrl))
      hls.attachMedia(video)

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(() => {})
      })

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          console.error('HLS fatal error:', data)
          setOpticalFlowError('Stream playback error.')
          setOpticalFlowStreamUrl(null)
        }
      })

      hlsRef.current = hls
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari/iOS native HLS support
      video.src = getMediaUrl(opticalFlowStreamUrl)
      video.addEventListener('loadedmetadata', () => {
        video.play().catch(() => {})
      })
    } else {
      setOpticalFlowError('HLS playback is not supported in this browser')
      setOpticalFlowStreamUrl(null)
    }

    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy()
        hlsRef.current = null
      }
    }
  }, [opticalFlowStreamUrl, mediaRef])

  // Setup HLS player when transcode stream is active (fallback when no interpolation)
  useEffect(() => {
    if (!transcodeStreamUrl || !mediaRef.current) return

    const video = mediaRef.current

    if (Hls.isSupported()) {
      // Pause video during transition to prevent playing old buffered content
      video.pause()

      // Cleanup previous instance - destroy() is synchronous and handles cleanup
      if (transcodeHlsRef.current) {
        transcodeHlsRef.current.destroy()
        transcodeHlsRef.current = null
      }

      // Remove direct video src (if any) - HLS will use MediaSource instead
      video.removeAttribute('src')

      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
        backBufferLength: 30
      })

      // Use getMediaUrl to handle dev mode (different ports for frontend/backend)
      hls.loadSource(getMediaUrl(transcodeStreamUrl))
      hls.attachMedia(video)

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        // Stream is ready, allow time updates again
        streamTransitioningRef.current = false
        video.play().catch(() => {})
      })

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          console.error('Transcode HLS fatal error:', data)
          streamTransitioningRef.current = false
          setTranscodeStreamUrl(null)
        }
      })

      transcodeHlsRef.current = hls
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari/iOS native HLS support
      video.src = getMediaUrl(transcodeStreamUrl)
      video.addEventListener('loadedmetadata', () => {
        streamTransitioningRef.current = false
        video.play().catch(() => {})
      })
    } else {
      console.error('HLS playback is not supported')
      setTranscodeStreamUrl(null)
    }

    return () => {
      if (transcodeHlsRef.current) {
        transcodeHlsRef.current.destroy()
        transcodeHlsRef.current = null
      }
    }
  }, [transcodeStreamUrl, mediaRef])

  // Setup HLS player when SVP stream URL is available
  // Keep normal video playing until HLS is ready, then switch
  useEffect(() => {
    if (!svpStreamUrl || !mediaRef.current) return

    const video = mediaRef.current
    let cancelled = false

    if (Hls.isSupported()) {
      // Pause video during transition to prevent playing old buffered content
      video.pause()

      // Cleanup previous instance - destroy() is synchronous and handles cleanup
      if (svpHlsRef.current) {
        svpHlsRef.current.destroy()
        svpHlsRef.current = null
      }

      // Remove direct video src (if any) - HLS will use MediaSource instead
      video.removeAttribute('src')

      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,  // Enable low latency for live-style HLS playlist
        backBufferLength: 30,
        maxBufferLength: 30,           // Buffer up to 30 seconds ahead
        maxMaxBufferLength: 60,        // Allow up to 60 seconds in buffer
        // Retry manifest loading while SVP produces initial segments
        manifestLoadingMaxRetry: 30,
        manifestLoadingRetryDelay: 500,   // 500ms between retries (faster feedback)
        manifestLoadingMaxRetryTimeout: 60000,
        // Also retry level/fragment loading
        levelLoadingMaxRetry: 10,
        levelLoadingRetryDelay: 500,
        fragLoadingMaxRetry: 10,
        fragLoadingRetryDelay: 500,
        // Start playback even with small buffer
        startPosition: 0,
      })

      // Attach media and load source
      // Use getMediaUrl to handle dev mode (different ports for frontend/backend)
      const fullStreamUrl = getMediaUrl(svpStreamUrl)
      console.log('[SVP HLS] Stream URL:', fullStreamUrl)
      hls.loadSource(fullStreamUrl)
      hls.attachMedia(video)  // Attach immediately (like optical flow code)

      // Debug logging for HLS events
      hls.on(Hls.Events.MANIFEST_LOADING, () => {
        console.log('[SVP HLS] Loading manifest...')
      })

      hls.on(Hls.Events.MANIFEST_LOADED, (event, data) => {
        console.log('[SVP HLS] Manifest loaded:', data.levels?.length, 'levels')
      })

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (cancelled) return
        console.log('[SVP HLS] Manifest parsed, starting playback')
        // Stream is ready, allow time updates again
        streamTransitioningRef.current = false
        setSvpLoading(false)
        video.play().catch(() => {})
      })

      hls.on(Hls.Events.FRAG_LOADED, (event, data) => {
        console.log('[SVP HLS] Fragment loaded:', data.frag.sn)
      })

      // Track available duration from HLS manifest for seek handling
      hls.on(Hls.Events.LEVEL_UPDATED, (event, data) => {
        if (cancelled) return
        const levelDetails = data.details
        if (levelDetails && levelDetails.totalduration) {
          const availableDuration = levelDetails.totalduration
          console.log('[SVP HLS] Level updated, available duration:', availableDuration.toFixed(1) + 's')
          setSvpBufferedDuration(availableDuration)
        }
      })

      let retryCount = 0
      const maxRetries = 60  // Allow up to 60 retries (1 min) for large files indexing

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (cancelled) return

        if (data.fatal) {
          // Check if it's a retryable network/manifest error during startup
          const isManifestError = data.details === 'manifestLoadError' ||
                                   data.details === 'manifestParsingError'
          const isNetworkError = data.type === Hls.ErrorTypes.NETWORK_ERROR

          if ((isManifestError || isNetworkError) && retryCount < maxRetries) {
            retryCount++
            console.log(`[SVP HLS] Fatal error, manual retry ${retryCount}/${maxRetries}:`, data.details)
            // HLS.js stops after fatal error - must manually restart loading
            setTimeout(() => {
              if (!cancelled) {
                hls.startLoad()
              }
            }, 1000)
          } else {
            // Give up - either not a retryable error, or retries exhausted
            console.error('[SVP HLS] Giving up after fatal error:', data)
            const errorMsg = retryCount >= maxRetries
              ? 'SVP stream failed to start (timeout)'
              : `SVP stream error: ${data.details || 'playback failed'}`
            streamTransitioningRef.current = false
            setSvpError(errorMsg)
            setSvpStreamUrl(null)
            setSvpTotalDuration(null)
            setSvpBufferedDuration(0)
            setSvpPendingSeek(null)
            setSvpLoading(false)
          }
        } else {
          // Non-fatal error - log but don't stop
          console.warn('[SVP HLS] Non-fatal error:', data.details)
        }
      })

      svpHlsRef.current = hls
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari/iOS native HLS support - just set src directly
      // It will handle buffering itself
      video.src = getMediaUrl(svpStreamUrl)
      video.addEventListener('loadedmetadata', () => {
        streamTransitioningRef.current = false
        video.play().catch(() => {})
        setSvpLoading(false)
      })
    } else {
      streamTransitioningRef.current = false
      setSvpError('HLS playback is not supported in this browser')
      setSvpStreamUrl(null)
      setSvpTotalDuration(null)
      setSvpBufferedDuration(0)
      setSvpPendingSeek(null)
      setSvpLoading(false)
    }

    return () => {
      cancelled = true
      if (svpHlsRef.current) {
        svpHlsRef.current.destroy()
        svpHlsRef.current = null
      }
    }
  }, [svpStreamUrl, mediaRef])

  // Cleanup HLS on unmount
  useEffect(() => {
    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy()
        hlsRef.current = null
      }
      if (svpHlsRef.current) {
        svpHlsRef.current.destroy()
        svpHlsRef.current = null
      }
      if (transcodeHlsRef.current) {
        transcodeHlsRef.current.destroy()
        transcodeHlsRef.current = null
      }
      // Only stop backend streams that were actually started
      if (hadSvpStreamRef.current) stopSVPStream().catch(() => {})
      if (hadOpticalFlowStreamRef.current) stopInterpolatedStream().catch(() => {})
      if (hadTranscodeStreamRef.current) stopTranscodeStream().catch(() => {})
    }
  }, [])

  // Reset streaming state (called when image changes)
  const resetStreamingState = useCallback(async (shouldStopStreams = false) => {
    // Reset optical flow state for new video
    setOpticalFlowError(null)
    setOpticalFlowStreamUrl(null)
    // Reset SVP state for new video
    setSvpError(null)
    setSvpStreamUrl(null)
    setSvpTotalDuration(null)
    setSvpBufferedDuration(0)
    setSvpPendingSeek(null)
    setSvpStartOffset(0)
    svpStartingRef.current = false  // Reset lock for new video
    transcodeStartingRef.current = false  // Reset lock for new video
    streamTransitioningRef.current = false  // Reset transition flag for new video
    setSourceResolution(null)  // Reset so it gets set from original video, not stream
    setTranscodeTotalDuration(null)
    setTranscodeStartOffset(0)
    setTranscodeStreamUrl(null)
    // Cleanup HLS instances
    if (hlsRef.current) {
      hlsRef.current.destroy()
      hlsRef.current = null
    }
    if (svpHlsRef.current) {
      svpHlsRef.current.destroy()
      svpHlsRef.current = null
    }
    if (transcodeHlsRef.current) {
      transcodeHlsRef.current.destroy()
      transcodeHlsRef.current = null
    }
    // Stop backend streams if requested
    if (shouldStopStreams) {
      stopSVPStream().catch(() => {})
      stopInterpolatedStream().catch(() => {})
      stopTranscodeStream().catch(() => {})
    }
  }, [])

  // Handle quality change
  const handleQualityChange = useCallback(async (qualityId, setCurrentTimeCallback) => {
    console.log('[Lightbox] Quality change requested:', qualityId)

    if (!mediaRef.current) {
      console.log('[Lightbox] No mediaRef available')
      return
    }

    // Get the absolute playback position before stopping/switching streams
    const absoluteTime = getCurrentAbsoluteTime()
    console.log('[Lightbox] Current absolute time:', absoluteTime)

    try {
      if (qualityId === 'original' && !needsTranscode(image?.filename)) {
        console.log('[Lightbox] Switching to original quality')
        // Stop streams and destroy HLS instances before setting direct src.
        // HLS.js attaches a MediaSource to the video element — we must destroy
        // it before setting a plain src, otherwise the video won't load.
        if (svpStreamUrl) {
          await stopSVPStream()
          if (svpHlsRef.current) {
            svpHlsRef.current.destroy()
            svpHlsRef.current = null
          }
          setSvpStreamUrl(null)
          setSvpStartOffset(0)
        }
        if (opticalFlowStreamUrl) {
          await stopInterpolatedStream()
          if (hlsRef.current) {
            hlsRef.current.destroy()
            hlsRef.current = null
          }
          setOpticalFlowStreamUrl(null)
        }
        if (transcodeStreamUrl) {
          await stopTranscodeStream()
          if (transcodeHlsRef.current) {
            transcodeHlsRef.current.destroy()
            transcodeHlsRef.current = null
          }
          setTranscodeStreamUrl(null)
          setTranscodeStartOffset(0)
        }

        if (mediaRef.current && image) {
          const video = mediaRef.current
          video.src = getMediaUrl(image.url)
          // Wait for metadata to load before seeking and playing — calling play()
          // immediately after setting src fails silently because the source hasn't loaded yet.
          video.addEventListener('loadedmetadata', () => {
            video.currentTime = absoluteTime
            video.play().catch(() => {})
          }, { once: true })
        }
      } else {
        // Restart stream with new quality at the current absolute position
        console.log('[Lightbox] Restarting stream with quality:', qualityId, 'at time:', absoluteTime)
        console.log('[Lightbox] SVP enabled/ready:', svpConfig?.enabled, svpConfig?.status?.ready)
        console.log('[Lightbox] OpticalFlow enabled:', opticalFlowConfig?.enabled)

        // Check if SVP is currently playing
        if (svpStreamUrl) {
          console.log('[Lightbox] Restarting SVP stream with quality')
          await stopSVPStream()
          const result = await playVideoSVP(image.file_path, absoluteTime, qualityId)
          console.log('[Lightbox] SVP play result:', result)
          if (result.success) {
            // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset
            setSvpStartOffset(absoluteTime)
            if (result.duration) setSvpTotalDuration(result.duration)
            if (result.source_resolution) setSourceResolution(result.source_resolution)
            setSvpStreamUrl(result.stream_url)
          } else {
            console.error('[Lightbox] SVP play failed:', result.error)
          }
        } else if (opticalFlowStreamUrl) {
          console.log('[Lightbox] Restarting OpticalFlow stream with quality')
          await stopInterpolatedStream()
          const result = await playVideoInterpolated(image.file_path, absoluteTime, qualityId)
          console.log('[Lightbox] OpticalFlow play result:', result)
          if (result.success) {
            setOpticalFlowStreamUrl(result.stream_url)
          } else {
            console.error('[Lightbox] OpticalFlow play failed:', result.error)
          }
        } else if (transcodeStreamUrl) {
          console.log('[Lightbox] Restarting transcode stream with quality')
          await stopTranscodeStream()
          const result = await playVideoTranscode(image.file_path, absoluteTime, qualityId)
          console.log('[Lightbox] Transcode play result:', result)
          if (result.success) {
            // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset
            setTranscodeStartOffset(absoluteTime)
            if (result.duration) setTranscodeTotalDuration(result.duration)
            if (result.source_resolution) setSourceResolution(result.source_resolution)
            setTranscodeStreamUrl(result.stream_url)
          } else {
            console.error('[Lightbox] Transcode play failed:', result.error)
          }
        } else {
          // No stream currently playing, starting fresh
          console.log('[Lightbox] No active stream, starting new one with quality')
          // Try SVP first (if enabled), fall back to OpticalFlow (if enabled), or use transcode
          if (svpConfig?.enabled) {
            // SVP is enabled, try to use it
            console.log('[Lightbox] Starting new SVP stream with quality')
            const result = await playVideoSVP(image.file_path, absoluteTime, qualityId)
            console.log('[Lightbox] SVP play result:', result)
            if (result.success) {
              hadSvpStreamRef.current = true
              // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset
              setSvpStartOffset(absoluteTime)
              if (result.duration) setSvpTotalDuration(result.duration)
              if (result.source_resolution) setSourceResolution(result.source_resolution)
              setSvpStreamUrl(result.stream_url)
            } else {
              console.error('[Lightbox] SVP play failed:', result.error)
              alert('Failed to start SVP stream: ' + result.error)
            }
          } else if (opticalFlowConfig?.enabled) {
            // OpticalFlow is enabled, try to use it
            console.log('[Lightbox] Starting new OpticalFlow stream with quality')
            const result = await playVideoInterpolated(image.file_path, absoluteTime, qualityId)
            console.log('[Lightbox] OpticalFlow play result:', result)
            if (result.success) {
              hadOpticalFlowStreamRef.current = true
              setOpticalFlowStreamUrl(result.stream_url)
            } else {
              console.error('[Lightbox] OpticalFlow play failed:', result.error)
              alert('Failed to start OpticalFlow stream: ' + result.error)
            }
          } else {
            // Neither SVP nor OpticalFlow enabled, use simple transcode
            console.log('[Lightbox] Using transcode (FFmpeg only) for quality change')
            const result = await playVideoTranscode(image.file_path, absoluteTime, qualityId)
            console.log('[Lightbox] Transcode play result:', result)
            if (result.success) {
              hadTranscodeStreamRef.current = true
              // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset
              setTranscodeStartOffset(absoluteTime)
              if (result.duration) setTranscodeTotalDuration(result.duration)
              if (result.source_resolution) setSourceResolution(result.source_resolution)
              setTranscodeStreamUrl(result.stream_url)
            } else {
              console.error('[Lightbox] Transcode play failed:', result.error)
              alert('Failed to transcode video: ' + result.error)
            }
          }
        }
      }
    } catch (err) {
      console.error('Failed to change quality:', err)
    }
  }, [image, mediaRef, svpStreamUrl, opticalFlowStreamUrl, transcodeStreamUrl, svpConfig, opticalFlowConfig, getCurrentAbsoluteTime])

  return {
    // Optical flow state
    opticalFlowConfig,
    opticalFlowLoading,
    opticalFlowError,
    opticalFlowStreamUrl,
    // SVP state
    svpConfig,
    setSvpConfig,
    svpConfigLoaded,
    svpLoading,
    setSvpLoading,
    svpError,
    setSvpError,
    svpStreamUrl,
    setSvpStreamUrl,
    svpTotalDuration,
    svpBufferedDuration,
    svpPendingSeek,
    setSvpPendingSeek,
    svpStartOffset,
    svpStartingRef,
    // Transcode state
    transcodeStreamUrl,
    transcodeTotalDuration,
    transcodeStartOffset,
    // Other
    sourceResolution,
    setSourceResolution,
    streamTransitioningRef,
    // Functions
    getCurrentAbsoluteTime,
    restartSVPFromPosition,
    restartTranscodeFromPosition,
    startInterpolatedStream,
    startSVPStream,
    startSVPStreamRef,
    stopSVP,
    resetStreamingState,
    handleQualityChange
  }
}
