import { useCallback, useState, useRef, useEffect } from 'react'
import { savePlaybackPosition } from '../../../api'
import { formatTime } from '../utils/helpers'

/**
 * Hook for managing video playback state and controls
 */
export function useVideoPlayback(mediaRef, streamState, imageId, directoryId, libraryId) {
  const {
    svpStreamUrl,
    svpStartOffset,
    svpBufferedDuration,
    svpPendingSeek,
    setSvpPendingSeek,
    transcodeStreamUrl,
    transcodeStartOffset,
    transcodeBufferedDuration,
    svpTotalDuration,
    transcodeTotalDuration,
    opticalFlowStreamUrl,
    streamTransitioningRef,
    getCurrentAbsoluteTime,
    restartSVPFromPosition,
    cancelPendingSVPRestart,
    restartTranscodeFromPosition,
    setAudioOutputVolume,
    setAudioOutputMuted
  } = streamState

  // Video player state
  const [isPlaying, setIsPlaying] = useState(true) // Start autoplaying
  const [_currentTime, _setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isSeeking, setIsSeeking] = useState(false)
  const [videoDisplayMode, setVideoDisplayMode] = useState('fit') // 'fit' | 'fill' | 'original'
  const [videoNaturalSize, setVideoNaturalSize] = useState({ width: 0, height: 0 })
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0)
  const timelineRef = useRef(null)
  const lastSavedPositionRef = useRef(0)
  const saveIntervalRef = useRef(null)

  // Refs for direct DOM updates during playback (bypasses React re-renders)
  const currentTimeRef = useRef(0)
  const durationRef = useRef(0)
  const progressBarRef = useRef(null)
  const playheadRef = useRef(null)
  const timeDisplayRef = useRef(null)

  // Wrapper for setCurrentTime that keeps ref in sync (used during seeking/interactions)
  const setCurrentTime = useCallback((timeOrFn) => {
    if (typeof timeOrFn === 'function') {
      _setCurrentTime(prev => {
        const result = timeOrFn(prev)
        currentTimeRef.current = result
        return result
      })
    } else {
      currentTimeRef.current = timeOrFn
      _setCurrentTime(timeOrFn)
    }
  }, [])

  const isStreamTimeBuffered = useCallback((absoluteTime, streamStartOffset, manifestDuration = 0) => {
    const video = mediaRef.current
    if (!video) return false

    const hlsTime = absoluteTime - streamStartOffset
    if (hlsTime < -0.25) return false

    const ranges = video.buffered
    for (let i = 0; i < ranges.length; i++) {
      if (hlsTime >= ranges.start(i) - 0.5 && hlsTime <= ranges.end(i) + 0.5) {
        return true
      }
    }

    // Before the browser exposes buffered ranges, keep the previous manifest
    // based behavior so startup seeks do not always restart the stream.
    return ranges.length === 0 && hlsTime >= -0.5 && hlsTime <= manifestDuration + 0.5
  }, [mediaRef])

  const seekWithinStream = useCallback((absoluteTime, streamStartOffset) => {
    const hlsTime = absoluteTime - streamStartOffset
    cancelPendingSVPRestart?.(absoluteTime)
    setSvpPendingSeek(null)
    mediaRef.current.currentTime = Math.max(0, hlsTime)
    setCurrentTime(absoluteTime)
  }, [mediaRef, setCurrentTime, setSvpPendingSeek, cancelPendingSVPRestart])

  // Direct DOM update for timeline elements (no React re-render)
  const updateTimeDisplay = useCallback((time) => {
    currentTimeRef.current = time
    const dur = durationRef.current
    if (timeDisplayRef.current) {
      timeDisplayRef.current.textContent = formatTime(time)
    }
    if (dur > 0) {
      const pct = `${(time / dur) * 100}%`
      if (progressBarRef.current) {
        progressBarRef.current.style.width = pct
      }
      if (playheadRef.current) {
        playheadRef.current.style.left = pct
      }
    }
  }, [])

  // Save playback position periodically (every 10s) and on pause/end/cleanup
  useEffect(() => {
    if (!imageId) return

    const savePosition = () => {
      if (!mediaRef.current) return
      const pos = getCurrentAbsoluteTime()
      const dur = duration
      if (dur <= 0 || pos <= 0) return
      // Avoid redundant saves within 5s of the same position
      if (Math.abs(pos - lastSavedPositionRef.current) < 5) return
      lastSavedPositionRef.current = pos
      savePlaybackPosition(imageId, pos, dur, directoryId, libraryId).catch(() => {})
    }

    saveIntervalRef.current = setInterval(savePosition, 10000)

    return () => {
      clearInterval(saveIntervalRef.current)
      // Save on cleanup (navigation away)
      savePosition()
    }
  }, [imageId, directoryId, libraryId, duration, mediaRef, getCurrentAbsoluteTime])

  // Seek forward/backward
  const seekVideo = useCallback((seconds) => {
    if (!mediaRef.current) return

    // For HLS streams, currentTime is in HLS time, need to convert to absolute video time
    const currentAbsoluteTime = getCurrentAbsoluteTime()
    const newTime = Math.max(0, Math.min(duration, currentAbsoluteTime + seconds))

    // For SVP streams, check if we need to restart from a new position
    if (svpStreamUrl) {
      if (!isStreamTimeBuffered(newTime, svpStartOffset, svpBufferedDuration)) {
        setCurrentTime(newTime)
        setSvpPendingSeek(newTime)
        restartSVPFromPosition(newTime)
        return
      }

      seekWithinStream(newTime, svpStartOffset)
      return
    }

    // For transcode streams, check if we can seek within buffered content
    if (transcodeStreamUrl) {
      const bufferedEnd = transcodeStartOffset + transcodeBufferedDuration
      const bufferedStart = transcodeStartOffset

      if (newTime >= bufferedStart - 1 && newTime <= bufferedEnd + 2) {
        // Seek within current stream
        const hlsTime = newTime - transcodeStartOffset
        mediaRef.current.currentTime = Math.max(0, hlsTime)
        setCurrentTime(newTime)
        return
      }

      restartTranscodeFromPosition(newTime)
      return
    }

    // Normal video seek (direct play) - use fastSeek for speed when available
    const wasPlaying = !mediaRef.current.paused
    if (mediaRef.current.fastSeek) {
      mediaRef.current.fastSeek(newTime)
    } else {
      mediaRef.current.currentTime = newTime
    }
    setCurrentTime(newTime)
    // Rapid seeks can cause the browser to stall in a paused state — nudge it back
    if (wasPlaying) {
      mediaRef.current.play().catch(() => {})
    }
  }, [mediaRef, duration, svpStreamUrl, svpBufferedDuration, svpStartOffset, transcodeStreamUrl, transcodeStartOffset, transcodeBufferedDuration, getCurrentAbsoluteTime, restartSVPFromPosition, restartTranscodeFromPosition, isStreamTimeBuffered, seekWithinStream, setSvpPendingSeek, setCurrentTime])

  // Toggle video play/pause
  const toggleVideoPlay = useCallback(() => {
    if (!mediaRef.current) return
    const video = mediaRef.current
    if (video.paused) {
      video.play().catch(() => {})
    } else {
      video.pause()
    }
  }, [mediaRef])

  // Sync play state with video element events
  const handleVideoPlay = useCallback(() => {
    setIsPlaying(true)
    // Note: Don't update sourceResolution here - it should only be set once from the original video
    // Updating it during streaming would give us the stream resolution (e.g., 480p) instead of source
  }, [])

  const handleVideoPause = useCallback(() => {
    setIsPlaying(false)
    // Save position on pause
    if (imageId && mediaRef.current && duration > 0) {
      const pos = getCurrentAbsoluteTime()
      if (pos > 0) {
        lastSavedPositionRef.current = pos
        savePlaybackPosition(imageId, pos, duration, directoryId, libraryId).catch(() => {})
      }
    }
  }, [imageId, directoryId, libraryId, mediaRef, duration, getCurrentAbsoluteTime])

  // Update current time as video plays — direct DOM write, no React re-render
  const handleTimeUpdate = useCallback(() => {
    if (!mediaRef.current || isSeeking) return
    // Don't update time display while waiting for pending seek or during stream transition
    if (svpPendingSeek || streamTransitioningRef.current) return
    // Add offset for streams that started from a seek position
    let actualTime = mediaRef.current.currentTime
    if (svpStreamUrl) {
      actualTime += svpStartOffset
    } else if (transcodeStreamUrl) {
      actualTime += transcodeStartOffset
    }
    updateTimeDisplay(actualTime)
  }, [mediaRef, isSeeking, svpPendingSeek, svpStreamUrl, svpStartOffset, transcodeStreamUrl, transcodeStartOffset, streamTransitioningRef, updateTimeDisplay])

  // Get duration and natural size when video metadata loads
  const handleLoadedMetadata = useCallback((setSourceResolution) => {
    if (!mediaRef.current) return
    // For HLS streams, use the known total duration from API if available
    // This allows the timeline to show the full video length even while segments are being generated
    // IMPORTANT: Never overwrite a known duration with a shorter HLS segment duration
    let newDuration = mediaRef.current.duration

    if (svpTotalDuration && svpStreamUrl) {
      newDuration = svpTotalDuration
    } else if (transcodeTotalDuration && transcodeStreamUrl) {
      newDuration = transcodeTotalDuration
    }

    // Only update duration if it's valid and either:
    // 1. We don't have a duration yet, OR
    // 2. The new duration is longer (we got the full video duration)
    // This prevents HLS segment duration from shrinking the timeline on seek
    setDuration(prev => {
      let result
      if (!prev || !isFinite(prev) || prev === 0) {
        result = newDuration
      } else {
        // Keep the longer duration (full video vs HLS segment)
        result = Math.max(prev, newDuration)
      }
      durationRef.current = result
      return result
    })

    const width = mediaRef.current.videoWidth
    const height = mediaRef.current.videoHeight

    // Only update natural size if it actually changed to prevent unnecessary re-renders
    setVideoNaturalSize(prev => {
      if (prev.width === width && prev.height === height) {
        return prev
      }
      return { width, height }
    })

    // Store source resolution for quality selector
    // ONLY set this when playing the ORIGINAL video (not a stream)
    // Stream dimensions would be the transcoded resolution (e.g., 480p), not the source
    const isStreaming = svpStreamUrl || opticalFlowStreamUrl || transcodeStreamUrl
    if (width > 0 && height > 0 && !isStreaming) {
      console.log('[Lightbox] Setting source resolution (original video):', width, 'x', height)
      setSourceResolution({
        width,
        height
      })
    } else {
      console.log('[Lightbox] Video dimensions not available yet, will retry on play')
    }
  }, [mediaRef, svpTotalDuration, svpStreamUrl, opticalFlowStreamUrl, transcodeStreamUrl, transcodeTotalDuration])

  // Handle seeking via timeline
  const handleSeek = useCallback((e) => {
    if (!mediaRef.current || !duration || !timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const percent = Math.max(0, Math.min(1, clickX / rect.width))
    const newTime = percent * duration  // Absolute video time

    // For SVP streams, check if we need to restart from a new position
    if (svpStreamUrl) {
      if (!isStreamTimeBuffered(newTime, svpStartOffset, svpBufferedDuration)) {
        console.log(`[SVP] Seeking to ${newTime.toFixed(1)}s outside browser buffer. Restarting stream...`)
        setCurrentTime(newTime)
        setSvpPendingSeek(newTime)
        restartSVPFromPosition(newTime)
        return
      }

      seekWithinStream(newTime, svpStartOffset)
      return
    }

    // For transcode streams, check if we can seek within buffered content
    if (transcodeStreamUrl) {
      const bufferedEnd = transcodeStartOffset + transcodeBufferedDuration
      const bufferedStart = transcodeStartOffset

      if (newTime >= bufferedStart - 1 && newTime <= bufferedEnd + 2) {
        // Seek within current stream
        const hlsTime = newTime - transcodeStartOffset
        mediaRef.current.currentTime = Math.max(0, hlsTime)
        setCurrentTime(newTime)
        return
      }

      console.log(`[Transcode] Seeking to ${newTime.toFixed(1)}s, buffered range: ${bufferedStart.toFixed(1)}-${bufferedEnd.toFixed(1)}s. Restarting stream...`)
      restartTranscodeFromPosition(newTime)
      return
    }

    // Normal video seek (direct playback) - use fastSeek for speed when available
    if (mediaRef.current.fastSeek) {
      mediaRef.current.fastSeek(newTime)
    } else {
      mediaRef.current.currentTime = newTime
    }
    setCurrentTime(newTime)
  }, [mediaRef, duration, svpStreamUrl, svpBufferedDuration, svpStartOffset, transcodeStreamUrl, transcodeStartOffset, transcodeBufferedDuration, restartSVPFromPosition, restartTranscodeFromPosition, isStreamTimeBuffered, seekWithinStream, setSvpPendingSeek, setCurrentTime])

  const handleSeekStart = useCallback((e) => {
    setIsSeeking(true)
    handleSeek(e)
  }, [handleSeek])

  const handleSeekMove = useCallback((e) => {
    if (!isSeeking) return
    // Only update display time during drag — actual seek happens on mouseup
    if (!mediaRef.current || !duration || !timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const percent = Math.max(0, Math.min(1, clickX / rect.width))
    const newTime = percent * duration
    setCurrentTime(newTime)
  }, [isSeeking, mediaRef, duration])

  const handleSeekEnd = useCallback(() => {
    if (!isSeeking) return
    setIsSeeking(false)

    // Seek to final position (read from ref — always in sync via setCurrentTime wrapper)
    if (!mediaRef.current || !duration) return
    const seekTime = currentTimeRef.current

    if (svpStreamUrl) {
      if (!isStreamTimeBuffered(seekTime, svpStartOffset, svpBufferedDuration)) {
        setCurrentTime(seekTime)
        setSvpPendingSeek(seekTime)
        restartSVPFromPosition(seekTime)
        return
      }
      seekWithinStream(seekTime, svpStartOffset)
      return
    }

    if (transcodeStreamUrl) {
      const bufferedEnd = transcodeStartOffset + transcodeBufferedDuration
      const bufferedStart = transcodeStartOffset
      if (seekTime >= bufferedStart - 1 && seekTime <= bufferedEnd + 2) {
        const hlsTime = seekTime - transcodeStartOffset
        mediaRef.current.currentTime = Math.max(0, hlsTime)
        return
      }
      restartTranscodeFromPosition(seekTime)
      return
    }

    // Direct play - precise seek to final position
    mediaRef.current.currentTime = seekTime
  }, [isSeeking, mediaRef, duration, svpStreamUrl, svpBufferedDuration, svpStartOffset, transcodeStreamUrl, transcodeStartOffset, transcodeBufferedDuration, restartSVPFromPosition, restartTranscodeFromPosition, isStreamTimeBuffered, seekWithinStream, setSvpPendingSeek, setCurrentTime])

  // Touch handlers for video timeline (mobile)
  const handleSeekTouchStart = useCallback((e) => {
    e.preventDefault()
    setIsSeeking(true)
    if (!mediaRef.current || !duration || !timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const touch = e.touches[0]
    const percent = Math.max(0, Math.min(1, (touch.clientX - rect.left) / rect.width))
    const newTime = percent * duration

    if (svpStreamUrl) {
      if (!isStreamTimeBuffered(newTime, svpStartOffset, svpBufferedDuration)) {
        setCurrentTime(newTime)
        setSvpPendingSeek(newTime)
        restartSVPFromPosition(newTime)
        return
      }
      seekWithinStream(newTime, svpStartOffset)
      return
    }
    if (transcodeStreamUrl) {
      const bufferedEnd = transcodeStartOffset + transcodeBufferedDuration
      const bufferedStart = transcodeStartOffset

      if (newTime >= bufferedStart - 1 && newTime <= bufferedEnd + 2) {
        const hlsTime = newTime - transcodeStartOffset
        mediaRef.current.currentTime = Math.max(0, hlsTime)
        setCurrentTime(newTime)
        return
      }

      restartTranscodeFromPosition(newTime)
      return
    }
    // Direct play - use fastSeek for speed when available
    if (mediaRef.current.fastSeek) {
      mediaRef.current.fastSeek(newTime)
    } else {
      mediaRef.current.currentTime = newTime
    }
    setCurrentTime(newTime)
  }, [mediaRef, duration, svpStreamUrl, svpBufferedDuration, svpStartOffset, transcodeStreamUrl, transcodeStartOffset, transcodeBufferedDuration, restartSVPFromPosition, restartTranscodeFromPosition, isStreamTimeBuffered, seekWithinStream, setSvpPendingSeek, setCurrentTime])

  const handleSeekTouchMove = useCallback((e) => {
    if (!isSeeking) return
    e.preventDefault()
    if (!mediaRef.current || !duration || !timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const touch = e.touches[0]
    const percent = Math.max(0, Math.min(1, (touch.clientX - rect.left) / rect.width))
    const newTime = percent * duration

    // For touch move, just update display time without seeking
    // Actual seek happens on touch end
    setCurrentTime(newTime)
  }, [mediaRef, isSeeking, duration])

  const handleSeekTouchEnd = useCallback((e) => {
    if (!isSeeking) return
    e.preventDefault()
    setIsSeeking(false)

    // Seek to final position (read from ref — always in sync via setCurrentTime wrapper)
    if (!mediaRef.current || !duration) return
    const seekTime = currentTimeRef.current

    if (svpStreamUrl) {
      if (!isStreamTimeBuffered(seekTime, svpStartOffset, svpBufferedDuration)) {
        setCurrentTime(seekTime)
        setSvpPendingSeek(seekTime)
        restartSVPFromPosition(seekTime)
        return
      }
      seekWithinStream(seekTime, svpStartOffset)
      return
    }
    if (transcodeStreamUrl) {
      const bufferedEnd = transcodeStartOffset + transcodeBufferedDuration
      const bufferedStart = transcodeStartOffset

      if (seekTime >= bufferedStart - 1 && seekTime <= bufferedEnd + 2) {
        const hlsTime = seekTime - transcodeStartOffset
        mediaRef.current.currentTime = Math.max(0, hlsTime)
        return
      }

      restartTranscodeFromPosition(seekTime)
      return
    }
    // Direct play - precise seek to final position
    mediaRef.current.currentTime = seekTime
  }, [mediaRef, isSeeking, duration, svpStreamUrl, svpBufferedDuration, svpStartOffset, transcodeStreamUrl, transcodeStartOffset, transcodeBufferedDuration, restartSVPFromPosition, restartTranscodeFromPosition, isStreamTimeBuffered, seekWithinStream, setSvpPendingSeek, setCurrentTime])

  // Handle volume change
  const handleVolumeChange = useCallback((e) => {
    const newVolume = parseFloat(e.target.value)
    setVolume(newVolume)
    if (mediaRef.current) {
      const handledByAudioGraph = setAudioOutputVolume?.(newVolume)
      if (!handledByAudioGraph) {
        mediaRef.current.volume = newVolume
        mediaRef.current.muted = newVolume === 0
      }
      setIsMuted(newVolume === 0)
    }
  }, [mediaRef, setAudioOutputVolume])

  // Toggle mute
  const toggleMute = useCallback(() => {
    if (!mediaRef.current) return
    if (isMuted) {
      const handledByAudioGraph = setAudioOutputMuted?.(false, volume)
      if (!handledByAudioGraph) {
        mediaRef.current.muted = false
      }
      setIsMuted(false)
    } else {
      const handledByAudioGraph = setAudioOutputMuted?.(true, volume)
      if (!handledByAudioGraph) {
        mediaRef.current.muted = true
      }
      setIsMuted(true)
    }
  }, [mediaRef, isMuted, volume, setAudioOutputMuted])

  // Increase playback speed
  const increaseSpeed = useCallback(() => {
    if (!mediaRef.current) return
    const newSpeed = Math.min(4.0, playbackSpeed + 0.25)
    setPlaybackSpeed(newSpeed)
    mediaRef.current.playbackRate = newSpeed
  }, [mediaRef, playbackSpeed])

  // Decrease playback speed
  const decreaseSpeed = useCallback(() => {
    if (!mediaRef.current) return
    const newSpeed = Math.max(0.25, playbackSpeed - 0.25)
    setPlaybackSpeed(newSpeed)
    mediaRef.current.playbackRate = newSpeed
  }, [mediaRef, playbackSpeed])

  // Reset playback speed to 1.0x
  const resetSpeed = useCallback(() => {
    if (!mediaRef.current) return
    setPlaybackSpeed(1.0)
    mediaRef.current.playbackRate = 1.0
  }, [mediaRef])

  // Frame advance (when paused)
  const frameAdvance = useCallback(() => {
    if (!mediaRef.current || isPlaying) return
    const newTime = Math.min(duration, mediaRef.current.currentTime + (1/30))
    mediaRef.current.currentTime = newTime
    setCurrentTime(newTime)
  }, [mediaRef, isPlaying, duration])

  // Adjust volume by delta
  const adjustVolume = useCallback((delta) => {
    if (!mediaRef.current) return
    const newVolume = Math.max(0, Math.min(1, volume + delta))
    setVolume(newVolume)
    const handledByAudioGraph = setAudioOutputVolume?.(newVolume)
    if (!handledByAudioGraph) {
      mediaRef.current.volume = newVolume
    }
    if (newVolume > 0 && isMuted) {
      setIsMuted(false)
      if (!handledByAudioGraph) {
        mediaRef.current.muted = false
      }
    }
  }, [mediaRef, volume, isMuted, setAudioOutputVolume])

  // Cycle video display mode
  const cycleDisplayMode = useCallback(() => {
    setVideoDisplayMode(prev => {
      switch (prev) {
        case 'fit': return 'original'
        case 'original': return 'fill'
        case 'fill': return 'fit'
        default: return 'fit'
      }
    })
  }, [])

  const restorePlaybackState = useCallback((state) => {
    if (!state) return []
    const position = Number.isFinite(state.position) ? Math.max(0, state.position) : 0
    const restoredDuration = Number.isFinite(state.duration) ? Math.max(0, state.duration) : 0
    const restoredVolume = Number.isFinite(state.volume) ? Math.max(0, Math.min(1, state.volume)) : 1
    const restoredSpeed = Number.isFinite(state.speed) ? Math.max(0.25, Math.min(4, state.speed)) : 1
    const displayMode = ['fit', 'fill', 'original'].includes(state.display_mode) ? state.display_mode : 'fit'
    setCurrentTime(position)
    setDuration(restoredDuration)
    durationRef.current = restoredDuration
    setIsPlaying(!state.paused)
    setVolume(restoredVolume)
    setIsMuted(Boolean(state.muted))
    setPlaybackSpeed(restoredSpeed)
    setVideoDisplayMode(displayMode)
    const media = mediaRef.current
    if (media) {
      const applyPosition = () => {
        if (Number.isFinite(media.duration)) media.currentTime = Math.min(position, media.duration)
      }
      if (media.readyState >= 1) applyPosition()
      else media.addEventListener('loadedmetadata', applyPosition, { once: true })
      media.volume = restoredVolume
      media.muted = Boolean(state.muted)
      media.playbackRate = restoredSpeed
      if (state.paused) media.pause()
      else media.play().catch(() => {})
    }
    const unsupported = []
    if (state.selected_audio_track != null) unsupported.push('selected_audio_track')
    if (state.selected_subtitle_track != null) unsupported.push('selected_subtitle_track')
    if (state.subtitle_delay) unsupported.push('subtitle_delay')
    if (state.svp_enabled) unsupported.push('native_svp_configuration')
    return unsupported
  }, [mediaRef, setCurrentTime])

  // Reset playback state
  const resetPlaybackState = useCallback(() => {
    setIsPlaying(true)
    setCurrentTime(0)
    setDuration(0)
    durationRef.current = 0
    setIsSeeking(false)
    setVideoDisplayMode('fit')
    setVideoNaturalSize({ width: 0, height: 0 })
    setPlaybackSpeed(1.0)
    if (mediaRef.current) {
      mediaRef.current.playbackRate = 1.0
    }
  }, [mediaRef, setCurrentTime])

  return {
    isPlaying,
    setIsPlaying,
    currentTime: _currentTime,
    setCurrentTime,
    duration,
    setDuration,
    isSeeking,
    videoDisplayMode,
    setVideoDisplayMode,
    videoNaturalSize,
    volume,
    isMuted,
    playbackSpeed,
    timelineRef,
    currentTimeRef,
    progressBarRef,
    playheadRef,
    timeDisplayRef,
    getCurrentAbsoluteTime,
    seekVideo,
    toggleVideoPlay,
    handleVideoPlay,
    handleVideoPause,
    handleTimeUpdate,
    handleLoadedMetadata,
    handleSeekStart,
    handleSeekMove,
    handleSeekEnd,
    handleSeekTouchStart,
    handleSeekTouchMove,
    handleSeekTouchEnd,
    handleVolumeChange,
    toggleMute,
    increaseSpeed,
    decreaseSpeed,
    resetSpeed,
    frameAdvance,
    adjustVolume,
    cycleDisplayMode,
    restorePlaybackState,
    resetPlaybackState
  }
}
