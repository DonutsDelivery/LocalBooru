import { useCallback, useState, useRef, useEffect } from 'react'
import Hls from 'hls.js'
import {
  getMediaUrl,
  getAssetUrl,
  isUsingLocalServer,
  playVideoInterpolated,
  stopInterpolatedStream,
  getSVPConfig,
  playVideoSVP,
  stopSVPStream,
  openSVPProcessingSession,
  getSVPProcessingEvents,
  fetchSVPProcessingSegment,
  acknowledgeSVPInitSegment,
  acknowledgeSVPMediaSegment,
  pauseSVPProcessingSession,
  resumeSVPProcessingSession,
  seekSVPProcessingSession,
  stopSVPProcessingSession,
  playVideoTranscode,
  stopTranscodeStream
} from '../../../api'
import { isLinuxDesktopApp, isMobileApp, isWindowsOrMacDesktopApp } from '../../../serverManager'
import { isVideo } from '../utils/helpers'
import { MSESessionController } from '../utils/MSESessionController'
import { useAudioNormalization } from './useAudioNormalization'
import { shouldRestartStalledSVP } from './svpStallGuard'
import { getCodecFallbackStartPosition } from './codecFallback'
import { shouldStartSVPPlayback } from './svpBuffering'
import {
  capturePlaybackIntent as captureTransitionIntent,
  createPlaybackTransitionOwner,
  nextPlaybackSourceRevision,
} from './playbackTransition'

const svpMSEClient = {
  open: openSVPProcessingSession,
  events: getSVPProcessingEvents,
  segment: fetchSVPProcessingSegment,
  ackInit: acknowledgeSVPInitSegment,
  ackMedia: acknowledgeSVPMediaSegment,
  pause: pauseSVPProcessingSession,
  resume: resumeSVPProcessingSession,
  seek: seekSVPProcessingSession,
  stop: stopSVPProcessingSession
}

/**
 * Hook for managing HLS/SVP/OpticalFlow video streaming
 * @param {object} addonStatus - { svpInstalled } from useAddonStatus
 */
export function useVideoStreaming(mediaRef, image, currentQuality, addonStatus = {}) {
  const { svpInstalled = false, enabled = true } = addonStatus
  const nativeSvpPlayback = isLinuxDesktopApp()
  const mseSvpPlayback = isWindowsOrMacDesktopApp() && typeof MediaSource !== 'undefined'
  const svpMseControllerRef = useRef(null)
  const imageRef = useRef(image)
  imageRef.current = image
  const currentImageKey = image?.url ?? image?.file_path ?? image?.id
  const { applyNormalization, resetGain, setOutputVolume, setOutputMuted } = useAudioNormalization(mediaRef)
  // HLS streaming ref
  const hlsRef = useRef(null)

  // Retained stream state for compatibility with existing lightbox consumers.
  // FFmpeg interpolation is retired; SVP is the sole interpolation engine.
  const [opticalFlowConfig] = useState({ enabled: false })
  const [opticalFlowLoading, setOpticalFlowLoading] = useState(false)
  const [opticalFlowError, setOpticalFlowError] = useState(null)
  const [opticalFlowStreamUrl, setOpticalFlowStreamUrl] = useState(null)

  // SVP interpolation state
  const [svpConfig, setSvpConfig] = useState(null)
  const [svpConfigLoaded, setSvpConfigLoaded] = useState(false)
  const [svpLoading, setSvpLoading] = useState(false)
  const [svpError, setSvpError] = useState(null)
  const [svpStreamUrl, setSvpStreamUrl] = useState(null)
  const [svpSourceRevision, setSvpSourceRevision] = useState(0)
  const [svpTotalDuration, setSvpTotalDuration] = useState(null)  // Known total duration from API
  const [svpBufferedDuration, setSvpBufferedDuration] = useState(0)  // Duration available in HLS manifest
  const [svpPendingSeek, setSvpPendingSeek] = useState(null)  // Target time waiting for buffer
  const [svpStartOffset, setSvpStartOffset] = useState(0)  // Offset when stream started from seek position
  const svpHlsRef = useRef(null)
  const svpStartingRef = useRef(null)  // Owns the currently starting request
  const svpRestartingRef = useRef(false)
  const svpRestartOwnerRef = useRef(null)
  const svpQueuedRestartRef = useRef(null)
  const svpRestartTokenRef = useRef(0)
  const mseSeekQueueRef = useRef(Promise.resolve())
  const mseSeekTokenRef = useRef(0)
  const mseSeekOwnerRef = useRef(null)

  // Transcode stream state (fallback when SVP/OpticalFlow not available)
  const [transcodeStreamUrl, setTranscodeStreamUrl] = useState(null)
  const [transcodeSourceRevision, setTranscodeSourceRevision] = useState(0)
  const [transcodeTotalDuration, setTranscodeTotalDuration] = useState(null)
  const [transcodeStartOffset, setTranscodeStartOffset] = useState(0)
  const [transcodeBufferedDuration, setTranscodeBufferedDuration] = useState(0)
  const transcodeHlsRef = useRef(null)
  const streamTransitioningRef = useRef(false)  // Block time updates during stream transitions
  const playbackTransitionOwnerRef = useRef(null)
  if (!playbackTransitionOwnerRef.current) {
    playbackTransitionOwnerRef.current = createPlaybackTransitionOwner()
  }
  const activeStreamOwnerRef = useRef(null)

  // Source resolution (from original video)
  const [sourceResolution, setSourceResolution] = useState(null)

  // Track which streams were ever started (for cleanup - avoid unnecessary stop calls)
  const hadSvpStreamRef = useRef(false)
  const hadOpticalFlowStreamRef = useRef(false)
  const hadTranscodeStreamRef = useRef(false)
  const [codecFallbackActive, setCodecFallbackActive] = useState(false)  // True when browser can't decode video codec
  const codecFallbackStartedRef = useRef(false)  // Ref guard to prevent double-start
  const [streamError, setStreamError] = useState(null)  // Generic stream error toast

  // Cleanup coordination: ensures old backend processes are stopped before new ones start
  const cleanupDoneRef = useRef(true)
  const cleanupGenerationRef = useRef(0)
  const isFirstRenderRef = useRef(true)
  const [cleanupSeq, setCleanupSeq] = useState(0)

  // Refs for callbacks (used by auto-start effect)
  const startSVPStreamRef = useRef(null)
  const startInterpolatedStreamRef = useRef(null)
  const handleQualityChangeRef = useRef(null)


  // Load SVP config on mount (only if addon installed)
  useEffect(() => {
    if (!svpInstalled && !nativeSvpPlayback) {
      setSvpConfig({ enabled: false })
      setSvpConfigLoaded(true)
      return
    }
    async function loadSVPConfig() {
      try {
        const config = await getSVPConfig()
        setSvpConfig(config)
      } catch (err) {
        console.error('Failed to load SVP config:', err)
        setSvpConfig({ enabled: false })
      } finally {
        setSvpConfigLoaded(true)
      }
    }
    loadSVPConfig()
  }, [svpInstalled, nativeSvpPlayback])

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

  // Auto-dismiss stream error toast after 5 seconds
  useEffect(() => {
    if (streamError) {
      const timer = setTimeout(() => {
        setStreamError(null)
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [streamError])

  // Helper to get the current absolute playback time (accounting for stream offsets)
  const getCurrentAbsoluteTime = useCallback(() => {
    const transitionIntent = playbackTransitionOwnerRef.current.currentIntent()
    const imageKey = image?.url ?? image?.file_path ?? image?.id
    if (streamTransitioningRef.current && transitionIntent?.imageKey === imageKey) {
      return transitionIntent.position
    }
    if (!mediaRef.current) return 0
    const hlsTime = mediaRef.current.currentTime
    if (svpStreamUrl) {
      return hlsTime + svpStartOffset
    } else if (transcodeStreamUrl) {
      return hlsTime + transcodeStartOffset
    }
    return hlsTime
  }, [image, mediaRef, svpStreamUrl, svpStartOffset, transcodeStreamUrl, transcodeStartOffset])

  const capturePlaybackIntent = useCallback((position = null) => {
    const video = mediaRef.current
    const absoluteTime = position ?? getCurrentAbsoluteTime()
    const imageKey = image?.url ?? image?.file_path ?? image?.id
    return captureTransitionIntent(video, absoluteTime, imageKey)
  }, [getCurrentAbsoluteTime, image, mediaRef])

  const beginPlaybackTransition = useCallback((intent = null, options = {}) => {
    const owner = playbackTransitionOwnerRef.current.begin(
      intent || capturePlaybackIntent(),
      options,
    )
    streamTransitioningRef.current = true
    owner.intent.media?.pause?.()
    return owner
  }, [capturePlaybackIntent])

  const isPlaybackTransitionCurrent = useCallback((owner) => {
    if (!owner) return false
    const imageKey = imageRef.current?.url ?? imageRef.current?.file_path ?? imageRef.current?.id
    return playbackTransitionOwnerRef.current.isCurrent(owner, imageKey, owner.intent.media)
      && mediaRef.current === owner.intent.media
  }, [mediaRef])

  const finishPlaybackTransition = useCallback((owner) => {
    if (!playbackTransitionOwnerRef.current.finish(owner)) return false
    streamTransitioningRef.current = false
    return true
  }, [])

  const failOpenMSE = useCallback(async (controller, error, absoluteTime, shouldResume) => {
    if (!controller || svpMseControllerRef.current !== controller) return
    const video = mediaRef.current
    const currentImage = imageRef.current
    if (!video || !currentImage) return

    const resumeAt = Number.isFinite(absoluteTime) ? absoluteTime : 0
    const transitionIntent = captureTransitionIntent(
      video,
      resumeAt,
      currentImage.url ?? currentImage.file_path ?? currentImage.id,
    )
    transitionIntent.shouldPlay = Boolean(shouldResume && !transitionIntent.ended)
    const transition = beginPlaybackTransition(transitionIntent, { reuseActiveIntent: false })

    svpMseControllerRef.current = null
    await controller.close().catch(() => {})
    if (!isPlaybackTransitionCurrent(transition)) return

    setSvpStreamUrl(null)
    setSvpTotalDuration(null)
    setSvpBufferedDuration(0)
    setSvpPendingSeek(null)
    setSvpLoading(false)
    setSvpError(error?.message || 'SVP processing failed; playing the original video')

    let directUrl = getMediaUrl(currentImage.url)
    if (isMobileApp() && isUsingLocalServer() && currentImage.file_path) {
      const assetUrl = getAssetUrl(currentImage.file_path)
      if (assetUrl) directUrl = assetUrl
    }

    const restoreDirectPlayback = () => {
      transition.signal.removeEventListener('abort', cancelRestore)
      if (!isPlaybackTransitionCurrent(transition)) return
      video.currentTime = resumeAt
      finishPlaybackTransition(transition)
      if (transition.intent.shouldPlay) video.play().catch(() => {})
    }
    const cancelRestore = () => video.removeEventListener('loadedmetadata', restoreDirectPlayback)
    transition.signal.addEventListener('abort', cancelRestore, { once: true })
    video.addEventListener('loadedmetadata', restoreDirectPlayback, { once: true })
    video.src = directUrl
    video.load?.()
    if (currentImage.file_path) applyNormalization(currentImage.file_path)
  }, [applyNormalization, mediaRef, beginPlaybackTransition, isPlaybackTransitionCurrent, finishPlaybackTransition])

  // Restart SVP stream from a specific position (for seeking beyond buffered content)
  const restartSVPFromPosition = useCallback(async (targetTime) => {
    if (!svpInstalled || !image || !svpConfig?.enabled) return

    if (mseSvpPlayback && svpMseControllerRef.current) {
      const controller = svpMseControllerRef.current
      const transition = beginPlaybackTransition(capturePlaybackIntent(targetTime), {
        reuseActiveIntent: false,
      })
      const seekToken = ++mseSeekTokenRef.current
      mseSeekOwnerRef.current = transition
      setSvpPendingSeek(targetTime)
      setSvpLoading(true)

      const queuedSeek = mseSeekQueueRef.current.catch(() => {}).then(async () => {
        if (seekToken !== mseSeekTokenRef.current
            || controller !== svpMseControllerRef.current
            || !isPlaybackTransitionCurrent(transition)) return

        const video = mediaRef.current
        try {
          video?.pause()
          await controller.seek(targetTime)
          if (seekToken !== mseSeekTokenRef.current
              || controller !== svpMseControllerRef.current
              || !isPlaybackTransitionCurrent(transition)) return
          if (video) video.currentTime = targetTime
          setSvpPendingSeek(null)
          setSvpLoading(false)
          finishPlaybackTransition(transition)
          if (transition.intent.shouldPlay) await video?.play().catch(() => {})
        } catch (error) {
          if (seekToken !== mseSeekTokenRef.current
              || controller !== svpMseControllerRef.current
              || !isPlaybackTransitionCurrent(transition)) return
          await failOpenMSE(
            controller,
            error,
            targetTime,
            transition.intent.shouldPlay,
          )
        }
      }).finally(() => {
        if (mseSeekOwnerRef.current === transition) mseSeekOwnerRef.current = null
      })
      mseSeekQueueRef.current = queuedSeek
      await queuedSeek
      return
    }

    if (svpRestartingRef.current) {
      svpQueuedRestartRef.current = {
        targetTime,
        imageKey: currentImageKey,
        restart: restartSVPFromPosition,
      }
      setSvpPendingSeek(targetTime)
      setSvpLoading(true)
      console.log(`[SVP] Queued restart from ${targetTime.toFixed(1)}s while another restart is in progress`)
      return
    }

    svpRestartingRef.current = true
    const restartToken = ++svpRestartTokenRef.current
    const transition = beginPlaybackTransition(capturePlaybackIntent(targetTime), {
      reuseActiveIntent: false,
    })
    svpRestartOwnerRef.current = transition
    console.log(`[SVP] Restarting stream from ${targetTime.toFixed(1)}s`)

    // Show loading indicator
    setSvpLoading(true)
    setSvpPendingSeek(targetTime)

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
      const result = await playVideoSVP(
        image.file_path,
        targetTime,
        currentQuality,
        transition.signal,
        transition.generation,
      )

      if (restartToken !== svpRestartTokenRef.current || !isPlaybackTransitionCurrent(transition)) {
        console.log(`[SVP] Ignoring stale restart result for ${targetTime.toFixed(1)}s`)
        return
      }

      const queuedRestart = svpQueuedRestartRef.current
      if (queuedRestart !== null && Math.abs(queuedRestart.targetTime - targetTime) > 0.25) {
        console.log(`[SVP] Ignoring stale restart result for ${targetTime.toFixed(1)}s; queued ${queuedRestart.targetTime.toFixed(1)}s`)
        return
      }
      svpQueuedRestartRef.current = null

      if (result.success && result.stream_url) {
        activeStreamOwnerRef.current = transition
        // Set offset BEFORE stream URL so handleTimeUpdate uses correct offset immediately
        setSvpStartOffset(targetTime)
        if (result.duration) {
          setSvpTotalDuration(result.duration)
        }
        if (result.source_resolution) {
          setSourceResolution(result.source_resolution)
        }
        // Set stream URL last - this triggers HLS setup
        setSvpSourceRevision(revision => nextPlaybackSourceRevision(revision))
        setSvpStreamUrl(result.stream_url)
      } else if (result.success && result.skipped) {
        if (result.source_resolution) setSourceResolution(result.source_resolution)
        setSvpError(result.reason || 'SVP is not needed for this video')
        setSvpLoading(false)
        finishPlaybackTransition(transition)
      } else {
        setSvpError(result.error || 'Failed to restart SVP stream')
        setSvpLoading(false)
        finishPlaybackTransition(transition)
      }
    } catch (err) {
      if (!isPlaybackTransitionCurrent(transition)) return
      if (err?.name !== 'CanceledError' && err?.name !== 'AbortError') {
        console.error('SVP restart error:', err)
        setSvpError(err.message || 'Failed to restart SVP stream')
      }
      setSvpLoading(false)
      finishPlaybackTransition(transition)
    } finally {
      if (svpRestartOwnerRef.current === transition) svpRestartOwnerRef.current = null
      svpRestartingRef.current = false
      const queuedRestart = svpQueuedRestartRef.current
      if (queuedRestart !== null && Math.abs(queuedRestart.targetTime - targetTime) > 0.25) {
        svpQueuedRestartRef.current = null
        queuedRestart.restart(queuedRestart.targetTime)
      } else {
        svpQueuedRestartRef.current = null
      }
    }
  }, [image, currentImageKey, svpConfig, svpInstalled, mseSvpPlayback, mediaRef, failOpenMSE, currentQuality, capturePlaybackIntent, beginPlaybackTransition, isPlaybackTransitionCurrent, finishPlaybackTransition])

  const cancelPendingSVPRestart = useCallback((targetTime = null) => {
    svpRestartTokenRef.current += 1
    svpQueuedRestartRef.current = null
    const mseSeekOwner = mseSeekOwnerRef.current
    if (mseSeekOwner && isPlaybackTransitionCurrent(mseSeekOwner)) {
      mseSeekTokenRef.current += 1
      mseSeekOwnerRef.current = null
      if (Number.isFinite(targetTime)) restartSVPFromPosition(targetTime)
      return
    }
    mseSeekTokenRef.current += 1
    const restartOwner = svpRestartOwnerRef.current
    if (restartOwner && isPlaybackTransitionCurrent(restartOwner)) {
      playbackTransitionOwnerRef.current.invalidate()
      streamTransitioningRef.current = false
    }
    svpRestartOwnerRef.current = null
    setSvpLoading(false)
  }, [isPlaybackTransitionCurrent, restartSVPFromPosition])

  // Restart transcode stream from a specific position (for seeking beyond buffered content)
  const restartTranscodeFromPosition = useCallback(async (targetTime) => {
    if (!image) return

    const transition = beginPlaybackTransition(capturePlaybackIntent(targetTime), {
      reuseActiveIntent: false,
    })
    console.log(`[Transcode] Restarting stream from ${targetTime.toFixed(1)}s`)

    if (transcodeHlsRef.current) {
      transcodeHlsRef.current.destroy()
      transcodeHlsRef.current = null
    }
    setTranscodeStreamUrl(null)
    setTranscodeBufferedDuration(0)

    try {
      const result = await playVideoTranscode(
        image.file_path,
        targetTime,
        currentQuality,
        transition.signal,
        transition.generation,
      )
      if (!isPlaybackTransitionCurrent(transition)) return

      if (result.success && result.stream_url) {
        activeStreamOwnerRef.current = transition
        setTranscodeStartOffset(targetTime)
        if (result.duration) setTranscodeTotalDuration(result.duration)
        if (result.source_resolution) setSourceResolution(result.source_resolution)
        setTranscodeSourceRevision(revision => nextPlaybackSourceRevision(revision))
        setTranscodeStreamUrl(result.stream_url)
      } else {
        console.error('Failed to restart transcode stream:', result.error)
        finishPlaybackTransition(transition)
      }
    } catch (err) {
      if (!isPlaybackTransitionCurrent(transition)) return
      if (err?.name !== 'CanceledError' && err?.name !== 'AbortError') {
        console.error('Transcode restart error:', err)
      }
      finishPlaybackTransition(transition)
    }
  }, [image, currentQuality, capturePlaybackIntent, beginPlaybackTransition, isPlaybackTransitionCurrent, finishPlaybackTransition])

  // Start optical flow interpolation for video (called automatically when enabled)
  const startInterpolatedStream = useCallback(async (startPosition = null) => {
    if (!image || !opticalFlowConfig?.enabled || !isVideo(image.filename)) return
    if (opticalFlowStreamUrl || opticalFlowLoading) return // Already active or starting
    const svpStopGeneration = playbackTransitionOwnerRef.current.currentGeneration() || undefined

    // Get current playback position before stopping other streams
    const playbackPosition = startPosition ?? getCurrentAbsoluteTime()

    // Stop any existing SVP stream
    if (svpStreamUrl) {
      setSvpStreamUrl(null)
      await stopSVPStream(svpStopGeneration)
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
  const startSVPStream = useCallback(async (
    startPosition = null,
    playbackIntent = null,
    enabledOverride = false,
    qualityOverride = null,
    replaceLoading = false,
  ) => {
    if (!svpInstalled) return  // Addon not installed
    console.log('[startSVPStream] Called', { image: image?.id, enabled: svpConfig?.enabled, isVideo: isVideo(image?.filename), startPosition })
    if (!image || (!svpConfig?.enabled && !enabledOverride) || !isVideo(image.filename)) {
      console.log('[startSVPStream] Early return: missing image/config/not video')
      return
    }
    if (svpStreamUrl || (svpLoading && !replaceLoading)) {
      console.log('[startSVPStream] Early return: already active or loading', { svpStreamUrl, svpLoading })
      return
    }

    if (svpStartingRef.current) {
      console.log('[startSVPStream] Early return: ref lock active')
      return
    }
    const startAttempt = {}
    svpStartingRef.current = startAttempt

    const requestedIntent = playbackIntent || playbackTransitionOwnerRef.current.currentIntent() || capturePlaybackIntent(startPosition)
    const transition = beginPlaybackTransition(requestedIntent)
    const playbackPosition = transition.intent.position
    let mseController = null
    const mseShouldResume = transition.intent.shouldPlay

    try {
      if (opticalFlowStreamUrl) {
        setOpticalFlowStreamUrl(null)
        await stopInterpolatedStream()
        if (!isPlaybackTransitionCurrent(transition)) return
      }

      if (transcodeStreamUrl) {
        setTranscodeStreamUrl(null)
        await stopTranscodeStream()
        if (!isPlaybackTransitionCurrent(transition)) return
      }

      setSvpLoading(true)
      setSvpError(null)

      console.log('[startSVPStream] Calling API with path:', image.file_path, 'position:', playbackPosition, 'quality:', qualityOverride ?? currentQuality)
      if (mseSvpPlayback && mediaRef.current) {
        const video = mediaRef.current
        let playbackEstablished = false
        const ownsMseSession = () => (
          svpMseControllerRef.current === mseController
          && mediaRef.current === video
          && (imageRef.current?.url ?? imageRef.current?.file_path ?? imageRef.current?.id) === transition.intent.imageKey
        )
        video.pause()
        mseController = new MSESessionController(video, svpMSEClient, {
          onBuffer: duration => {
            if (ownsMseSession()) setSvpBufferedDuration(duration)
          },
          onMetadata: metadata => {
            if (!ownsMseSession()) return
            setSvpTotalDuration(metadata.source_duration)
            setSourceResolution({ width: metadata.width, height: metadata.height })
          },
          onError: error => {
            if (!ownsMseSession()) return
            console.error('[SVP MSE] Processing error:', error)
            const resumeDirect = playbackEstablished ? !video.paused : mseShouldResume
            failOpenMSE(mseController, error, video.currentTime || playbackPosition, resumeDirect)
          }
        })
        svpMseControllerRef.current = mseController
        await mseController.open(image.file_path, playbackPosition, 1)
        if (!isPlaybackTransitionCurrent(transition)) {
          await mseController.close().catch(() => {})
          return
        }
        hadSvpStreamRef.current = true
        activeStreamOwnerRef.current = transition
        resetGain()
        setSvpStartOffset(0)
        setSvpSourceRevision(revision => nextPlaybackSourceRevision(revision))
        setSvpStreamUrl('mse://svp-session')
        video.currentTime = playbackPosition
        if (mseShouldResume) await video.play().catch(() => {})
        playbackEstablished = true
        setSvpLoading(false)
        finishPlaybackTransition(transition)
        return
      }
      const result = await playVideoSVP(
        image.file_path,
        playbackPosition,
        qualityOverride ?? currentQuality,
        transition.signal,
        transition.generation,
      )
      if (!isPlaybackTransitionCurrent(transition)) return
      console.log('[startSVPStream] API result:', result)

      if (result.success && result.stream_url) {
        hadSvpStreamRef.current = true
        activeStreamOwnerRef.current = transition
        resetGain()  // SVP FFmpeg already normalizes audio
        setSvpStartOffset(playbackPosition)
        if (result.duration) setSvpTotalDuration(result.duration)
        if (result.source_resolution) setSourceResolution(result.source_resolution)
        setSvpSourceRevision(revision => nextPlaybackSourceRevision(revision))
        setSvpStreamUrl(result.stream_url)
      } else if (result.success && result.skipped) {
        if (result.source_resolution) setSourceResolution(result.source_resolution)
        setSvpError(result.reason || 'SVP is not needed for this video')
        setSvpLoading(false)
        finishPlaybackTransition(transition)
      } else {
        setSvpError(result.error || 'Failed to start SVP playback')
        setSvpLoading(false)
        finishPlaybackTransition(transition)
      }
    } catch (err) {
      if (!isPlaybackTransitionCurrent(transition)) return
      console.error('SVP error:', err)
      if (mseController) {
        await failOpenMSE(
          mseController,
          err,
          playbackPosition,
          mseShouldResume
        )
      } else if (err?.name !== 'CanceledError' && err?.name !== 'AbortError') {
        setSvpError(err.message || 'Failed to start SVP playback')
        setSvpLoading(false)
        finishPlaybackTransition(transition)
      }
    } finally {
      if (svpStartingRef.current === startAttempt) svpStartingRef.current = null
    }
    // svpLoading stays true until the owning HLS source has buffered playable media.
  }, [image, svpInstalled, svpConfig, svpStreamUrl, svpLoading, opticalFlowStreamUrl, transcodeStreamUrl, currentQuality, mseSvpPlayback, mediaRef, resetGain, failOpenMSE, capturePlaybackIntent, beginPlaybackTransition, isPlaybackTransitionCurrent, finishPlaybackTransition])

  // Update refs after callbacks are defined (used by auto-start effect and SVP menu restart)
  startSVPStreamRef.current = startSVPStream
  startInterpolatedStreamRef.current = startInterpolatedStream

  useEffect(() => {
    if (svpStreamUrl !== 'mse://svp-session' || !mediaRef.current) return
    const video = mediaRef.current
    const controller = svpMseControllerRef.current
    if (!controller) return

    const pauseProcessing = () => {
      if (streamTransitioningRef.current) return
      controller.pause().catch(error => {
        failOpenMSE(controller, error, video.currentTime, false)
      })
    }
    const resumeProcessing = () => {
      if (streamTransitioningRef.current) return
      controller.resume().catch(error => {
        failOpenMSE(controller, error, video.currentTime, true)
      })
    }

    video.addEventListener('pause', pauseProcessing)
    video.addEventListener('play', resumeProcessing)
    return () => {
      video.removeEventListener('pause', pauseProcessing)
      video.removeEventListener('play', resumeProcessing)
    }
  }, [failOpenMSE, mediaRef, svpStreamUrl])

  // Cleanup old streams when image changes.
  // MUST be declared BEFORE the auto-start effect so React runs it first.
  // This ensures backend processes (FFmpeg, SVP) are stopped before new ones start.
  useEffect(() => {
    if (isFirstRenderRef.current) {
      isFirstRenderRef.current = false
      return  // No cleanup needed on first mount
    }

    const cleanupGeneration = ++cleanupGenerationRef.current
    const playbackGeneration = playbackTransitionOwnerRef.current.invalidate()
    activeStreamOwnerRef.current = null
    svpRestartTokenRef.current += 1
    mseSeekTokenRef.current += 1
    svpRestartOwnerRef.current = null
    svpQueuedRestartRef.current = null
    svpRestartingRef.current = false
    cleanupDoneRef.current = false

    // Synchronous: destroy HLS instances immediately to stop network requests
    if (hlsRef.current) { hlsRef.current.destroy(); hlsRef.current = null }
    if (svpHlsRef.current) { svpHlsRef.current.destroy(); svpHlsRef.current = null }
    if (transcodeHlsRef.current) { transcodeHlsRef.current.destroy(); transcodeHlsRef.current = null }
    const mseCleanup = svpMseControllerRef.current?.close().catch(() => {})
    svpMseControllerRef.current = null

    // Reset all streaming state for the new video
    resetGain()
    setOpticalFlowError(null)
    setOpticalFlowStreamUrl(null)
    setOpticalFlowLoading(false)
    setStreamError(null)
    setSvpError(null)
    setSvpStreamUrl(null)
    setSvpTotalDuration(null)
    setSvpBufferedDuration(0)
    setSvpPendingSeek(null)
    setSvpStartOffset(0)
    setSvpLoading(false)
    svpStartingRef.current = null
    streamTransitioningRef.current = false
    setCodecFallbackActive(false)
    codecFallbackStartedRef.current = false
    setSourceResolution(null)
    setTranscodeTotalDuration(null)
    setTranscodeStartOffset(0)
    setTranscodeBufferedDuration(0)
    setTranscodeStreamUrl(null)

    // Async: stop all backend processes, then signal auto-start can proceed
    Promise.all([
      mseCleanup,
      stopSVPStream(playbackGeneration).catch(() => {}),
      stopInterpolatedStream().catch(() => {}),
      stopTranscodeStream().catch(() => {})
    ]).then(() => {
      if (cleanupGeneration !== cleanupGenerationRef.current) return
      cleanupDoneRef.current = true
      setCleanupSeq(s => s + 1)
    })
  }, [currentImageKey])

  // Native GTK owns playback exclusively. Tear down every browser/HLS producer
  // as soon as native ownership is selected so two decoders cannot run.
  useEffect(() => {
    if (enabled) return
    const cleanupGeneration = ++cleanupGenerationRef.current
    cleanupDoneRef.current = false
    const playbackGeneration = playbackTransitionOwnerRef.current.invalidate()
    activeStreamOwnerRef.current = null
    svpRestartTokenRef.current += 1
    mseSeekTokenRef.current += 1
    svpRestartOwnerRef.current = null
    svpQueuedRestartRef.current = null
    svpRestartingRef.current = false
    svpStartingRef.current = null
    streamTransitioningRef.current = false
    setSvpStreamUrl(null)
    setSvpLoading(false)
    setTranscodeStreamUrl(null)
    setOpticalFlowStreamUrl(null)
    if (hlsRef.current) { hlsRef.current.destroy(); hlsRef.current = null }
    if (svpHlsRef.current) { svpHlsRef.current.destroy(); svpHlsRef.current = null }
    if (transcodeHlsRef.current) { transcodeHlsRef.current.destroy(); transcodeHlsRef.current = null }
    const mseCleanup = svpMseControllerRef.current?.close().catch(() => {})
    svpMseControllerRef.current = null
    resetGain()
    Promise.all([
      mseCleanup,
      stopSVPStream(playbackGeneration).catch(() => {}),
      stopInterpolatedStream().catch(() => {}),
      stopTranscodeStream().catch(() => {}),
    ]).then(() => {
      if (cleanupGeneration !== cleanupGenerationRef.current) return
      cleanupDoneRef.current = true
      setCleanupSeq(s => s + 1)
    })
  }, [enabled, resetGain])

  // Stop SVP stream
  const stopSVP = useCallback(async (transitionGeneration = null) => {
    const generation = transitionGeneration
      || playbackTransitionOwnerRef.current.currentGeneration()
      || undefined
    if (svpHlsRef.current) {
      svpHlsRef.current.destroy()
      svpHlsRef.current = null
    }
    const mseCleanup = svpMseControllerRef.current?.close().catch(() => {})
    svpMseControllerRef.current = null
    setSvpStreamUrl(null)
    setSvpTotalDuration(null)
    setSvpBufferedDuration(0)
    setSvpPendingSeek(null)
    setSvpError(null)
    await Promise.all([mseCleanup, stopSVPStream(generation)])
  }, [])

  // Cancel SVP while it's loading/buffering — stops the stream and disables SVP in-memory
  const cancelSVPLoading = useCallback(() => {
    const playbackIntent = playbackTransitionOwnerRef.current.currentIntent() || capturePlaybackIntent()
    playbackTransitionOwnerRef.current.invalidate()
    activeStreamOwnerRef.current = null
    svpRestartTokenRef.current += 1
    svpQueuedRestartRef.current = null
    svpStartingRef.current = null
    svpRestartingRef.current = false
    setSvpLoading(false)
    setSvpPendingSeek(null)
    if (svpHlsRef.current) {
      svpHlsRef.current.destroy()
      svpHlsRef.current = null
    }
    setSvpStreamUrl(null)
    setSvpTotalDuration(null)
    setSvpBufferedDuration(0)
    setSvpError(null)
    setSvpConfig(prev => prev ? { ...prev, enabled: false } : prev)
    svpMseControllerRef.current?.close().catch(() => {})
    svpMseControllerRef.current = null
    handleQualityChangeRef.current?.('original', playbackIntent)
  }, [capturePlaybackIntent])

  // Use a ref for currentQuality so the auto-start effect doesn't re-fire on quality changes.
  // Quality changes are handled by handleQualityChange — auto-start only handles new image opens.
  const currentQualityRef = useRef(currentQuality)
  currentQualityRef.current = currentQuality

  // Auto-start interpolated stream when video opens
  // Priority: SVP (if enabled) > Optical Flow (if enabled) > Transcode (if quality != original)
  // NOTE: currentQuality is intentionally NOT a dependency — quality changes are handled by handleQualityChange
  useEffect(() => {
    if (!enabled) return
    // Wait for cleanup to complete before starting new streams. Without this,
    // stop_all_*_streams() from cleanup could kill the newly started stream.
    if (!cleanupDoneRef.current) return

    // Wait for the sole supported interpolation engine to load.
    // Without this gate, the effect fires 3 times during startup:
    //   1. initial mount (both configs null)
    //   2. svpConfig loads (undefined→false)
    //   3. opticalFlowConfig loads (undefined→false)
    // Each fire starts a transcode that kills the previous one via stop_all_transcode_streams().
    if (svpConfig === null) return

    if (image && isVideo(image.filename)) {
      const quality = currentQualityRef.current
      console.log('[Auto-start] Checking...', {
        svpEnabled: svpConfig?.enabled,
        opticalFlowEnabled: opticalFlowConfig?.enabled,
        currentQuality: quality
      })
      // Desktop LocalBooru uses the original WebKit player with the
      // Manager-controlled GStreamer/VapourSynth filter. Do not start the
      // retired local HLS producer in that mode.
      if (svpConfig?.enabled && nativeSvpPlayback) {
        if (image.file_path) applyNormalization(image.file_path)
      }
      // Remote/mobile clients retain the existing streaming route.
      else if (svpConfig?.enabled) {
        console.log('[Auto-start] Starting SVP stream...')
        startSVPStreamRef.current()
      }
      // If quality is not original, use transcode
      else if ((!transcodeStreamUrl) && quality !== 'original') {
        console.log('[Auto-start] Starting transcode stream for quality:', quality)
        handleQualityChangeRef.current?.(quality, capturePlaybackIntent())
      }
      // Otherwise play direct (original quality, no interpolation) — apply Web Audio gain
      else if (image.file_path) {
        applyNormalization(image.file_path)
      }
    }
  }, [enabled, currentImageKey, cleanupSeq, svpConfig?.enabled, nativeSvpPlayback])

  // Setup HLS player when optical flow stream is active
  useEffect(() => {
    if (!opticalFlowStreamUrl || !mediaRef.current) return

    const video = mediaRef.current
    let cancelled = false
    let playOnReady = null
    let shouldAutoResume = !video.paused
    let manuallyPaused = false

    const playOpticalFlowVideo = () => {
      if (cancelled || !shouldAutoResume || manuallyPaused) return
      video.play().catch(() => {})
    }

    const trackManualPause = () => {
      if (!streamTransitioningRef.current) {
        manuallyPaused = true
        shouldAutoResume = false
      }
    }

    const trackManualPlay = () => {
      manuallyPaused = false
      shouldAutoResume = true
    }

    if (Hls.isSupported()) {
      // Pause video during transition to prevent playing old buffered content
      video.pause()
      video.addEventListener('pause', trackManualPause)
      video.addEventListener('play', trackManualPlay)

      // Cleanup previous instance - destroy() is synchronous and handles cleanup
      if (hlsRef.current) {
        hlsRef.current.destroy()
        hlsRef.current = null
      }

      // Remove direct video src (if any) - HLS will use MediaSource instead
      video.removeAttribute('src')

      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: false,
        startPosition: 0,
        backBufferLength: 30,
        maxBufferLength: 30,
        maxMaxBufferLength: 60,
      })

      // Use getMediaUrl to handle dev mode (different ports for frontend/backend)
      hls.loadSource(getMediaUrl(opticalFlowStreamUrl))
      hls.attachMedia(video)

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (cancelled) return
        if (video.readyState >= 3) {
          playOpticalFlowVideo()
        } else {
          playOnReady = () => {
            playOpticalFlowVideo()
            video.removeEventListener('canplay', playOnReady)
          }
          video.addEventListener('canplay', playOnReady)
        }
      })

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (cancelled) return
        if (data.fatal) {
          console.error('HLS fatal error:', data)
          hls.destroy()
          hlsRef.current = null
          setOpticalFlowError('Stream playback error.')
          setOpticalFlowStreamUrl(null)
        }
      })

      hlsRef.current = hls
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari/iOS native HLS support
      video.src = getMediaUrl(opticalFlowStreamUrl)
      video.addEventListener('loadedmetadata', () => {
        playOpticalFlowVideo()
      })
    } else {
      setOpticalFlowError('HLS playback is not supported in this browser')
      setOpticalFlowStreamUrl(null)
    }

    return () => {
      cancelled = true
      if (playOnReady) {
        video.removeEventListener('canplay', playOnReady)
      }
      video.removeEventListener('pause', trackManualPause)
      video.removeEventListener('play', trackManualPlay)
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
    const transition = activeStreamOwnerRef.current
    const ownsPlayback = () => !transition || isPlaybackTransitionCurrent(transition)
    let cancelled = false
    let playOnReady = null
    let nativeLoadedMetadata = null
    let nativeError = null
    let ownedHls = null
    let shouldAutoResume = transition?.intent.shouldPlay ?? !video.paused
    let manuallyPaused = false

    const playTranscodeVideo = () => {
      if (cancelled || !shouldAutoResume || manuallyPaused) return
      video.play().catch(() => {})
    }

    const trackManualPause = () => {
      if (!streamTransitioningRef.current) {
        manuallyPaused = true
        shouldAutoResume = false
      }
    }

    const trackManualPlay = () => {
      manuallyPaused = false
      shouldAutoResume = true
    }

    if (Hls.isSupported()) {
      // Pause video during transition to prevent playing old buffered content
      video.pause()
      video.addEventListener('pause', trackManualPause)
      video.addEventListener('play', trackManualPlay)

      // Cleanup previous instance - destroy() is synchronous and handles cleanup
      if (transcodeHlsRef.current) {
        transcodeHlsRef.current.destroy()
        transcodeHlsRef.current = null
      }

      // Remove direct video src (if any) - HLS will use MediaSource instead
      video.removeAttribute('src')

      const hls = new Hls({
        enableWorker: true,
        // Don't use lowLatencyMode — it causes hls.js to start at the live edge
        // of our growing HLS stream instead of segment 0, causing playback stalls
        lowLatencyMode: false,
        startPosition: 0,
        backBufferLength: 30,
        maxBufferLength: 30,
        maxMaxBufferLength: 60,
      })
      ownedHls = hls

      const transcodeUrl = getMediaUrl(transcodeStreamUrl)
      let manifestRetries = 0

      // Use getMediaUrl to handle dev mode (different ports for frontend/backend)
      hls.loadSource(transcodeUrl)
      hls.attachMedia(video)

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (cancelled || !ownsPlayback()) return
        if (transition) finishPlaybackTransition(transition)
        // Wait for enough data to be buffered before playing — calling play()
        // at MANIFEST_PARSED often fails because no segments are buffered yet
        if (video.readyState >= 3) {
          playTranscodeVideo()
        } else {
          playOnReady = () => {
            playTranscodeVideo()
            video.removeEventListener('canplay', playOnReady)
          }
          video.addEventListener('canplay', playOnReady)
        }
      })

      // Track available duration from HLS manifest for seek handling
      hls.on(Hls.Events.LEVEL_UPDATED, (event, data) => {
        if (cancelled || !ownsPlayback()) return
        const levelDetails = data.details
        if (levelDetails && levelDetails.totalduration) {
          setTranscodeBufferedDuration(levelDetails.totalduration)
        }
      })

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (cancelled || !ownsPlayback()) return
        if (data.fatal) {
          // HLS.js refuses to retry 404s — manually retry for manifest not-ready
          if (data.details === 'manifestLoadError' && data.response?.code === 404 && manifestRetries < 15) {
            manifestRetries++
            console.log(`[Transcode] Playlist not ready, retry ${manifestRetries}/15...`)
            setTimeout(() => {
              if (transcodeHlsRef.current === hls) {
                hls.loadSource(transcodeUrl)
              }
            }, 1000)
            return
          }
          console.error('Transcode HLS fatal error:', data)
          hls.destroy()
          if (transcodeHlsRef.current === hls) transcodeHlsRef.current = null
          setTranscodeStreamUrl(null)
          stopTranscodeStream().catch(() => {})
          handleQualityChangeRef.current?.('original', transition?.intent || capturePlaybackIntent())
        }
      })

      transcodeHlsRef.current = hls
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari/iOS native HLS support
      nativeLoadedMetadata = () => {
        if (cancelled || !ownsPlayback()) return
        if (transition) finishPlaybackTransition(transition)
        playTranscodeVideo()
      }
      nativeError = () => {
        if (cancelled || !ownsPlayback()) return
        setTranscodeStreamUrl(null)
        stopTranscodeStream().catch(() => {})
        handleQualityChangeRef.current?.('original', transition?.intent || capturePlaybackIntent())
      }
      video.src = getMediaUrl(transcodeStreamUrl)
      video.addEventListener('loadedmetadata', nativeLoadedMetadata, { once: true })
      video.addEventListener('error', nativeError, { once: true })
    } else {
      console.error('HLS playback is not supported')
      setTranscodeStreamUrl(null)
      stopTranscodeStream().catch(() => {})
      handleQualityChangeRef.current?.('original', transition?.intent || capturePlaybackIntent())
    }

    return () => {
      cancelled = true
      if (playOnReady) {
        video.removeEventListener('canplay', playOnReady)
      }
      if (nativeLoadedMetadata) {
        video.removeEventListener('loadedmetadata', nativeLoadedMetadata)
      }
      if (nativeError) {
        video.removeEventListener('error', nativeError)
      }
      video.removeEventListener('pause', trackManualPause)
      video.removeEventListener('play', trackManualPlay)
      if (ownedHls) {
        if (transcodeHlsRef.current === ownedHls) transcodeHlsRef.current = null
        ownedHls.destroy()
      }
    }
  }, [transcodeStreamUrl, transcodeSourceRevision, mediaRef, isPlaybackTransitionCurrent, finishPlaybackTransition, capturePlaybackIntent])

  // Setup HLS player when SVP stream URL is available
  // Keep normal video playing until HLS is ready, then switch
  useEffect(() => {
    if (!svpStreamUrl || svpStreamUrl === 'mse://svp-session' || !mediaRef.current) return

    const video = mediaRef.current
    const transition = activeStreamOwnerRef.current
    const ownsPlayback = () => !transition || isPlaybackTransitionCurrent(transition)
    let cancelled = false
    let startupTimer = null
    let playOnCanPlay = null
    let nativeLoadedMetadata = null
    let nativeError = null
    let ownedHls = null
    let trackManualPause = null
    let trackManualPlay = null
    let stallTimer = null
    let handleStall = null
    let hasLoadedFragment = false
    let hasBufferedFragment = false
    let shouldAutoResume = transition?.intent.shouldPlay ?? !video.paused
    let manuallyPaused = false

    const playSVPVideo = () => {
      if (cancelled || !shouldAutoResume || manuallyPaused) return
      video.play().catch((err) => {
        console.warn('[SVP HLS] play() failed:', err?.message || err)
      })
    }

    const isTimeBuffered = (time) => {
      const ranges = video.buffered
      for (let i = 0; i < ranges.length; i++) {
        if (time >= ranges.start(i) && time <= ranges.end(i)) return true
      }
      return false
    }

    const bufferedAhead = () => {
      const ranges = video.buffered
      for (let i = 0; i < ranges.length; i++) {
        if (ranges.start(i) <= video.currentTime && ranges.end(i) > video.currentTime) {
          return ranges.end(i) - video.currentTime
        }
      }
      return 0
    }

    if (Hls.isSupported()) {
      // Pause video during transition to prevent playing old buffered content
      video.pause()

      trackManualPause = () => {
        if (!streamTransitioningRef.current) {
          manuallyPaused = true
          shouldAutoResume = false
        }
      }
      trackManualPlay = () => {
        manuallyPaused = false
        shouldAutoResume = true
      }
      video.addEventListener('pause', trackManualPause)
      video.addEventListener('play', trackManualPlay)

      // Cleanup previous instance - destroy() is synchronous and handles cleanup
      if (svpHlsRef.current) {
        svpHlsRef.current.destroy()
        svpHlsRef.current = null
      }

      // Remove direct video src (if any) - HLS will use MediaSource instead
      video.removeAttribute('src')

      const hls = new Hls({
        enableWorker: true,
        autoStartLoad: true,
        lowLatencyMode: false,
        startPosition: 0,
        backBufferLength: 600,
        maxBufferLength: 120,
        maxMaxBufferLength: 600,
        // Retry manifest loading while SVP produces initial segments
        manifestLoadingMaxRetry: 30,
        manifestLoadingRetryDelay: 500,
        manifestLoadingMaxRetryTimeout: 60000,
        levelLoadingMaxRetry: 10,
        levelLoadingRetryDelay: 500,
        fragLoadingMaxRetry: 10,
        fragLoadingRetryDelay: 500,
      })
      ownedHls = hls

      // Attach media and load source
      // Use getMediaUrl to handle dev mode (different ports for frontend/backend)
      const fullStreamUrl = getMediaUrl(svpStreamUrl)
      console.log('[SVP HLS] Stream URL:', fullStreamUrl)
      hls.attachMedia(video)

      hls.on(Hls.Events.MEDIA_ATTACHED, () => {
        if (cancelled || !ownsPlayback()) return
        console.log('[SVP HLS] Media attached, loading source')
        hls.loadSource(fullStreamUrl)
        hls.startLoad(0)
      })

      // Debug logging for HLS events
      hls.on(Hls.Events.MANIFEST_LOADING, () => {
        console.log('[SVP HLS] Loading manifest...')
      })

      hls.on(Hls.Events.MANIFEST_LOADED, (event, data) => {
        console.log('[SVP HLS] Manifest loaded:', data.levels?.length, 'levels')
      })

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        if (cancelled || !ownsPlayback()) return
        console.log('[SVP HLS] Manifest parsed, waiting for media buffer')
        hls.startLoad(0)
      })

      hls.on(Hls.Events.FRAG_LOADED, (event, data) => {
        if (cancelled || !ownsPlayback()) return
        hasLoadedFragment = true
        console.log('[SVP HLS] Fragment loaded:', data.frag.sn)
      })

      hls.on(Hls.Events.FRAG_BUFFERED, (event, data) => {
        if (cancelled || !ownsPlayback()) return
        hasBufferedFragment = true
        console.log('[SVP HLS] Fragment buffered:', data.frag.sn)
        if (startupTimer) {
          clearTimeout(startupTimer)
          startupTimer = null
        }
        setSvpPendingSeek(null)
        setSvpLoading(false)
        if (transition) finishPlaybackTransition(transition)
        if (shouldStartSVPPlayback(bufferedAhead())) playSVPVideo()
      })

      playOnCanPlay = () => {
        if (cancelled || !ownsPlayback()) return
        console.log('[SVP HLS] Video canplay')
        if (startupTimer) {
          clearTimeout(startupTimer)
          startupTimer = null
        }
        if (stallTimer) {
          clearTimeout(stallTimer)
          stallTimer = null
        }
        setSvpPendingSeek(null)
        setSvpLoading(false)
        if (transition) finishPlaybackTransition(transition)
        if (shouldStartSVPPlayback(bufferedAhead())) playSVPVideo()
      }
      video.addEventListener('canplay', playOnCanPlay)

      handleStall = () => {
        if (cancelled || !ownsPlayback() || streamTransitioningRef.current) return
        if (stallTimer) clearTimeout(stallTimer)
        const stalledHlsTime = video.currentTime || 0
        const stalledAbsoluteTime = svpStartOffset + stalledHlsTime

        stallTimer = setTimeout(() => {
          if (cancelled || !ownsPlayback() || streamTransitioningRef.current) return
          if (!shouldRestartStalledSVP({
            hasBufferedFragment,
            readyState: video.readyState,
            isBuffered: isTimeBuffered(video.currentTime || 0),
          })) return

          console.warn('[SVP HLS] Stalled outside buffered range, restarting stream', {
            hlsTime: video.currentTime,
            absoluteTime: stalledAbsoluteTime,
            readyState: video.readyState,
            buffered: Array.from({ length: video.buffered.length }, (_, i) => [
              video.buffered.start(i),
              video.buffered.end(i),
            ]),
          })
          setSvpPendingSeek(stalledAbsoluteTime)
          setSvpLoading(true)
          restartSVPFromPosition(stalledAbsoluteTime)
        }, 2500)
      }
      video.addEventListener('waiting', handleStall)
      video.addEventListener('stalled', handleStall)

      // Track available duration from HLS manifest for seek handling
      hls.on(Hls.Events.LEVEL_UPDATED, (event, data) => {
        if (cancelled || !ownsPlayback()) return
        const levelDetails = data.details
        if (levelDetails && levelDetails.totalduration) {
          const availableDuration = levelDetails.totalduration
          console.log('[SVP HLS] Level updated, available duration:', availableDuration.toFixed(1) + 's')
          if (Number.isFinite(availableDuration) && availableDuration > 0) {
            setSvpBufferedDuration(availableDuration)
          }
        }
      })

      let retryCount = 0
      let mediaRecoverCount = 0
      const maxRetries = 10  // Cap retries to avoid infinite retry loops
      const maxMediaRecoveries = 3
      startupTimer = setTimeout(() => {
        if (cancelled || !ownsPlayback() || video.readyState >= 3 || hasBufferedFragment) return
        console.warn('[SVP HLS] Still waiting for playable media', {
          readyState: video.readyState,
          buffered: video.buffered.length,
          hasLoadedFragment,
          hasBufferedFragment,
          hlsState: hls.constructor?.version ? `hls.js ${hls.constructor.version}` : 'hls.js'
        })
        if (!hasLoadedFragment) {
          hls.startLoad(0)
        }
      }, 60000)

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (cancelled || !ownsPlayback()) return

        if (data.fatal) {
          // Check if it's a retryable network/manifest error during startup
          const isManifestError = data.details === 'manifestLoadError' ||
                                   data.details === 'manifestParsingError'
          const isNetworkError = data.type === Hls.ErrorTypes.NETWORK_ERROR
          const isMediaError = data.type === Hls.ErrorTypes.MEDIA_ERROR
          const isBufferAppendError = data.details === 'bufferAppendError' ||
                                      data.details === 'bufferAppendingError'

          if ((isManifestError || isNetworkError) && retryCount < maxRetries) {
            retryCount++
            // Exponential backoff: 1s, 2s, 4s, 8s... capped at 30s
            const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 30000)
            console.log(`[SVP HLS] Fatal error, manual retry ${retryCount}/${maxRetries} in ${delay}ms:`, data.details)
            // HLS.js stops after fatal error - must manually restart loading
            setTimeout(() => {
              if (!cancelled && ownsPlayback()) {
                hls.startLoad()
              }
            }, delay)
          } else if ((isMediaError || isBufferAppendError) && mediaRecoverCount < maxMediaRecoveries) {
            mediaRecoverCount++
            console.warn(`[SVP HLS] Recovering media error ${mediaRecoverCount}/${maxMediaRecoveries}:`, data.details)
            setSvpLoading(false)
            try {
              hls.recoverMediaError()
            } catch (err) {
              console.warn('[SVP HLS] recoverMediaError failed, restarting load:', err?.message || err)
              hls.startLoad(0)
            }
            playSVPVideo()
          } else {
            // Give up - either not a retryable error, or retries exhausted
            console.error('[SVP HLS] Giving up after fatal error:', data)
            const errorMsg = retryCount >= maxRetries
              ? 'SVP stream failed to start (timeout)'
              : `SVP stream error: ${data.details || 'playback failed'}`
            setSvpError(errorMsg)
            setSvpStreamUrl(null)
            setSvpTotalDuration(null)
            setSvpBufferedDuration(0)
            setSvpPendingSeek(null)
            setSvpLoading(false)
            stopSVPStream(transition?.generation).catch(() => {})
            handleQualityChangeRef.current?.('original', transition?.intent || capturePlaybackIntent())
          }
        } else {
          // Non-fatal error - log but don't stop
          console.warn('[SVP HLS] Non-fatal error:', data.details)
        }
      })

      svpHlsRef.current = hls
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari/iOS native HLS support
      nativeLoadedMetadata = () => {
        if (cancelled || !ownsPlayback()) return
        if (transition) finishPlaybackTransition(transition)
        playSVPVideo()
        setSvpLoading(false)
      }
      nativeError = () => {
        if (cancelled || !ownsPlayback()) return
        setSvpError('SVP stream playback failed')
        setSvpStreamUrl(null)
        setSvpLoading(false)
        stopSVPStream(transition?.generation).catch(() => {})
        handleQualityChangeRef.current?.('original', transition?.intent || capturePlaybackIntent())
      }
      video.src = getMediaUrl(svpStreamUrl)
      video.addEventListener('loadedmetadata', nativeLoadedMetadata, { once: true })
      video.addEventListener('error', nativeError, { once: true })
    } else {
      setSvpError('HLS playback is not supported in this browser')
      setSvpStreamUrl(null)
      setSvpTotalDuration(null)
      setSvpBufferedDuration(0)
      setSvpPendingSeek(null)
      setSvpLoading(false)
      stopSVPStream(transition?.generation).catch(() => {})
      handleQualityChangeRef.current?.('original', transition?.intent || capturePlaybackIntent())
    }

    return () => {
      cancelled = true
      if (playOnCanPlay) {
        video.removeEventListener('canplay', playOnCanPlay)
      }
      if (nativeLoadedMetadata) {
        video.removeEventListener('loadedmetadata', nativeLoadedMetadata)
      }
      if (nativeError) {
        video.removeEventListener('error', nativeError)
      }
      if (trackManualPause) {
        video.removeEventListener('pause', trackManualPause)
      }
      if (trackManualPlay) {
        video.removeEventListener('play', trackManualPlay)
      }
      if (handleStall) {
        video.removeEventListener('waiting', handleStall)
        video.removeEventListener('stalled', handleStall)
      }
      if (startupTimer) clearTimeout(startupTimer)
      if (stallTimer) clearTimeout(stallTimer)
      if (ownedHls) {
        if (svpHlsRef.current === ownedHls) svpHlsRef.current = null
        ownedHls.destroy()
      }
    }
  }, [svpStreamUrl, svpSourceRevision, mediaRef, isPlaybackTransitionCurrent, finishPlaybackTransition, restartSVPFromPosition, svpStartOffset, capturePlaybackIntent])

  // Cleanup HLS on unmount
  useEffect(() => {
    return () => {
      const playbackGeneration = playbackTransitionOwnerRef.current.invalidate()
      activeStreamOwnerRef.current = null
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
      // Stop SVP unconditionally: a /play request may still be in flight even
      // before hadSvpStreamRef is set, and leaving vspipe alive can poison the
      // next stream start.
      svpMseControllerRef.current?.close().catch(() => {})
      svpMseControllerRef.current = null
      stopSVPStream(playbackGeneration).catch(() => {})
      if (hadOpticalFlowStreamRef.current) stopInterpolatedStream().catch(() => {})
      if (hadTranscodeStreamRef.current) stopTranscodeStream().catch(() => {})
    }
  }, [])

  // Reset streaming state (called when image changes)
  const resetStreamingState = useCallback(async (shouldStopStreams = false) => {
    // Reset optical flow state for new video
    setOpticalFlowError(null)
    setOpticalFlowStreamUrl(null)
    setStreamError(null)
    // Reset SVP state for new video
    setSvpError(null)
    setSvpStreamUrl(null)
    setSvpTotalDuration(null)
    setSvpBufferedDuration(0)
    setSvpPendingSeek(null)
    setSvpStartOffset(0)
    svpStartingRef.current = null  // Reset lock for new video
    streamTransitioningRef.current = false  // Reset transition flag for new video
    setCodecFallbackActive(false)  // Reset codec fallback for new video
    codecFallbackStartedRef.current = false
    setSourceResolution(null)  // Reset so it gets set from original video, not stream
    setTranscodeTotalDuration(null)
    setTranscodeStartOffset(0)
    setTranscodeBufferedDuration(0)
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
    const mseCleanup = svpMseControllerRef.current?.close().catch(() => {})
    svpMseControllerRef.current = null
    // Stop backend streams if requested
    if (shouldStopStreams) {
      await Promise.all([
        mseCleanup,
        stopSVPStream(playbackTransitionOwnerRef.current.currentGeneration() || undefined).catch(() => {}),
        stopInterpolatedStream().catch(() => {}),
        stopTranscodeStream().catch(() => {})
      ])
    }
  }, [])

  // Handle quality change
  const handleQualityChange = useCallback(async (qualityId, playbackIntent = null) => {
    console.log('[Lightbox] Quality change requested:', qualityId)

    if (!mediaRef.current || !image) {
      console.log('[Lightbox] No media available')
      return
    }

    const transition = beginPlaybackTransition(playbackIntent || capturePlaybackIntent())
    const absoluteTime = transition.intent.position
    console.log('[Lightbox] Current absolute time:', absoluteTime)

    try {
      if (qualityId === 'original') {
        await Promise.all([
          stopSVP(transition.generation).catch(() => {}),
          stopInterpolatedStream().catch(() => {}),
          stopTranscodeStream().catch(() => {}),
        ])
        if (!isPlaybackTransitionCurrent(transition)) return

        if (svpHlsRef.current) { svpHlsRef.current.destroy(); svpHlsRef.current = null }
        if (hlsRef.current) { hlsRef.current.destroy(); hlsRef.current = null }
        if (transcodeHlsRef.current) { transcodeHlsRef.current.destroy(); transcodeHlsRef.current = null }
        activeStreamOwnerRef.current = null
        setSvpStreamUrl(null)
        setSvpStartOffset(0)
        setSvpTotalDuration(null)
        setSvpBufferedDuration(0)
        setSvpPendingSeek(null)
        setSvpLoading(false)
        setOpticalFlowStreamUrl(null)
        setTranscodeStreamUrl(null)
        setTranscodeStartOffset(0)
        setTranscodeTotalDuration(null)
        setTranscodeBufferedDuration(0)

        const video = mediaRef.current
        let directUrl = getMediaUrl(image.url)
        if (isMobileApp() && isUsingLocalServer() && image.file_path) {
          const assetUrl = getAssetUrl(image.file_path)
          if (assetUrl) directUrl = assetUrl
        }
        const restoreDirectPlayback = () => {
          if (!isPlaybackTransitionCurrent(transition)) return
          video.currentTime = absoluteTime
          finishPlaybackTransition(transition)
          if (transition.intent.shouldPlay) video.play().catch(() => {})
        }
        video.addEventListener('loadedmetadata', restoreDirectPlayback, { once: true })
        video.src = directUrl
        video.load?.()
        if (image.file_path) applyNormalization(image.file_path)
        return
      }

      if (svpInstalled && svpConfig?.enabled) {
        if (svpStreamUrl === 'mse://svp-session') {
          await restartSVPFromPosition(absoluteTime)
          if (isPlaybackTransitionCurrent(transition)) finishPlaybackTransition(transition)
          return
        }
        if (!svpStreamUrl) {
          svpStartingRef.current = null
          setSvpLoading(false)
          await startSVPStream(absoluteTime, transition.intent, false, qualityId, true)
          return
        }

        if (svpHlsRef.current) { svpHlsRef.current.destroy(); svpHlsRef.current = null }
        setSvpStreamUrl(null)
        setSvpLoading(true)
        await stopSVPStream(transition.generation)
        if (!isPlaybackTransitionCurrent(transition)) return
        const result = await playVideoSVP(
          image.file_path,
          absoluteTime,
          qualityId,
          transition.signal,
        )
        if (!isPlaybackTransitionCurrent(transition)) return
        if (result.success && result.stream_url) {
          activeStreamOwnerRef.current = transition
          setSvpStartOffset(absoluteTime)
          if (result.duration) setSvpTotalDuration(result.duration)
          if (result.source_resolution) setSourceResolution(result.source_resolution)
          setSvpSourceRevision(revision => nextPlaybackSourceRevision(revision))
          setSvpStreamUrl(result.stream_url)
        } else if (result.success && result.skipped) {
          setSvpError(result.reason || 'SVP is not needed for this video')
          setSvpLoading(false)
          finishPlaybackTransition(transition)
        } else {
          throw new Error(result.error || 'Failed to start SVP stream')
        }
        return
      }

      if (opticalFlowStreamUrl || opticalFlowConfig?.enabled) {
        await stopInterpolatedStream()
        if (!isPlaybackTransitionCurrent(transition)) return
        const result = await playVideoInterpolated(image.file_path, absoluteTime, qualityId)
        if (!isPlaybackTransitionCurrent(transition)) return
        if (!result.success) throw new Error(result.error || 'Failed to start interpolated stream')
        setOpticalFlowStreamUrl(result.stream_url)
        finishPlaybackTransition(transition)
        return
      }

      if (transcodeHlsRef.current) { transcodeHlsRef.current.destroy(); transcodeHlsRef.current = null }
      setTranscodeStreamUrl(null)
      await stopTranscodeStream()
      if (!isPlaybackTransitionCurrent(transition)) return
      const result = await playVideoTranscode(
        image.file_path,
        absoluteTime,
        qualityId,
        transition.signal,
      )
      if (!isPlaybackTransitionCurrent(transition)) return
      if (!result.success || !result.stream_url) {
        throw new Error(result.error || 'Failed to transcode video')
      }
      hadTranscodeStreamRef.current = true
      activeStreamOwnerRef.current = transition
      resetGain()
      setTranscodeStartOffset(absoluteTime)
      if (result.duration) setTranscodeTotalDuration(result.duration)
      if (result.source_resolution) setSourceResolution(result.source_resolution)
      setTranscodeSourceRevision(revision => nextPlaybackSourceRevision(revision))
      setTranscodeStreamUrl(result.stream_url)
    } catch (err) {
      if (!isPlaybackTransitionCurrent(transition)) return
      if (err?.name === 'CanceledError' || err?.name === 'AbortError') return
      console.error('Failed to change quality:', err)
      setStreamError(err.message || 'Failed to change playback source')
      setSvpLoading(false)
      if (qualityId !== 'original') {
        await handleQualityChangeRef.current?.('original', transition.intent)
      } else {
        finishPlaybackTransition(transition)
      }
    }
  }, [image, mediaRef, svpStreamUrl, opticalFlowStreamUrl, transcodeStreamUrl, svpInstalled, svpConfig, opticalFlowConfig, applyNormalization, restartSVPFromPosition, stopSVP, startSVPStream, resetGain, capturePlaybackIntent, beginPlaybackTransition, isPlaybackTransitionCurrent, finishPlaybackTransition])
  handleQualityChangeRef.current = handleQualityChange

  // Check if browser can't decode the video codec (e.g. HEVC on Linux WebKitGTK/Chromium)
  // Called from Lightbox onCanPlay — if videoWidth is 0, the video track isn't decoding
  const checkCodecFallback = useCallback((video) => {
    // Only check when playing direct (no stream active, original quality)
    if (svpStreamUrl || opticalFlowStreamUrl || transcodeStreamUrl) return
    // Use ref guard to prevent multiple starts (state update is async)
    if (codecFallbackStartedRef.current) return
    if (!image || !isVideo(image.filename)) return

    if (video.videoWidth === 0 && video.videoHeight === 0 && video.readyState >= 2) {
      console.warn('[Codec] Browser cannot decode video codec, falling back to transcode')
      codecFallbackStartedRef.current = true
      setCodecFallbackActive(true)
      const startPosition = getCodecFallbackStartPosition(video.currentTime)
      const transition = beginPlaybackTransition(capturePlaybackIntent(startPosition), {
        reuseActiveIntent: false,
      })

      // Stop audio-only playback immediately so user doesn't hear disembodied audio
      video.pause()
      video.removeAttribute('src')
      video.load()

      playVideoTranscode(image.file_path, startPosition, null, transition.signal).then(result => {
        if (!isPlaybackTransitionCurrent(transition)) return
        if (result.success && result.stream_url) {
          hadTranscodeStreamRef.current = true
          activeStreamOwnerRef.current = transition
          setTranscodeStartOffset(startPosition)
          if (result.duration) setTranscodeTotalDuration(result.duration)
          if (result.source_resolution) setSourceResolution(result.source_resolution)
          setTranscodeSourceRevision(revision => nextPlaybackSourceRevision(revision))
          setTranscodeStreamUrl(result.stream_url)
        } else {
          console.error('[Codec] Transcode fallback failed:', result.error)
          setCodecFallbackActive(false)
          finishPlaybackTransition(transition)
        }
      }).catch(err => {
        if (!isPlaybackTransitionCurrent(transition)) return
        if (err?.name !== 'CanceledError' && err?.name !== 'AbortError') {
          console.error('[Codec] Transcode fallback error:', err)
        }
        setCodecFallbackActive(false)
        finishPlaybackTransition(transition)
      })
    }
  }, [image, svpStreamUrl, opticalFlowStreamUrl, transcodeStreamUrl, capturePlaybackIntent, beginPlaybackTransition, isPlaybackTransitionCurrent, finishPlaybackTransition])

  return {
    // Optical flow state
    opticalFlowConfig,
    opticalFlowLoading,
    opticalFlowError,
    opticalFlowStreamUrl,
    // SVP state
    nativeSvpPlayback,
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
    transcodeBufferedDuration,
    // Other
    sourceResolution,
    setSourceResolution,
    streamTransitioningRef,
    // Functions
    getCurrentAbsoluteTime,
    capturePlaybackIntent,
    restartSVPFromPosition,
    cancelPendingSVPRestart,
    restartTranscodeFromPosition,
    startInterpolatedStream,
    startSVPStream,
    startSVPStreamRef,
    stopSVP,
    cancelSVPLoading,
    resetStreamingState,
    handleQualityChange,
    codecFallbackActive,
    checkCodecFallback,
    setAudioOutputVolume: setOutputVolume,
    setAudioOutputMuted: setOutputMuted,
    streamError
  }
}
