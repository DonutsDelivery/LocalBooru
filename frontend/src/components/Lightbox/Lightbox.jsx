import { useEffect, useLayoutEffect, useCallback, useState, useRef, useMemo } from 'react'
import { getMediaUrl, getAssetUrl, isUsingLocalServer, getSVPConfig, updateSVPConfig, getPlaybackPosition, fetchCollections, addToCollection, createCollection, getShareNetworkInfo, uploadImage, getFileDimensions } from '../../api'
import { isMobileApp } from '../../serverManager'
import { getDesktopAPI } from '../../tauriAPI'
import { toast } from '../Toast'
import ContextMenu from '../ContextMenu'
import SVPSideMenu from '../SVPSideMenu'
import QualitySelector from '../QualitySelector'
import '../Lightbox.css'
import { isVideo, formatTime } from './utils/helpers'
import { useUIVisibility } from './hooks/useUIVisibility'
import { useZoomPan } from './hooks/useZoomPan'
import { useVideoStreaming } from './hooks/useVideoStreaming'
import { useVideoPlayback } from './hooks/useVideoPlayback'
import { useTimelinePreview } from './hooks/useTimelinePreview'
import { useWhisperSubtitles } from './hooks/useWhisperSubtitles'
import { useAutoAdvance } from './hooks/useAutoAdvance'
import { useShareStream } from './hooks/useShareStream'
import { useCastSession } from './hooks/useCastSession'
import { useVideoGestures } from './hooks/useVideoGestures'
import { useAddonStatus } from '../../hooks/useAddonStatus'
import { curationActionForSwipe } from '../../utils/lightboxGestures.js'
import { isVideoMediaElement, releaseVideoMedia } from '../../utils/lightboxMedia.js'
import { adjustmentControlState, adjustmentLocator, appendCacheBuster, commitAdjustmentSourceTransition, createAdjustmentOperationOwner, createImageSourceOwner, imageFileHash } from '../../utils/imageAdjustments.js'

// Video diagnostics overlay — press I to toggle, B for bare mode (video only)
function FPSMonitor({ videoRef, visible, onToggleBare }) {
  const statsRef = useRef(null)
  const [bare, setBare] = useState(false)
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'b' && !e.target.closest('input,textarea')) { setBare(p => { const n = !p; onToggleBare(n); return n }) } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onToggleBare])

  useEffect(() => {
    if (!visible) return
    const video = videoRef.current
    if (!video) return
    let lastPresented = 0, lastDropped = 0, rvfcCount = 0, rafCount = 0, lastTime = performance.now()
    const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype
    let rvfcId = null, rafId = null

    if (hasRVFC) {
      const onFrame = () => { rvfcCount++; rvfcId = video.requestVideoFrameCallback(onFrame) }
      rvfcId = video.requestVideoFrameCallback(onFrame)
    }

    const onRaf = () => { rafCount++; rafId = requestAnimationFrame(onRaf) }
    rafId = requestAnimationFrame(onRaf)

    const iv = setInterval(() => {
      if (!statsRef.current) return
      const now = performance.now()
      const dt = (now - lastTime) / 1000
      const lines = []
      lines.push(`rAF: ${(rafCount/dt).toFixed(1)} fps | ${bare ? 'BARE MODE' : 'press B for bare'}`)
      rafCount = 0
      if (!video.paused) {
        const q = video.getVideoPlaybackQuality?.()
        if (q) {
          const newP = q.totalVideoFrames - lastPresented
          const newD = q.droppedVideoFrames - lastDropped
          lines.push(`Presented: ${(newP/dt).toFixed(1)} fps`)
          lines.push(`Dropped: ${newD} (${q.droppedVideoFrames} total)`)
          lastPresented = q.totalVideoFrames
          lastDropped = q.droppedVideoFrames
        }
        if (hasRVFC) {
          lines.push(`RVFC: ${(rvfcCount/dt).toFixed(1)} fps`)
          rvfcCount = 0
        }
      }
      lines.push(`Size: ${video.videoWidth}x${video.videoHeight}`)
      statsRef.current.textContent = lines.join('\n')
      lastTime = now
    }, 1000)

    return () => { clearInterval(iv); cancelAnimationFrame(rafId) }
  }, [videoRef.current, bare, visible])

  if (!visible) return null

  return <pre ref={statsRef} style={{
    position:'absolute',top:10,left:10,background:'rgba(0,0,0,0.8)',
    color:'#0f0',font:'14px monospace',padding:10,borderRadius:4,
    zIndex:9999,pointerEvents:'none',whiteSpace:'pre'
  }}>Loading...</pre>
}

function Lightbox({ images, currentIndex, total, onClose, onNav, onTagClick, onImageUpdate, onSidebarHover, sidebarOpen, onDelete, curationMode = null }) {
  const [processing, setProcessing] = useState(false)
  const [isFavorited, setIsFavorited] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [copyFeedback, setCopyFeedback] = useState(null) // 'success' | 'error' | null
  const deleteDialogFocusIndex = useRef(0) // 0 = Cancel, 1 = Delete
  const cancelBtnRef = useRef(null)
  const deleteBtnRef = useRef(null)

  // Image adjustment state (Gwenview-style ranges)
  // All sliders: -100 to +100 (0 = no change)
  const [showAdjustments, setShowAdjustments] = useState(false)
  const [showMobileActions, setShowMobileActions] = useState(false)
  const [adjustments, setAdjustments] = useState({ brightness: 0, contrast: 0, gamma: 0 })
  const [applyingAdjustments, setApplyingAdjustments] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [previewIdentity, setPreviewIdentity] = useState(null)
  const [generatingPreview, setGeneratingPreview] = useState(false)
  const [imageLoadError, setImageLoadError] = useState(false)
  const [imageRetryKey, setImageRetryKey] = useState(0)

  // SVP side menu state
  const [showSVPMenu, setShowSVPMenu] = useState(false)

  // Subtitle menu state
  const [showSubtitleMenu, setShowSubtitleMenu] = useState(false)

  // Resume playback state
  const [resumePosition, setResumePosition] = useState(null) // {position, duration}
  const resumeTimerRef = useRef(null)

  // Collection picker state
  const [showCollectionPicker, setShowCollectionPicker] = useState(false)
  const [collectionsList, setCollectionsList] = useState([])
  const [collectionFeedback, setCollectionFeedback] = useState(null)
  const [newCollectionName, setNewCollectionName] = useState('')

  // Video diagnostics overlay (toggle with I key) and bare mode (B key)
  const [showDiagnostics, setShowDiagnostics] = useState(false)
  const [debugBare, setDebugBare] = useState(false)

  // Share popover state
  const [showSharePopover, setShowSharePopover] = useState(false)
  const [shareNetworkInfo, setShareNetworkInfo] = useState(null)
  const [shareCopied, setShareCopied] = useState(false)

  // Quality selector state
  const [showQualitySelector, setShowQualitySelector] = useState(false)
  const [currentQuality, setCurrentQuality] = useState(() => {
    // Load quality preference from localStorage on init
    return localStorage.getItem('video_quality_preference') || 'original'
  })

  // Context menu state
  const [contextMenu, setContextMenu] = useState(null)

  // Refs
  const mediaRef = useRef(null)
  const containerRef = useRef(null)
  const svpResumeRef = useRef(null)
  const svpSourceFpsRef = useRef(null)
  const svpFpsProbePendingRef = useRef(null)
  const svpTransitionRef = useRef({ active: false, token: 0, timer: null })
  const svpFilterActiveRef = useRef(false)
  const svpFailOpenRef = useRef(false)
  const svpToggleGenerationRef = useRef(0)
  const svpDesiredEnabledRef = useRef(null)
  const svpToggleWriteRef = useRef(Promise.resolve())
  const activeImageKeyRef = useRef(null)
  const adjustmentRequestOwnerRef = useRef(null)
  if (!adjustmentRequestOwnerRef.current) {
    adjustmentRequestOwnerRef.current = createAdjustmentOperationOwner()
  }
  const imageSourceOwnerRef = useRef(null)
  if (!imageSourceOwnerRef.current) {
    imageSourceOwnerRef.current = createImageSourceOwner()
  }
  const mountedRef = useRef(true)
  const svpPathEnabledRef = useRef(false)
  const [svpPipelineGeneration, setSvpPipelineGeneration] = useState(0)

  const image = images[currentIndex]
  const currentImageKey = image?.url ?? image?.file_path ?? image?.id
  activeImageKeyRef.current = currentImageKey
  const renderedImageUrl = isVideo(image?.original_filename)
    ? null
    : getMediaUrl(previewUrl || image?.url)
  const renderedImageSource = useMemo(
    () => imageSourceOwnerRef.current.activate(renderedImageUrl),
    [currentImageKey, renderedImageUrl, imageRetryKey]
  )

  useEffect(() => () => {
    mountedRef.current = false
    adjustmentRequestOwnerRef.current.invalidatePreview()
  }, [])
  const isVideoFile = isVideo(image?.original_filename)

  // UI visibility and fullscreen hook
  const {
    showUI,
    isFullscreen,
    resetHideTimer,
    handleMouseMove,
    handleTouchInteractionStart,
    consumeRevealTap,
    cancelRevealTap,
    handleToggleFullscreen
  } = useUIVisibility(containerRef)

  // Addon install status (hide UI when addon not installed, gate streaming)
  const { installed: whisperInstalled } = useAddonStatus('whisper-subtitles')
  const { installed: castInstalled } = useAddonStatus('cast')
  const { installed: svpInstalled } = useAddonStatus('svp')
  const casting = useCastSession(mediaRef, image)
  // Video streaming hook
  const streaming = useVideoStreaming(mediaRef, image, currentQuality, {
    svpInstalled,
    enabled: !casting.isCasting && !image?.is_local_direct_file,
  })
  const svpPathEnabled = Boolean(
    streaming.nativeSvpPlayback
    && streaming.svpConfig?.enabled
    && !casting.isCasting
    && !streaming.svpStreamUrl
    && !streaming.opticalFlowStreamUrl
    && !streaming.transcodeStreamUrl
  )
  const libraryImageId = image?.is_local_direct_file ? null : image?.id
  svpPathEnabledRef.current = svpPathEnabled

  useEffect(() => {
    if (streaming.svpConfig?.enabled !== undefined) {
      svpDesiredEnabledRef.current = Boolean(streaming.svpConfig.enabled)
    }
  }, [streaming.svpConfig?.enabled])

  useEffect(() => {
    const transition = svpTransitionRef.current
    svpToggleGenerationRef.current += 1
    svpResumeRef.current = null
    transition.active = false
    transition.token += 1
    if (transition.timer) {
      clearTimeout(transition.timer)
      transition.timer = null
    }
    return () => {
      svpToggleGenerationRef.current += 1
      svpResumeRef.current = null
      transition.active = false
      transition.token += 1
      if (transition.timer) {
        clearTimeout(transition.timer)
        transition.timer = null
      }
      getDesktopAPI()?.updateSvpManagerPlayback?.({ enabled: false }).catch(() => {})
    }
  }, [currentImageKey])

  // Video playback hook - pass streaming state
  const playback = useVideoPlayback(mediaRef, {
    svpStreamUrl: streaming.svpStreamUrl,
    svpStartOffset: streaming.svpStartOffset,
    svpBufferedDuration: streaming.svpBufferedDuration,
    svpPendingSeek: streaming.svpPendingSeek,
    setSvpPendingSeek: streaming.setSvpPendingSeek,
    transcodeStreamUrl: streaming.transcodeStreamUrl,
    transcodeStartOffset: streaming.transcodeStartOffset,
    transcodeBufferedDuration: streaming.transcodeBufferedDuration,
    svpTotalDuration: streaming.svpTotalDuration,
    transcodeTotalDuration: streaming.transcodeTotalDuration,
    opticalFlowStreamUrl: streaming.opticalFlowStreamUrl,
    streamTransitioningRef: streaming.streamTransitioningRef,
    getCurrentAbsoluteTime: streaming.getCurrentAbsoluteTime,
    restartSVPFromPosition: streaming.restartSVPFromPosition,
    cancelPendingSVPRestart: streaming.cancelPendingSVPRestart,
    restartTranscodeFromPosition: streaming.restartTranscodeFromPosition,
    setAudioOutputVolume: streaming.setAudioOutputVolume,
    setAudioOutputMuted: streaming.setAudioOutputMuted
  }, libraryImageId, image?.directory_id, image?.library_id)

  const reportSvpPlayback = useCallback((video = mediaRef.current, fps = svpSourceFpsRef.current) => {
    const desktopAPI = getDesktopAPI()
    if (!desktopAPI?.updateSvpManagerPlayback
        || !isVideoMediaElement(video)
        || mediaRef.current !== video
        || activeImageKeyRef.current !== currentImageKey
        || !svpPathEnabled
        || svpFailOpenRef.current
        || !image?.file_path
        || !fps) return
    svpSourceFpsRef.current = fps
    desktopAPI.updateSvpManagerPlayback({
      enabled: true,
      mediaKey: currentImageKey,
      path: image.file_path,
      width: video.videoWidth,
      height: video.videoHeight,
      fps,
      duration: Number.isFinite(video.duration) ? video.duration : 0,
      paused: video.paused,
    }).catch(error => console.warn('[SVPManager] playback update failed:', error))
  }, [currentImageKey, image?.file_path, svpPathEnabled])

  const measureAndReportSvpPlayback = useCallback((video) => {
    if (!svpPathEnabled || !image?.file_path) return
    if (svpSourceFpsRef.current) {
      reportSvpPlayback(video)
      return
    }
    const metadataFps = Number(image?.video_fps || image?.frame_rate || image?.fps)
    if (Number.isFinite(metadataFps) && metadataFps > 0) {
      reportSvpPlayback(video, metadataFps)
      return
    }
    const sampleDecodedFrames = () => {
      if (mediaRef.current !== video || activeImageKeyRef.current !== currentImageKey) return
      if (!video.requestVideoFrameCallback) return
      let previousMediaTime = null
      const sampleFrame = (_now, metadata) => {
        if (mediaRef.current !== video || activeImageKeyRef.current !== currentImageKey) return
        if (previousMediaTime !== null && metadata.mediaTime > previousMediaTime) {
          const measuredFps = 1 / (metadata.mediaTime - previousMediaTime)
          if (Number.isFinite(measuredFps) && measuredFps > 1 && measuredFps < 240) {
            reportSvpPlayback(video, measuredFps)
            return
          }
        }
        previousMediaTime = metadata.mediaTime
        video.requestVideoFrameCallback(sampleFrame)
      }
      video.requestVideoFrameCallback(sampleFrame)
    }
    if (!image?.is_local_direct_file && !svpFpsProbePendingRef.current) {
      const probe = { video, imageKey: currentImageKey }
      svpFpsProbePendingRef.current = probe
      getFileDimensions(image.file_path)
        .then(info => {
          if (svpFpsProbePendingRef.current !== probe) return
          const fps = Number(info?.fps)
          if (Number.isFinite(fps) && fps > 0) reportSvpPlayback(video, fps)
          else sampleDecodedFrames()
        })
        .catch(() => {
          if (svpFpsProbePendingRef.current === probe) sampleDecodedFrames()
        })
        .finally(() => {
          if (svpFpsProbePendingRef.current === probe) svpFpsProbePendingRef.current = null
        })
      return
    }
    sampleDecodedFrames()
  }, [currentImageKey, image?.file_path, image?.video_fps, image?.frame_rate, image?.fps, image?.is_local_direct_file, reportSvpPlayback, svpPathEnabled])

  useEffect(() => {
    setImageLoadError(false)
    setImageRetryKey(0)
  }, [currentImageKey, previewUrl])

  useEffect(() => {
    svpSourceFpsRef.current = null
    svpFpsProbePendingRef.current = null
    svpFailOpenRef.current = false
  }, [currentImageKey])

  useEffect(() => {
    const video = mediaRef.current
    if (svpPathEnabled && isVideoMediaElement(video) && video.readyState >= 1) measureAndReportSvpPlayback(video)
  }, [svpPathEnabled, currentImageKey, measureAndReportSvpPlayback])

  useEffect(() => {
    const desktopAPI = getDesktopAPI()
    if (!desktopAPI?.subscribeToSvpManager) return
    let unsubscribe = () => {}
    let cancelled = false
    desktopAPI.subscribeToSvpManager({
      onFilterChanged: ({ enabled, mediaKey }) => {
        if (!svpPathEnabledRef.current) return
        if (mediaKey && mediaKey !== activeImageKeyRef.current) return
        svpFilterActiveRef.current = Boolean(enabled)
        const video = mediaRef.current
        if (!isVideoMediaElement(video)) return
        const imageKey = activeImageKeyRef.current
        if (!svpResumeRef.current) {
          svpResumeRef.current = {
            currentTime: video.currentTime,
            paused: video.paused,
            imageKey,
            media: video,
          }
        }

        const transition = svpTransitionRef.current
        transition.active = true
        transition.token += 1
        const token = transition.token
        if (transition.timer) clearTimeout(transition.timer)

        releaseVideoMedia(video)

        transition.timer = setTimeout(() => {
          if (svpTransitionRef.current.token !== token
              || activeImageKeyRef.current !== imageKey
              || mediaRef.current !== video) return
          svpTransitionRef.current.timer = null
          setSvpPipelineGeneration(generation => generation + 1)
        }, 150)
      },
      onPaused: (payload) => {
        const paused = typeof payload === 'boolean' ? payload : payload?.paused
        const mediaKey = typeof payload === 'object' ? payload?.mediaKey : null
        if (!svpPathEnabledRef.current) return
        if (mediaKey && mediaKey !== activeImageKeyRef.current) return
        const video = mediaRef.current
        if (!isVideoMediaElement(video)) return
        const imageKey = activeImageKeyRef.current
        const resume = svpResumeRef.current
        if (resume && (resume.imageKey !== imageKey || resume.media !== video)) {
          svpResumeRef.current = null
        }
        if (paused) {
          if (!svpResumeRef.current) {
            svpResumeRef.current = {
              currentTime: video.currentTime,
              paused: video.paused,
              imageKey,
              media: video,
            }
          }
          video.pause()
        } else if (svpTransitionRef.current.active) {
          if (svpResumeRef.current) svpResumeRef.current.paused = false
        } else {
          video.play().catch(() => {})
        }
      },
    }).then(unlisten => {
      if (cancelled) unlisten()
      else unsubscribe = unlisten
    })
    return () => {
      cancelled = true
      unsubscribe()
      if (svpTransitionRef.current.timer) {
        clearTimeout(svpTransitionRef.current.timer)
        svpTransitionRef.current.timer = null
      }
    }
  }, [])

  useEffect(() => {
    if (svpPathEnabled) return
    svpFilterActiveRef.current = false
    svpFailOpenRef.current = false
    getDesktopAPI()?.updateSvpManagerPlayback?.({ enabled: false }).catch(() => {})
  }, [svpPathEnabled, currentImageKey])

  const failOpenNativeSvp = useCallback((video) => {
    if (!svpPathEnabled || !svpFilterActiveRef.current || svpFailOpenRef.current) return
    svpFailOpenRef.current = true
    svpFilterActiveRef.current = false
    if (isVideoMediaElement(video)) video.pause()
    getDesktopAPI()?.updateSvpManagerPlayback?.({ enabled: false }).catch(error => {
      console.warn('[SVPManager] failed to disable broken native graph:', error)
    })
  }, [svpPathEnabled])

  useEffect(() => {
    const video = mediaRef.current
    return () => {
      // A keyed remount may leave the old element alive briefly. Never dereference
      // mediaRef here: it may already point at the next video's element.
      if (mediaRef.current !== video) releaseVideoMedia(video)
    }
  }, [currentImageKey])

  // Zoom and pan hook
  const zoomPan = useZoomPan(
    mediaRef,
    containerRef,
    resetHideTimer,
    image,
    curationMode?.gestureVersion,
  )

  // Timeline preview hook (for video thumbnail preview on hover)
  const timelinePreview = useTimelinePreview(image, playback.duration)

  // Whisper subtitle hook
  const subtitles = useWhisperSubtitles(mediaRef, image)

  // Auto-advance hook
  const autoAdvance = useAutoAdvance(mediaRef, {
    onNav,
    currentIndex,
    totalImages: images.length,
    isVideoFile: isVideo(image?.original_filename),
    streamTransitioningRef: streaming.streamTransitioningRef,
    getCurrentAbsoluteTime: streaming.getCurrentAbsoluteTime,
    durationRef: playback.durationRef,
    isStreaming: Boolean(streaming.svpStreamUrl || streaming.transcodeStreamUrl || streaming.opticalFlowStreamUrl),
    isBuffering: Boolean(streaming.svpLoading || streaming.opticalFlowLoading),
    pendingSeek: streaming.svpPendingSeek,
    bufferedEnd: streaming.svpStreamUrl
      ? streaming.svpStartOffset + streaming.svpBufferedDuration
      : streaming.transcodeStreamUrl
        ? streaming.transcodeStartOffset + streaming.transcodeBufferedDuration
        : 0,
  })

  // Share stream hook (host side)
  const shareStream = useShareStream(mediaRef, {
    imageId: libraryImageId,
    directoryId: image?.directory_id,
    isVideoFile: isVideo(image?.original_filename),
  })


  // Video gesture hook (tap zones + drag-to-seek)
  const gestures = useVideoGestures(playback, resetHideTimer, cancelRevealTap)

  useEffect(() => {
    cancelRevealTap()
  }, [currentIndex, cancelRevealTap])

  // Preload next 3 images (skip videos) for smoother navigation
  useEffect(() => {
    if (!images || images.length === 0) return

    const preloadCount = 3
    const preloadedImages = []

    // Find next 3 non-video images
    let found = 0
    for (let i = currentIndex + 1; i < images.length && found < preloadCount; i++) {
      const nextImage = images[i]
      if (nextImage?.url && !isVideo(nextImage.filename)) {
        const img = new Image()
        img.src = getMediaUrl(nextImage.url)
        preloadedImages.push(img)
        found++
      }
    }

    // Find previous non-video image for back navigation
    for (let i = currentIndex - 1; i >= 0; i--) {
      const prevImage = images[i]
      if (prevImage?.url && !isVideo(prevImage.filename)) {
        const img = new Image()
        img.src = getMediaUrl(prevImage.url)
        preloadedImages.push(img)
        break // Only need 1 previous image
      }
    }

    // Cleanup: images will be garbage collected when effect re-runs
    return () => {
      preloadedImages.length = 0
    }
  }, [currentIndex, images])

  // Reset adjustments, preview, zoom, video state, and interpolation when changing images
  // Note: streaming cleanup is handled internally by useVideoStreaming (coordinated with auto-start)
  useLayoutEffect(() => {
    adjustmentRequestOwnerRef.current.invalidatePreview()
    setAdjustments({ brightness: 0, contrast: 0, gamma: 0 })
    setShowAdjustments(false)
    setPreviewUrl(null)
    setPreviewIdentity(null)
    setGeneratingPreview(false)
    zoomPan.resetZoom()
    playback.resetPlaybackState()
    subtitles.stopSubtitlesStream()
    setShowSubtitleMenu(false)
  }, [image?.library_id, image?.directory_id, image?.id])

  useLayoutEffect(() => {
    adjustmentRequestOwnerRef.current.invalidatePreview()
    setPreviewUrl(null)
    setPreviewIdentity(null)
    setGeneratingPreview(false)
  }, [adjustments.brightness, adjustments.contrast, adjustments.gamma])

  // Check for resume position when opening a video
  useEffect(() => {
    if (!libraryImageId || !image || !isVideo(image.filename)) return
    setResumePosition(null)
    clearTimeout(resumeTimerRef.current)

    getPlaybackPosition(libraryImageId, image.directory_id, image.library_id).then(data => {
      const position = data.playback_position ?? data.position
      if (position > 10 && !data.completed) {
        setResumePosition({ ...data, position })
        // Auto-dismiss after 5s
        resumeTimerRef.current = setTimeout(() => setResumePosition(null), 5000)
      }
    }).catch(() => {})

    return () => clearTimeout(resumeTimerRef.current)
  }, [libraryImageId, image?.directory_id, image?.library_id, image?.filename])

  // Auto-generate subtitles when opening a video (if enabled)
  useEffect(() => {
    if (image && isVideo(image.filename)) {
      subtitles.autoGenerate()
    }
  }, [image?.id, subtitles.whisperConfig?.auto_generate])

  // Track favorite state for current image
  useEffect(() => {
    if (image) {
      setIsFavorited(image.is_favorite || false)
    }
  }, [image?.id, image?.is_favorite])

  // Double-click handler: zoom to fill or reset (images), toggle fullscreen (videos)
  const handleDoubleClick = useCallback((e) => {
    // Don't zoom if clicking on interactive elements
    if (e.target.closest('.lightbox-toolbar, .lightbox-counter, .lightbox-confirm-overlay, .lightbox-adjustments, .lightbox-video-controls')) return

    resetHideTimer()

    // Videos: double-click toggles fullscreen (VLC behavior)
    if (isVideo(image?.original_filename)) {
      handleToggleFullscreen()
      return
    }

    // Image zoom behavior (unchanged)
    if (zoomPan.isZoomDefault()) {
      // Zoom to fill at click position
      const fillScale = zoomPan.calculateFillScale()
      if (fillScale === null) {
        // Image not ready, ignore
        return
      }
      if (fillScale <= 1.05) {
        // Image already fills or nearly fills, zoom to 2x instead
        zoomPan.setZoom({ scale: 2, x: 0, y: 0 })
      } else {
        zoomPan.setZoom({ scale: fillScale, x: 0, y: 0 })
      }
    } else {
      // Reset zoom
      zoomPan.setZoom({ scale: 1, x: 0, y: 0 })
    }
  }, [zoomPan, resetHideTimer, image?.original_filename, handleToggleFullscreen])

  // Toggle favorite
  const handleToggleFavorite = useCallback(async () => {
    if (processing || !image) return
    setProcessing(true)

    const wasActive = isFavorited
    setIsFavorited(!wasActive)

    try {
      const locator = adjustmentLocator(image)
      const { toggleFavorite } = await import('../../api')
      const result = await toggleFavorite(locator.imageId, locator.directoryId, locator.libraryId)
      setIsFavorited(result.is_favorite)
      // Update parent state so the change persists when navigating
      if (onImageUpdate) {
        onImageUpdate(locator, { is_favorite: result.is_favorite })
      }
    } catch (err) {
      console.error('Failed to toggle favorite:', err)
      setIsFavorited(wasActive)
    }

    setProcessing(false)
  }, [image, isFavorited, processing, onImageUpdate])

  // Delete image with filesystem deletion
  const handleDelete = useCallback(async () => {
    if (processing || !image) return
    setProcessing(true)

    try {
      const locator = adjustmentLocator(image)
      const { deleteImage } = await import('../../api')
      await deleteImage(locator.imageId, true, locator.directoryId, locator.libraryId)
      setShowDeleteConfirm(false)

      // Notify parent to remove the image and navigate
      if (onDelete) {
        onDelete(locator)
      }
    } catch (err) {
      console.error('Failed to delete image:', err)
      toast.error('Failed to delete image: ' + err.message)
    }

    setProcessing(false)
  }, [image, processing, onDelete])

  // Copy image to clipboard
  const handleCopyImage = useCallback(async () => {
    if (!image) return

    // Don't copy videos
    if (isVideo(image.original_filename)) {
      setCopyFeedback('error')
      setTimeout(() => setCopyFeedback(null), 1500)
      return
    }

    try {
      // Use desktop API if available (Electron or Tauri)
      const desktopAPI = getDesktopAPI()
      if (desktopAPI?.copyImageToClipboard) {
        const result = await desktopAPI.copyImageToClipboard(getMediaUrl(image.url))
        if (result.success) {
          setCopyFeedback('success')
        } else {
          throw new Error(result.error)
        }
      } else {
        // Fallback for browser - fetch and copy
        const response = await fetch(getMediaUrl(image.url))
        const blob = await response.blob()
        await navigator.clipboard.write([
          new ClipboardItem({ [blob.type]: blob })
        ])
        setCopyFeedback('success')
      }
    } catch (error) {
      console.error('Failed to copy image:', error)
      setCopyFeedback('error')
    }

    setTimeout(() => setCopyFeedback(null), 1500)
  }, [image])

  // Paste image from clipboard into the current image's directory
  const handlePasteImage = useCallback(async () => {
    try {
      const items = await navigator.clipboard.read()
      let imageBlob = null
      for (const item of items) {
        const imageType = item.types.find(t => t.startsWith('image/'))
        if (imageType) {
          imageBlob = await item.getType(imageType)
          break
        }
      }
      if (!imageBlob) {
        toast.error('No image found in clipboard')
        return
      }
      const ext = imageBlob.type.split('/')[1] || 'png'
      const file = new File([imageBlob], `pasted-image.${ext}`, { type: imageBlob.type })
      await uploadImage(file, image.directory_id)
      toast.success('Image pasted!')
      onImageUpdate?.()
    } catch (err) {
      toast.error('Failed to paste image: ' + err.message)
    }
  }, [image, onImageUpdate])

  // Show context menu on right-click
  const handleImageContextMenu = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setContextMenu({ x: e.clientX, y: e.clientY })
  }, [])

  // Handle click on video — delegated to gesture hook (tap zones)
  const handleVideoClick = useCallback((e) => {
    if (consumeRevealTap()) return
    if (!isVideo(image?.original_filename)) return
    if (casting.isCasting) {
      e.stopPropagation()
      if (casting.castStatus?.state === 'playing') {
        casting.castPause()
      } else {
        casting.castResume()
      }
      return
    }
    gestures.handleVideoClick(e)
  }, [image?.original_filename, gestures, casting, consumeRevealTap])

  // Collection picker handlers
  const handleOpenCollectionPicker = useCallback(async () => {
    if (showCollectionPicker) {
      setShowCollectionPicker(false)
      return
    }
    try {
      const data = await fetchCollections()
      setCollectionsList(data.collections || [])
    } catch (e) { /* ignore */ }
    setShowCollectionPicker(true)
  }, [showCollectionPicker])

  const handleAddToCollection = useCallback(async (collectionId) => {
    if (!image) return
    try {
      await addToCollection(collectionId, [image.id])
      setCollectionFeedback('Added!')
      setTimeout(() => setCollectionFeedback(null), 1500)
      setShowCollectionPicker(false)
    } catch (e) {
      console.error('Failed to add to collection:', e)
    }
  }, [image])

  const handleQuickCreateCollection = useCallback(async () => {
    if (!newCollectionName.trim() || !image) return
    try {
      const result = await createCollection(newCollectionName.trim())
      await addToCollection(result.id, [image.id])
      setCollectionFeedback('Created & added!')
      setTimeout(() => setCollectionFeedback(null), 1500)
      setShowCollectionPicker(false)
      setNewCollectionName('')
    } catch (e) {
      console.error('Failed to create collection:', e)
    }
  }, [newCollectionName, image])

  // Share stream handlers
  const handleToggleSharePopover = useCallback(() => {
    setShowSharePopover(prev => !prev)
  }, [])

  const handleStartSharing = useCallback(async () => {
    await shareStream.startSharing()
    try {
      const info = await getShareNetworkInfo()
      setShareNetworkInfo(info)
    } catch (e) { /* ignore */ }
  }, [shareStream])

  const handleStopSharing = useCallback(async () => {
    await shareStream.stopSharing()
  }, [shareStream])

  const handleCopyShareLink = useCallback(() => {
    if (shareStream.shareUrl) {
      navigator.clipboard.writeText(shareStream.shareUrl).then(() => {
        setShareCopied(true)
        setTimeout(() => setShareCopied(false), 2000)
      })
    }
  }, [shareStream.shareUrl])

  // Handle quality change
  const handleQualityChange = useCallback(async (qualityId) => {
    const playbackIntent = streaming.capturePlaybackIntent()
    setCurrentQuality(qualityId)
    localStorage.setItem('video_quality_preference', qualityId)
    await streaming.handleQualityChange(qualityId, playbackIntent)
  }, [streaming])

  // Toggle SVP on/off
  const handleToggleSVP = useCallback(() => {
    const confirmedEnabled = Boolean(streaming.svpConfig?.enabled)
    const desiredEnabled = svpDesiredEnabledRef.current ?? confirmedEnabled
    const newEnabled = !desiredEnabled
    const generation = ++svpToggleGenerationRef.current
    const playbackIntent = streaming.capturePlaybackIntent()
    svpDesiredEnabledRef.current = newEnabled

    const write = svpToggleWriteRef.current
      .catch(() => {})
      .then(async () => {
        // Coalesce queued clicks before issuing another persistent write.
        if (generation !== svpToggleGenerationRef.current) return
        const updatedConfig = await updateSVPConfig({ enabled: newEnabled })
        if (generation !== svpToggleGenerationRef.current) return

        svpDesiredEnabledRef.current = Boolean(updatedConfig.enabled)
        streaming.setSvpConfig(updatedConfig)
        if (newEnabled) {
          if (!streaming.nativeSvpPlayback) {
            streaming.startSVPStream(playbackIntent.position, playbackIntent, true).catch(error => {
              console.error('Failed to start SVP:', error)
            })
          }
        } else {
          setCurrentQuality('original')
          localStorage.setItem('video_quality_preference', 'original')
          streaming.handleQualityChange('original', playbackIntent).catch(error => {
            console.error('Failed to restore original playback:', error)
          })
          streaming.setSvpError(null)
          streaming.setSvpLoading(false)
        }
      })
      .catch(async err => {
        if (generation !== svpToggleGenerationRef.current) return
        console.error('Failed to toggle SVP:', err)
        toast.error('Failed to change SVP playback: ' + err.message)
        try {
          let actualConfig
          try {
            actualConfig = await updateSVPConfig({ enabled: newEnabled })
          } catch {
            actualConfig = await getSVPConfig()
          }
          if (generation !== svpToggleGenerationRef.current) return
          const actualEnabled = Boolean(actualConfig.enabled)
          svpDesiredEnabledRef.current = actualEnabled
          streaming.setSvpConfig(actualConfig)
          if (actualEnabled && !streaming.nativeSvpPlayback) {
            streaming.startSVPStream(playbackIntent.position, playbackIntent, true).catch(() => {})
          } else if (!actualEnabled) {
            streaming.handleQualityChange('original', playbackIntent).catch(() => {})
          }
        } catch (reconcileError) {
          if (generation !== svpToggleGenerationRef.current) return
          svpDesiredEnabledRef.current = newEnabled
          streaming.setSvpConfig(previous => previous ? { ...previous, enabled: newEnabled } : previous)
          console.error('Failed to reconcile SVP config:', reconcileError)
        }
      })

    svpToggleWriteRef.current = write
  }, [streaming])

  // Generate preview of adjustments
  const handleGeneratePreview = useCallback(async () => {
    if (!image || generatingPreview) return
    if (adjustments.brightness === 0 && adjustments.contrast === 0 && adjustments.gamma === 0) return

    let locator
    try {
      locator = adjustmentLocator(image)
    } catch (err) {
      toast.error(err.message)
      return
    }

    const requestedAdjustments = { ...adjustments }
    const request = adjustmentRequestOwnerRef.current.beginPreview(locator, requestedAdjustments)
    setGeneratingPreview(true)
    try {
      const { previewImageAdjustments } = await import('../../api')
      const result = await previewImageAdjustments(locator, requestedAdjustments)
      if (!adjustmentRequestOwnerRef.current.ownsPreview(request) || !mountedRef.current) return
      setPreviewIdentity(result)
      setPreviewUrl(appendCacheBuster(result.preview_url))
    } catch (err) {
      if (!adjustmentRequestOwnerRef.current.ownsPreview(request) || !mountedRef.current) return
      console.error('Failed to generate preview:', err)
      toast.error('Failed to generate preview: ' + err.message)
    } finally {
      if (adjustmentRequestOwnerRef.current.ownsPreview(request) && mountedRef.current) {
        setGeneratingPreview(false)
      }
    }
  }, [image, adjustments, generatingPreview])

  // Discard one exact preview generation and go back to CSS filter mode
  const handleDiscardPreview = useCallback(async () => {
    if (!image || !previewIdentity) return
    let locator
    try {
      locator = adjustmentLocator(image)
    } catch {
      return
    }
    const request = adjustmentRequestOwnerRef.current.beginPreview(locator, adjustments)
    try {
      const { discardImagePreview } = await import('../../api')
      await discardImagePreview(locator, previewIdentity)
    } catch (err) {
      if (adjustmentRequestOwnerRef.current.ownsPreview(request)) {
        console.error('Failed to discard preview:', err)
      }
    }
    if (adjustmentRequestOwnerRef.current.ownsPreview(request) && mountedRef.current) {
      setPreviewUrl(null)
      setPreviewIdentity(null)
    }
  }, [image, adjustments, previewIdentity])

  // Apply adjustments to the captured file. Backend completion always updates that locator.
  const handleApplyAdjustments = useCallback(async () => {
    if (!image || adjustmentRequestOwnerRef.current.isApplyInFlight()) return
    if (adjustments.brightness === 0 && adjustments.contrast === 0 && adjustments.gamma === 0) return

    let locator
    try {
      locator = adjustmentLocator(image)
    } catch (err) {
      toast.error(err.message)
      return
    }
    const expectedFileHash = imageFileHash(image)
    if (!expectedFileHash) {
      toast.error('Image metadata is missing its file hash')
      return
    }

    const requestedAdjustments = { ...adjustments }
    const capturedPreview = previewIdentity
    const operation = adjustmentRequestOwnerRef.current.beginApply(
      locator,
      requestedAdjustments,
      expectedFileHash
    )
    if (!operation) return
    const uiRequest = adjustmentRequestOwnerRef.current.beginPreview(locator, requestedAdjustments)
    setApplyingAdjustments(true)
    try {
      const { applyImageAdjustments, discardImagePreview } = await import('../../api')
      const result = await applyImageAdjustments(locator, requestedAdjustments, expectedFileHash)
      const updates = {
        url: result.url || appendCacheBuster(image.url),
        thumbnail_url: result.thumbnail_url || appendCacheBuster(image.thumbnail_url)
      }
      for (const key of ['file_hash', 'filename', 'file_size', 'width', 'height', 'file_modified_at']) {
        if (result[key] != null) updates[key] = result[key]
      }
      const publishCommittedSource = () => onImageUpdate?.(locator, updates)
      const cleanupPreview = capturedPreview
        ? () => discardImagePreview(locator, capturedPreview)
        : null

      if (adjustmentRequestOwnerRef.current.ownsPreview(uiRequest) && mountedRef.current) {
        commitAdjustmentSourceTransition({
          operationOwner: adjustmentRequestOwnerRef.current,
          sourceOwner: imageSourceOwnerRef.current,
          committedSource: getMediaUrl(updates.url),
          clearPreview: () => {
            setPreviewUrl(null)
            setPreviewIdentity(null)
            setAdjustments({ brightness: 0, contrast: 0, gamma: 0 })
            setShowAdjustments(false)
          },
          publishCommittedSource,
          cleanupPreview,
        })
      } else {
        publishCommittedSource()
        if (cleanupPreview) Promise.resolve().then(cleanupPreview).catch(() => {})
      }
    } catch (err) {
      if (adjustmentRequestOwnerRef.current.ownsPreview(uiRequest) && mountedRef.current) {
        console.error('Failed to apply adjustments:', err)
        toast.error('Failed to apply adjustments: ' + err.message)
      }
    } finally {
      adjustmentRequestOwnerRef.current.finishApply(operation)
      if (mountedRef.current) setApplyingAdjustments(false)
    }
  }, [image, adjustments, previewIdentity, onImageUpdate])

  // Gamma exponent for SVG filter (computed outside getFilterStyle so JSX can use it)
  const adjustmentControls = adjustmentControlState({
    applying: applyingAdjustments,
    generatingPreview,
    adjustments,
  })

  const gammaExponent = adjustments.gamma !== 0
    ? Math.pow(3.0, -adjustments.gamma / 100)
    : 1

  // Generate CSS filter string for preview
  // Uses CSS brightness/contrast + SVG filter for gamma to match backend
  const getFilterStyle = () => {
    if (adjustments.brightness === 0 && adjustments.contrast === 0 && adjustments.gamma === 0) {
      return {}
    }

    // Brightness: multiplicative (CSS brightness is a multiplier)
    // slider -100 to +100 maps to 0.0 to 2.0 multiplier
    // Extended range -200 to +200 maps to -1.0 to 3.0, clamped to 0
    const cssBrightness = Math.max(0, 1 + (adjustments.brightness / 100))

    // Contrast: CSS contrast multiplier
    // slider -100 to +100 maps to 0.0 to 2.0
    const cssContrast = (adjustments.contrast + 100) / 100

    // Build filter string: brightness and contrast via CSS, gamma via SVG
    const filters = []

    if (adjustments.brightness !== 0 || adjustments.contrast !== 0) {
      filters.push(`brightness(${cssBrightness}) contrast(${cssContrast})`)
    }

    if (adjustments.gamma !== 0) {
      filters.push(`url(#lb-gamma-${adjustments.gamma})`)
    }

    return {
      filter: filters.join(' ')
    }
  }

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

      if (curationMode) {
        if (e.repeat || curationMode.busy) return
        switch (e.key) {
          case 'Escape':
            e.preventDefault()
            onClose()
            return
          case 'ArrowLeft':
          case 'd':
          case 'D':
            e.preventDefault()
            curationMode.onDiscard()
            return
          case 'ArrowRight':
          case 'k':
          case 'K':
            e.preventDefault()
            curationMode.onKeep()
            return
          case 'Backspace':
          case 'u':
          case 'U':
            e.preventDefault()
            if (curationMode.lastAction) curationMode.onUndo()
            return
          case 'Delete':
          case 'f':
          case 'F':
            e.preventDefault()
            return
        }
      }

      // Handle delete dialog keyboard navigation
      if (showDeleteConfirm) {
        switch (e.key) {
          case 'Escape':
            e.preventDefault()
            setShowDeleteConfirm(false)
            break
          case 'Enter':
            e.preventDefault()
            if (deleteDialogFocusIndex.current === 0) {
              // Cancel is focused
              setShowDeleteConfirm(false)
            } else {
              // Delete is focused
              handleDelete()
            }
            break
          case 'ArrowLeft':
            e.preventDefault()
            deleteDialogFocusIndex.current = 0
            cancelBtnRef.current?.focus()
            break
          case 'ArrowRight':
            e.preventDefault()
            deleteDialogFocusIndex.current = 1
            deleteBtnRef.current?.focus()
            break
        }
        return // Don't process other keys when dialog is open
      }

      // Ctrl+C to copy image
      if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        e.preventDefault()
        handleCopyImage()
        return
      }

      const isVideoFile = isVideo(image?.original_filename)

      // VLC-like video controls — seeking requires Ctrl/Shift modifiers.
      // Bare arrow keys always navigate the gallery.
      if (isVideoFile && mediaRef.current) {
        switch (e.key) {
          case ' ':
            e.preventDefault()
            if (casting.isCasting) {
              casting.castStatus?.state === 'playing' ? casting.castPause() : casting.castResume()
            } else {
              playback.toggleVideoPlay()
            }
            return
          case 'ArrowLeft':
            if (e.ctrlKey || e.metaKey) {
              e.preventDefault()
              casting.isCasting
                ? casting.castSeekRelative(-30)
                : playback.seekVideo(-30)
              return
            }
            if (e.shiftKey) {
              e.preventDefault()
              casting.isCasting
                ? casting.castSeekRelative(-1)
                : playback.seekVideo(-1)
              return
            }
            // No modifier: gallery navigation (falls through below)
            break
          case 'ArrowRight':
            if (e.ctrlKey || e.metaKey) {
              e.preventDefault()
              casting.isCasting
                ? casting.castSeekRelative(30)
                : playback.seekVideo(30)
              return
            }
            if (e.shiftKey) {
              e.preventDefault()
              casting.isCasting
                ? casting.castSeekRelative(1)
                : playback.seekVideo(1)
              return
            }
            // No modifier: gallery navigation (falls through below)
            break
          case 'ArrowUp':
            e.preventDefault()
            casting.isCasting ? casting.castVolumeRelative(0.05) : playback.adjustVolume(0.05)
            return
          case 'ArrowDown':
            e.preventDefault()
            casting.isCasting ? casting.castVolumeRelative(-0.05) : playback.adjustVolume(-0.05)
            return
          case 'm':
          case 'M':
            e.preventDefault()
            if (!casting.isCasting) playback.toggleMute()
            return
          case 'f':
          case 'F':
            e.preventDefault()
            handleToggleFullscreen()
            return
          case '+':
          case '=':
          case ']':
            e.preventDefault()
            if (casting.isCasting) return
            playback.increaseSpeed() // Speed +0.25x
            return
          case '-':
          case '[':
            e.preventDefault()
            if (casting.isCasting) return
            playback.decreaseSpeed() // Speed -0.25x
            return
          case 'Backspace':
            e.preventDefault()
            if (casting.isCasting) return
            playback.resetSpeed() // Reset to 1.0x
            return
          case 'e':
          case 'E':
            e.preventDefault()
            if (casting.isCasting) return
            playback.frameAdvance() // Frame advance (when paused)
            return
          case 'c':
          case 'C':
            e.preventDefault()
            subtitles.toggleSubtitles()
            return
          case 'i':
          case 'I':
            e.preventDefault()
            setShowDiagnostics(p => !p)
            return
        }
      }

      switch (e.key) {
        case 'Escape':
          onClose()
          break
        case 'ArrowLeft':
        case 'a':
          onNav(-1)
          break
        case 'ArrowRight':
        case 'd':
          onNav(1)
          break
        case 'f':
          // Only toggle favorite for images (videos use F for fullscreen)
          if (!isVideoFile) {
            handleToggleFavorite()
          }
          break
        case 'Delete':
          setShowDeleteConfirm(true)
          deleteDialogFocusIndex.current = 0 // Default focus to Cancel for safety
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    document.body.style.overflow = 'hidden'

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = ''
    }
  }, [onNav, onClose, handleToggleFavorite, handleCopyImage, handleDelete, showDeleteConfirm, playback, image?.original_filename, handleToggleFullscreen, subtitles, curationMode])

  // Auto-focus Cancel button when delete dialog opens
  useEffect(() => {
    if (showDeleteConfirm) {
      deleteDialogFocusIndex.current = 0
      // Focus after a short delay to ensure DOM is ready
      setTimeout(() => cancelBtnRef.current?.focus(), 10)
    }
  }, [showDeleteConfirm])

  // Handle click navigation - left side = prev, right side = next
  const handleNavClick = (e) => {
    if (curationMode) return
    if (consumeRevealTap()) return
    // Don't navigate if we handled this as a touch gesture or if touch moved
    if (zoomPan.touchMoved.current || zoomPan.touchHandled.current) {
      zoomPan.touchMoved.current = false
      zoomPan.touchHandled.current = false
      return
    }

    // Don't navigate on videos (navigation uses buttons only)
    if (isVideo(image?.original_filename)) return

    // Don't navigate if clicking on interactive elements (but allow video area for navigation)
    if (e.target.closest('.lightbox-toolbar, .lightbox-counter, .lightbox-confirm-overlay, .lightbox-video-controls, .lightbox-adjust-container')) return

    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const width = rect.width

    // When zoomed in, only allow tap-to-navigate on touch devices. Desktop users
    // pan with the mouse, and a click while zoomed should not jump to next/prev.
    // On phones, a real pan sets touchMoved.current (already handled above), so
    // reaching this point with zoom > 1 means it was a tap, not a pan.
    if (zoomPan.zoom.scale > 1 && width > 768) return

    // Left 40% = previous, right 40% = next, middle 20% = do nothing
    // On desktop, exclude the sidebar hover zone (~100px). On mobile, the zone is smaller (40px)
    // and sidebar is controlled by swipe, so we can use a smaller buffer
    const sidebarBuffer = width > 768 ? 100 : 40
    if (clickX < width * 0.4 && clickX > sidebarBuffer) {
      onNav(-1)
    } else if (clickX > width * 0.6) {
      onNav(1)
    }
  }

  // Handle touch end while preserving ordinary Lightbox swipes during curation
  const handleTouchEndWithSidebar = useCallback((e) => {
    const onCurationSwipe = curationMode ? curationActionForSwipe : null
    zoomPan.handleTouchEnd(e, onSidebarHover, sidebarOpen, onCurationSwipe)
  }, [zoomPan, onSidebarHover, sidebarOpen, curationMode])

  // Handle loaded metadata with source resolution setter
  const handleLoadedMetadataWithResolution = useCallback(() => {
    playback.handleLoadedMetadata(streaming.setSourceResolution)
  }, [playback.handleLoadedMetadata, streaming.setSourceResolution])

  // Handle video canplay event - ensure video plays even if autoPlay is blocked
  // Also reset hide timer to ensure auto-hide works on mobile/Capacitor
  // Also check if browser can decode the video codec (fallback to transcode if not)
  const handleVideoCanPlay = useCallback((e) => {
    const streamActive = Boolean(streaming.svpStreamUrl || streaming.transcodeStreamUrl || streaming.opticalFlowStreamUrl)
    if (!casting.isCasting && !streamActive && playback.isPlaying) {
      e.target.play().catch(() => {})
    }
    resetHideTimer()
    streaming.checkCodecFallback(e.target)
  }, [
    casting.isCasting,
    playback.isPlaying,
    resetHideTimer,
    streaming.svpStreamUrl,
    streaming.transcodeStreamUrl,
    streaming.opticalFlowStreamUrl,
    streaming.checkCodecFallback,
  ])

  // Handle video context menu
  const handleVideoContextMenu = useCallback((e) => {
    e.preventDefault()
    setContextMenu({ x: e.clientX, y: e.clientY })
  }, [])

  // Determine if we should play the video directly (no streaming)
  const shouldPlayDirect = useMemo(() => {
    if (image?.is_local_direct_file) return true
    return streaming.svpConfigLoaded
      && !streaming.svpStreamUrl
      && !streaming.opticalFlowStreamUrl
      && !streaming.transcodeStreamUrl
      && !streaming.svpLoading
      && !streaming.codecFallbackActive
      && (currentQuality === 'original' || (streaming.nativeSvpPlayback && streaming.svpConfig?.enabled))
      && (!streaming.svpConfig?.enabled || streaming.nativeSvpPlayback || streaming.svpError)
      && (!streaming.opticalFlowConfig?.enabled || streaming.opticalFlowError)
  }, [
    streaming.svpConfigLoaded, streaming.svpStreamUrl, streaming.opticalFlowStreamUrl,
    streaming.transcodeStreamUrl, streaming.svpLoading, streaming.codecFallbackActive,
    currentQuality, streaming.svpConfig?.enabled, streaming.nativeSvpPlayback, streaming.svpError,
    streaming.opticalFlowConfig?.enabled, streaming.opticalFlowError,
    image?.is_local_direct_file
  ])

  // On Tauri mobile with local server, use asset protocol to serve videos directly from disk.
  // WRY's shouldInterceptRequest buffers entire HTTP responses before returning them to the WebView,
  // which causes video elements with http:// src to load endlessly for large files.
  // The asset protocol serves from the filesystem directly, bypassing this bottleneck.
  const directVideoSrc = useMemo(() => {
    if (!shouldPlayDirect || !image?.url) return undefined
    if (image?.is_local_direct_file) {
      return getMediaUrl(image.url)
    }
    if (isMobileApp() && isUsingLocalServer() && image?.file_path) {
      const assetUrl = getAssetUrl(image.file_path)
      if (assetUrl) return assetUrl
    }
    return getMediaUrl(image.url)
  }, [shouldPlayDirect, image?.url, image?.file_path, image?.is_local_direct_file])

  const directFileStartedRef = useRef(false)
  const directFileLastTimeRef = useRef(0)
  useEffect(() => {
    directFileStartedRef.current = false
    directFileLastTimeRef.current = 0
  }, [image?.id])

  const reportDirectFileStage = useCallback((stage, video = mediaRef.current) => {
    if (!image?.is_local_direct_file) return
    const error = video?.error
    let details = ''
    if (video) {
      details = ` ready=${video.readyState} network=${video.networkState} time=${video.currentTime.toFixed(3)} duration=${Number.isFinite(video.duration) ? video.duration.toFixed(3) : video.duration} paused=${video.paused} loop=${video.loop}`
      if (error) details += ` error=${error.code}:${error.message}`
    }
    import('@tauri-apps/api/core')
      .then(({ invoke }) => invoke('report_direct_file_stage', { stage: `${stage}${details}` }))
      .catch(() => {})
  }, [image?.is_local_direct_file])

  if (!image) return null

  const fileStatus = image.file_status || 'available'
  const isUnavailable = fileStatus !== 'available'

  return (
    <div
      className={`lightbox ${!showUI ? 'ui-hidden' : ''} ${zoomPan.zoom.scale > 1 ? 'zoomed' : ''} ${isFullscreen ? 'fullscreen' : ''} ${isVideoFile ? 'lightbox-video' : ''} ${casting.isCasting ? 'casting-active' : ''} ${curationMode ? 'curation-active' : ''}`}
      onClick={handleNavClick}
      onDoubleClick={handleDoubleClick}
      onMouseMove={(e) => { handleMouseMove(); zoomPan.handleMouseMoveDrag(e); }}
      onMouseDown={zoomPan.handleMouseDown}
      onMouseUp={zoomPan.handleMouseUp}
      onMouseLeave={zoomPan.handleMouseUp}
      onTouchStart={(event) => {
        handleTouchInteractionStart()
        zoomPan.handleTouchStart(event)
      }}
      onTouchMove={(e) => { zoomPan.handleTouchMove(e); zoomPan.handleTouchMoveZoom(e); }}
      onTouchEnd={(e) => {
        handleTouchEndWithSidebar(e)
        if (zoomPan.touchMoved.current || zoomPan.touchHandled.current) cancelRevealTap()
        zoomPan.handleTouchEndZoom()
      }}
      onTouchCancel={cancelRevealTap}
      ref={containerRef}
    >
      {/* Hidden SVG filter for gamma correction */}
      <svg style={{ position: 'absolute', width: 0, height: 0, pointerEvents: 'none' }}>
        <filter id={`lb-gamma-${adjustments.gamma}`}>
          <feComponentTransfer>
            <feFuncR type="gamma" amplitude="1" exponent={gammaExponent} offset="0" />
            <feFuncG type="gamma" amplitude="1" exponent={gammaExponent} offset="0" />
            <feFuncB type="gamma" amplitude="1" exponent={gammaExponent} offset="0" />
          </feComponentTransfer>
        </filter>
      </svg>
      {/* Top toolbar */}
      {!debugBare && <div className={`lightbox-toolbar ${showMobileActions ? 'mobile-actions-open' : ''}`}>
        <button
          className="lightbox-btn lightbox-menu"
          onClick={() => onSidebarHover && onSidebarHover(!sidebarOpen)}
          title="Media information"
          aria-label="Media information"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="3" y1="6" x2="21" y2="6"/>
            <line x1="3" y1="12" x2="21" y2="12"/>
            <line x1="3" y1="18" x2="21" y2="18"/>
          </svg>
        </button>
        <button
          className={`lightbox-btn lightbox-favorite ${isFavorited ? 'active' : ''}`}
          onClick={handleToggleFavorite}
          disabled={processing}
          title={isFavorited ? 'Remove from favorites (F)' : 'Add to favorites (F)'}
          aria-label={isFavorited ? 'Remove from favorites' : 'Add to favorites'}
        >
          <svg viewBox="0 0 24 24" fill={isFavorited ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
        </button>
        <div className="lightbox-collection-container">
          <button
            className={`lightbox-btn lightbox-collection ${showCollectionPicker ? 'active' : ''}`}
            onClick={handleOpenCollectionPicker}
            title="Add to collection"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6z"/>
              <rect x="6" y="2" width="16" height="16" rx="2"/>
              <path d="M14 6v8M10 10h8"/>
            </svg>
          </button>
          {showCollectionPicker && (
            <div className="collection-picker" onClick={(e) => e.stopPropagation()}>
              <div className="collection-picker-header">Add to Collection</div>
              {collectionsList.length > 0 && (
                <div className="collection-picker-list">
                  {collectionsList.map(c => (
                    <button key={c.id} className="collection-picker-item" onClick={() => handleAddToCollection(c.id)}>
                      {c.name} <span className="collection-picker-count">({c.item_count})</span>
                    </button>
                  ))}
                </div>
              )}
              <div className="collection-picker-create">
                <input
                  type="text"
                  placeholder="New collection..."
                  value={newCollectionName}
                  onChange={(e) => setNewCollectionName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleQuickCreateCollection() }}
                />
                <button onClick={handleQuickCreateCollection} disabled={!newCollectionName.trim()}>Create</button>
              </div>
            </div>
          )}
          {collectionFeedback && <div className="collection-feedback">{collectionFeedback}</div>}
        </div>
        {castInstalled && isVideo(image?.original_filename) && casting.castConfig?.enabled && (
          <div className="lightbox-cast-container lightbox-secondary-action">
            <button
              className={`lightbox-btn lightbox-cast ${casting.isCasting ? 'active' : ''}`}
              onClick={casting.toggleDevicePicker}
              title={casting.isCasting ? 'Casting active' : 'Cast to device'}
            >
              <svg viewBox="0 0 24 24" fill={casting.isCasting ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2">
                <path d="M2 16.1A5 5 0 0 1 5.9 20M2 12.05A9 9 0 0 1 9.95 20M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/>
                <line x1="2" y1="20" x2="2.01" y2="20"/>
              </svg>
            </button>
            {casting.showDevicePicker && !casting.isCasting && (
              <>
              <div className="cast-picker-backdrop" onClick={(e) => { e.stopPropagation(); casting.toggleDevicePicker() }} />
              <div className="cast-device-picker" onClick={(e) => e.stopPropagation()}>
                <div className="cast-picker-header">
                  <div className="cast-picker-title">
                    <span>Cast to</span>
                    <span className="cast-picker-subtitle">
                      {casting.devicesLoading ? 'Scanning for devices…' : `${casting.devices.length} device${casting.devices.length === 1 ? '' : 's'} found`}
                    </span>
                  </div>
                  <button className="cast-picker-refresh" onClick={casting.refreshDevices} disabled={casting.devicesLoading}>
                    {casting.devicesLoading ? (
                      <div className="cast-picker-spinner" />
                    ) : (
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                        <path d="M3 3v5h5"/>
                      </svg>
                    )}
                  </button>
                </div>
                {casting.devices.length > 0 ? (
                  <div className="cast-picker-list">
                    {casting.devices.map(device => (
                      <button
                        key={device.id}
                        className="cast-picker-device"
                        onClick={() => casting.startCasting(device.id)}
                      >
                        <span className={`cast-device-icon ${device.type}`}>
                          {device.type === 'chromecast' ? (
                            <svg viewBox="0 0 24 24" fill="currentColor">
                              <path d="M1 18v3h3c0-1.66-1.34-3-3-3zm0-4v2c2.76 0 5 2.24 5 5h2c0-3.87-3.13-7-7-7zm18-7H5v1.63c3.96 1.28 7.09 4.41 8.37 8.37H19V7zM1 10v2c4.97 0 9 4.03 9 9h2c0-6.08-4.93-11-11-11zm20-7H3c-1.1 0-2 .9-2 2v3h2V5h18v14h-7v2h7c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/>
                            </svg>
                          ) : (
                            <svg viewBox="0 0 24 24" fill="currentColor">
                              <path d="M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14z"/>
                            </svg>
                          )}
                        </span>
                        <div className="cast-device-info">
                          <span className="cast-device-name">{device.name}</span>
                          <span className="cast-device-model">{device.model || device.type}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="cast-picker-empty">
                    {casting.devicesLoading ? (
                      <>
                        <div className="cast-picker-spinner large" />
                        <span>Looking for devices on your network…</span>
                      </>
                    ) : (
                      <>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                          <path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/>
                          <line x1="2" y1="20" x2="2.01" y2="20"/>
                        </svg>
                        <span>No devices found</span>
                        <span className="cast-picker-empty-hint">Make sure your TV is on and connected to the same network</span>
                      </>
                    )}
                  </div>
                )}
              </div>
              </>
            )}
          </div>
        )}
        {isVideo(image?.original_filename) && (
          <div className="lightbox-share-container lightbox-secondary-action">
            <button
              className={`lightbox-btn lightbox-share ${shareStream.isSharing ? 'active' : ''}`}
              onClick={handleToggleSharePopover}
              title={shareStream.isSharing ? 'Sharing active' : 'Share stream'}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/>
                <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>
              </svg>
            </button>
            {showSharePopover && (
              <div className="share-popover" onClick={(e) => e.stopPropagation()}>
                <div className="share-popover-header">Share Stream</div>
                {shareStream.isSharing ? (
                  <>
                    <div className="share-popover-link">
                      <input
                        type="text"
                        readOnly
                        value={shareStream.shareUrl || ''}
                        onClick={(e) => e.target.select()}
                      />
                      <button onClick={handleCopyShareLink}>
                        {shareCopied ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                    {shareNetworkInfo && !shareNetworkInfo.tailscale_installed && (
                      <div className="share-popover-notice">
                        <span className="share-notice-icon">&#9888;</span>
                        <span>LAN only — not reachable from the internet.</span>
                      </div>
                    )}
                    {shareNetworkInfo && shareNetworkInfo.tailscale_installed && shareNetworkInfo.tailscale_url && (
                      <div className="share-popover-notice share-notice-ok">
                        <span className="share-notice-icon">&#10003;</span>
                        <span>{shareNetworkInfo.tailscale_https ? 'Shareable over internet (HTTPS)' : 'Shareable over internet'}</span>
                      </div>
                    )}
                    {shareNetworkInfo && shareNetworkInfo.tailscale_installed && shareNetworkInfo.tailscale_needs_operator && (
                      <div className="share-popover-tailscale">
                        <span>Enable HTTPS share links:</span>
                        <code className="share-operator-cmd">sudo tailscale set --operator=$USER</code>
                        <span className="share-tailscale-hint">Run once, then restart LocalBooru</span>
                      </div>
                    )}
                    {shareNetworkInfo && !shareNetworkInfo.tailscale_installed && (
                      <div className="share-popover-tailscale">
                        <span>Want to share over the internet?</span>
                        <a
                          href={`https://tailscale.com/download/${shareNetworkInfo.os}`}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Set up Tailscale &rarr;
                        </a>
                        <span className="share-tailscale-hint">Free, takes ~2 minutes</span>
                      </div>
                    )}
                    <button className="share-popover-stop" onClick={handleStopSharing}>
                      Stop Sharing
                    </button>
                  </>
                ) : (
                  <div className="share-popover-start">
                    <p>Generate a link to watch this video in sync with others on your network.</p>
                    <button className="share-popover-start-btn" onClick={handleStartSharing}>
                      Start Sharing
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        <button
          className="lightbox-btn lightbox-delete lightbox-secondary-action"
          onClick={() => setShowDeleteConfirm(true)}
          disabled={processing}
          title="Delete image"
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            <line x1="10" y1="11" x2="10" y2="17"/>
            <line x1="14" y1="11" x2="14" y2="17"/>
          </svg>
        </button>
        {!isVideoFile && (
          <div className="lightbox-adjust-container lightbox-secondary-action">
            <button
              className={`lightbox-btn lightbox-adjust ${showAdjustments ? 'active' : ''}`}
              onClick={() => setShowAdjustments(!showAdjustments)}
              disabled={processing || isUnavailable}
              title="Adjust image"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="3"/>
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
              </svg>
            </button>
            {/* Image adjustment dropdown */}
            {showAdjustments && (
              <div className="lightbox-adjustments" onClick={e => e.stopPropagation()}>
                <div className="adjustment-slider">
                  <label>
                    <span>Brightness</span>
                    <span className="adjustment-value">{adjustments.brightness > 0 ? '+' : ''}{adjustments.brightness}</span>
                  </label>
                  <input
                    type="range"
                    min="-200"
                    max="200"
                    step="1"
                    value={adjustments.brightness}
                    disabled={adjustmentControls.inputsDisabled}
                    onChange={e => {
                      adjustmentRequestOwnerRef.current.invalidatePreview()
                      setAdjustments(prev => ({ ...prev, brightness: parseInt(e.target.value) }))
                      if (previewUrl) setPreviewUrl(null) // Clear stale preview
                    }}
                  />
                </div>
                <div className="adjustment-slider">
                  <label>
                    <span>Contrast</span>
                    <span className="adjustment-value">{adjustments.contrast > 0 ? '+' : ''}{adjustments.contrast}</span>
                  </label>
                  <input
                    type="range"
                    min="-100"
                    max="100"
                    step="1"
                    value={adjustments.contrast}
                    disabled={adjustmentControls.inputsDisabled}
                    onChange={e => {
                      adjustmentRequestOwnerRef.current.invalidatePreview()
                      setAdjustments(prev => ({ ...prev, contrast: parseInt(e.target.value) }))
                      if (previewUrl) setPreviewUrl(null) // Clear stale preview
                    }}
                  />
                </div>
                <div className="adjustment-slider">
                  <label>
                    <span>Gamma</span>
                    <span className="adjustment-value">{adjustments.gamma > 0 ? '+' : ''}{adjustments.gamma}</span>
                  </label>
                  <input
                    type="range"
                    min="-100"
                    max="100"
                    step="1"
                    value={adjustments.gamma}
                    disabled={adjustmentControls.inputsDisabled}
                    onChange={e => {
                      adjustmentRequestOwnerRef.current.invalidatePreview()
                      setAdjustments(prev => ({ ...prev, gamma: parseInt(e.target.value) }))
                      if (previewUrl) setPreviewUrl(null) // Clear stale preview
                    }}
                  />
                </div>
                <div className="adjustment-actions">
                  <button
                    className="adjustment-reset"
                    disabled={adjustmentControls.resetDisabled}
                    onClick={() => {
                      setAdjustments({ brightness: 0, contrast: 0, gamma: 0 })
                      if (previewUrl) handleDiscardPreview()
                    }}
                  >
                    Reset
                  </button>
                  <button
                    className="adjustment-preview"
                    onClick={previewUrl ? handleDiscardPreview : handleGeneratePreview}
                    disabled={adjustmentControls.previewDisabled}
                  >
                    {generatingPreview ? 'Rendering...' : previewUrl ? 'Clear Exact Preview' : 'Render Exact Preview'}
                  </button>
                  <button
                    className="adjustment-apply"
                    onClick={handleApplyAdjustments}
                    disabled={adjustmentControls.applyDisabled}
                  >
                    {applyingAdjustments ? 'Saving...' : 'Apply'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
        {isVideoFile && svpInstalled && (
          <button
            className="lightbox-btn lightbox-svp lightbox-secondary-action"
            onClick={() => setShowSVPMenu(true)}
            title="SVP Settings"
          >
            <svg viewBox="0 0 24 24" fill="currentColor">
              <circle cx="5" cy="12" r="2"/>
              <circle cx="12" cy="12" r="2"/>
              <circle cx="19" cy="12" r="2"/>
            </svg>
          </button>
        )}
        {isVideoFile && !casting.isCasting && (
          <button
            className={`lightbox-btn lightbox-mobile-quality lightbox-secondary-action ${currentQuality !== 'original' ? 'active' : ''}`}
            onClick={() => setShowQualitySelector(value => !value)}
            title={`Quality: ${currentQuality === 'original' ? 'Original' : currentQuality}`}
            aria-label={`Video quality: ${currentQuality === 'original' ? 'Original' : currentQuality}`}
          >
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zm-5.04-6.71l-2.75 3.54h2.79v2.71h2V13.83h2.79l-2.75-3.54zM7 9h2v2H7z"/>
            </svg>
          </button>
        )}
        {isVideoFile ? (
          /* Videos: cycle display mode button */
          <button
            className={`lightbox-btn lightbox-display-mode lightbox-secondary-action ${playback.videoDisplayMode !== 'fit' ? 'active' : ''}`}
            onClick={playback.cycleDisplayMode}
            title={`Display: ${playback.videoDisplayMode} (click to cycle)`}
          >
            {playback.videoDisplayMode === 'fit' ? (
              /* Fit icon - arrows pointing inward */
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <path d="M9 9l-3-3M9 9H6M9 9V6"/>
                <path d="M15 9l3-3M15 9h3M15 9V6"/>
                <path d="M9 15l-3 3M9 15H6M9 15v3"/>
                <path d="M15 15l3 3M15 15h3M15 15v3"/>
              </svg>
            ) : playback.videoDisplayMode === 'original' ? (
              /* Original/1:1 icon */
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <text x="12" y="16" textAnchor="middle" fontSize="10" fill="currentColor" stroke="none">1:1</text>
              </svg>
            ) : (
              /* Fill/crop icon */
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <path d="M7 3v18M17 3v18" strokeDasharray="3 3"/>
              </svg>
            )}
          </button>
        ) : (
          /* Images: fullscreen toggle button */
          <button
            className={`lightbox-btn lightbox-fullscreen lightbox-secondary-action ${isFullscreen ? 'active' : ''}`}
            onClick={handleToggleFullscreen}
            title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/>
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3"/>
              </svg>
            )}
          </button>
        )}
        <button
          className={`lightbox-btn lightbox-more ${showMobileActions ? 'active' : ''}`}
          onClick={() => setShowMobileActions(value => !value)}
          aria-label="More media actions"
          aria-expanded={showMobileActions}
          title="More actions"
        >
          <svg viewBox="0 0 24 24" fill="currentColor">
            <circle cx="5" cy="12" r="2"/>
            <circle cx="12" cy="12" r="2"/>
            <circle cx="19" cy="12" r="2"/>
          </svg>
        </button>
        <button className="lightbox-btn lightbox-close" onClick={onClose} title="Close (Esc)" aria-label="Close lightbox">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      </div>}

      {/* Hover zone to trigger main sidebar */}
      <div
        className="lightbox-sidebar-trigger"
        onMouseEnter={() => onSidebarHover && onSidebarHover(true)}
      />

      <div className="lightbox-content">
        {isUnavailable ? (
          <div className={`lightbox-unavailable ${fileStatus}`}>
            {fileStatus === 'drive_offline' ? (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                  <path d="M3 3v5h5"/>
                  <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
                  <path d="M16 16h5v5"/>
                </svg>
                <h3>Drive Offline</h3>
                <p>The storage device containing this file is not connected.</p>
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M15 9l-6 6M9 9l6 6"/>
                </svg>
                <h3>File Missing</h3>
                <p>This file has been deleted or moved.</p>
              </>
            )}
          </div>
        ) : isVideoFile ? (
          <div
            className="lightbox-video-container"
            onTouchStart={casting.isCasting || curationMode ? undefined : gestures.handleTouchStart}
            onTouchMove={casting.isCasting || curationMode ? undefined : gestures.handleTouchMove}
            onTouchEnd={casting.isCasting || curationMode ? undefined : gestures.handleTouchEnd}
            onTouchCancel={casting.isCasting || curationMode ? cancelRevealTap : gestures.handleTouchCancel}
          >
            <video
              key={`${currentImageKey}-${svpPipelineGeneration}`}
              ref={mediaRef}
              src={directVideoSrc}
              preload="auto"
              autoPlay
              playsInline
              loop={false}
              className={`lightbox-media video-display-${playback.videoDisplayMode} ${streaming.svpStreamUrl ? 'svp-streaming' : streaming.opticalFlowStreamUrl ? 'interpolated-streaming' : streaming.transcodeStreamUrl ? 'transcode-streaming' : ''}`}
              style={zoomPan.getZoomTransform()}
              onClick={curationMode ? undefined : handleVideoClick}
              onLoadStart={(event) => reportDirectFileStage('loadstart', event.currentTarget)}
              onLoadedData={(event) => reportDirectFileStage('loadeddata', event.currentTarget)}
              onPlay={(event) => {
                playback.handleVideoPlay(event)
                reportSvpPlayback(event.currentTarget)
                reportDirectFileStage('play', event.currentTarget)
              }}
              onPlaying={(event) => reportDirectFileStage('playing', event.currentTarget)}
              onPause={(event) => {
                playback.handleVideoPause(event)
                reportSvpPlayback(event.currentTarget)
                reportDirectFileStage('pause', event.currentTarget)
              }}
              onTimeUpdate={(event) => {
                playback.handleTimeUpdate(event)
                const previousTime = directFileLastTimeRef.current
                const nextTime = event.currentTarget.currentTime
                if (image?.is_local_direct_file && previousTime > 1 && nextTime < previousTime - 1) {
                  reportDirectFileStage(`time-reset-from-${previousTime.toFixed(3)}`, event.currentTarget)
                }
                directFileLastTimeRef.current = nextTime
                if (!directFileStartedRef.current && event.currentTarget.currentTime > 0) {
                  directFileStartedRef.current = true
                  reportDirectFileStage('time-advanced', event.currentTarget)
                }
              }}
              onLoadedMetadata={(event) => {
                handleLoadedMetadataWithResolution(event)
                const resume = svpResumeRef.current
                if (resume && resume.imageKey === currentImageKey) {
                  event.currentTarget.currentTime = resume.currentTime
                  if (!resume.paused) event.currentTarget.play().catch(() => {})
                  svpResumeRef.current = null
                } else if (resume) {
                  svpResumeRef.current = null
                }
                svpTransitionRef.current.active = false
                measureAndReportSvpPlayback(event.currentTarget)
                reportDirectFileStage('loadedmetadata', event.currentTarget)
              }}
              onCanPlay={(event) => {
                handleVideoCanPlay(event)
                reportDirectFileStage('canplay', event.currentTarget)
              }}
              onWaiting={(event) => reportDirectFileStage('waiting', event.currentTarget)}
              onStalled={(event) => reportDirectFileStage('stalled', event.currentTarget)}
              onSeeking={(event) => reportDirectFileStage('seeking', event.currentTarget)}
              onSeeked={(event) => reportDirectFileStage('seeked', event.currentTarget)}
              onEmptied={(event) => reportDirectFileStage('emptied', event.currentTarget)}
              onAbort={(event) => reportDirectFileStage('abort', event.currentTarget)}
              onEnded={(event) => reportDirectFileStage('ended', event.currentTarget)}
              onError={(event) => {
                reportDirectFileStage('error', event.currentTarget)
                failOpenNativeSvp(event.currentTarget)
              }}
              onContextMenu={handleVideoContextMenu}
            />
            {/* Video diagnostics — press I to toggle, B for bare mode */}
            <FPSMonitor videoRef={mediaRef} visible={showDiagnostics} onToggleBare={setDebugBare} />
            {/* Drag-to-seek overlay */}
            {gestures.dragSeek && (
              <div className="video-seek-overlay">
                {gestures.dragSeek.amount > 0 ? '+' : ''}{gestures.dragSeek.amount}s
              </div>
            )}
            {/* Tap seek indicator (left/right flash) */}
            {gestures.seekIndicator && (
              <div
                key={gestures.seekIndicator.key}
                className={`video-tap-seek-indicator ${gestures.seekIndicator.side}`}
              >
                {gestures.seekIndicator.amount > 0 ? '+' : ''}{gestures.seekIndicator.amount}s
              </div>
            )}
            {/* Custom video controls */}
            {!debugBare && <div
              className="lightbox-video-controls"
              onClick={(e) => e.stopPropagation()}
              onTouchStart={(e) => e.stopPropagation()}
              onTouchMove={(e) => e.stopPropagation()}
              onTouchEnd={(e) => e.stopPropagation()}
            >
              {/* Timeline and playback controls */}
              <div className="video-controls-row">
              <span className="video-time" ref={casting.isCasting ? null : playback.timeDisplayRef}>
                {formatTime(casting.isCasting ? (casting.castStatus?.current_time || 0) : playback.currentTime)}
              </span>
              <div
                ref={casting.isCasting ? null : playback.timelineRef}
                className={`video-timeline ${!(casting.isCasting ? casting.castStatus?.duration : playback.duration) ? 'loading' : ''}`}
                onClick={casting.isCasting ? ((e) => {
                  const duration = casting.castStatus?.duration || 0
                  if (!duration) return
                  const rect = e.currentTarget.getBoundingClientRect()
                  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
                  casting.castSeek(pct * duration)
                }) : undefined}
                onMouseDown={casting.isCasting ? undefined : playback.handleSeekStart}
                onMouseMove={(e) => {
                  if (!casting.isCasting) {
                    playback.handleSeekMove(e)
                    timelinePreview.handleTimelineHover(e)
                  }
                }}
                onMouseUp={casting.isCasting ? undefined : playback.handleSeekEnd}
                onMouseLeave={(e) => {
                  if (!casting.isCasting) {
                    playback.handleSeekEnd(e)
                    timelinePreview.handleTimelineHoverEnd()
                  }
                }}
                onTouchStart={casting.isCasting ? undefined : playback.handleSeekTouchStart}
                onTouchMove={casting.isCasting ? undefined : playback.handleSeekTouchMove}
                onTouchEnd={casting.isCasting ? undefined : playback.handleSeekTouchEnd}
              >
                {/* Timeline thumbnail preview */}
                {timelinePreview.hoverTime !== null && timelinePreview.hasPreviewFrames && (
                  <div
                    className="video-timeline-preview"
                    style={{ left: `${timelinePreview.hoverX}px` }}
                  >
                    <img src={timelinePreview.getCurrentFrame()} alt="" />
                    <span className="preview-time">{formatTime(timelinePreview.hoverTime)}</span>
                  </div>
                )}
                <div className="video-timeline-track">
                  {/* Buffer indicator for SVP streams - shows how much is available */}
                  {/* Buffer indicator for transcode streams */}
                  {streaming.transcodeStreamUrl && streaming.transcodeBufferedDuration > 0 && playback.duration > 0 && (
                    <div
                      className="video-timeline-buffer"
                      style={{
                        left: `${(streaming.transcodeStartOffset / playback.duration) * 100}%`,
                        width: `${(streaming.transcodeBufferedDuration / playback.duration) * 100}%`
                      }}
                    />
                  )}
                  {/* Buffer indicator for SVP streams - shows the buffered range */}
                  {streaming.svpStreamUrl && streaming.svpBufferedDuration > 0 && playback.duration > 0 && (
                    <div
                      className="video-timeline-buffer"
                      style={{
                        left: `${(streaming.svpStartOffset / playback.duration) * 100}%`,
                        width: `${(streaming.svpBufferedDuration / playback.duration) * 100}%`
                      }}
                    />
                  )}
                  <div
                    ref={playback.progressBarRef}
                    className="video-timeline-progress"
                    style={{ width: `${casting.isCasting ? (casting.castStatus?.duration ? ((casting.castStatus?.current_time || 0) / casting.castStatus.duration) * 100 : 0) : (playback.duration ? (playback.currentTime / playback.duration) * 100 : 0)}%` }}
                  />
                  <div
                    ref={playback.playheadRef}
                    className="video-timeline-playhead"
                    style={{ left: `${casting.isCasting ? (casting.castStatus?.duration ? ((casting.castStatus?.current_time || 0) / casting.castStatus.duration) * 100 : 0) : (playback.duration ? (playback.currentTime / playback.duration) * 100 : 0)}%` }}
                  />
                </div>
              </div>
              <span className="video-time">{formatTime(casting.isCasting ? (casting.castStatus?.duration || 0) : playback.duration, true)}</span>
              <div className="video-playback-controls">
                {!curationMode && <button className="video-nav-btn" onClick={() => onNav(-1)} title="Previous (Left Arrow)" aria-label="Previous media">
                  <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 6h2v12H6V6zm3.5 6l8.5 6V6l-8.5 6z"/></svg>
                </button>}
                <button
                  className="video-play-btn-center"
                  onClick={casting.isCasting ? (() => casting.castStatus?.state === 'playing' ? casting.castPause() : casting.castResume()) : playback.toggleVideoPlay}
                  title={(casting.isCasting ? casting.castStatus?.state === 'playing' : playback.isPlaying) ? 'Pause (Space)' : 'Play (Space)'}
                  aria-label={(casting.isCasting ? casting.castStatus?.state === 'playing' : playback.isPlaying) ? 'Pause' : 'Play'}
                >
                  {(casting.isCasting ? casting.castStatus?.state === 'playing' : playback.isPlaying) ? (
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/></svg>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                  )}
                </button>
                {!curationMode && <button className="video-nav-btn" onClick={() => onNav(1)} title="Next (Right Arrow)" aria-label="Next media">
                  <svg viewBox="0 0 24 24" fill="currentColor"><path d="M16 6v12h2V6h-2zm-3.5 6l-8.5 6V6l8.5 6z"/></svg>
                </button>}
              </div>
              <div className="video-utility-controls">
              {!casting.isCasting && whisperInstalled && (
              <div className="subtitle-btn-container">
                <button
                  className={`video-control-btn subtitle-btn ${subtitles.subtitlesEnabled ? 'active' : ''} ${subtitles.installing ? 'installing' : ''}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    subtitles.toggleSubtitles()
                  }}
                  onContextMenu={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    setShowSubtitleMenu(!showSubtitleMenu)
                  }}
                  disabled={subtitles.installing}
                  title={subtitles.installing ? 'Installing faster-whisper...' : subtitles.subtitlesEnabled ? 'Hide subtitles (C) | Right-click: language' : 'Show subtitles (C) | Right-click: language'}
                >
                  {subtitles.generating || subtitles.installing ? (
                    <div className="subtitle-spinner" />
                  ) : (
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path d="M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H4V6h16v12zM6 10h2v2H6v-2zm0 4h8v2H6v-2zm10 0h2v2h-2v-2zm-6-4h8v2h-8v-2z"/>
                    </svg>
                  )}
                </button>
                <button
                  className="subtitle-menu-arrow"
                  onClick={(e) => {
                    e.stopPropagation()
                    setShowSubtitleMenu(!showSubtitleMenu)
                  }}
                  title="Subtitle language & task"
                  aria-label="Subtitle settings"
                >
                  <svg viewBox="0 0 12 8" fill="currentColor">
                    <path d="M1.41 7.41L6 2.83l4.59 4.58L12 6 6 0 0 6l1.41 1.41z"/>
                  </svg>
                </button>
              </div>
              )}
              {!casting.isCasting && svpInstalled && (
              <button
                className={`video-control-btn svp-toggle-btn ${streaming.svpConfig?.enabled ? 'active' : ''} ${streaming.svpLoading ? 'loading' : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  handleToggleSVP()
                }}
                title={streaming.svpConfig?.enabled ? 'Disable SVP interpolation' : 'Enable SVP interpolation'}
              >
                {streaming.svpLoading ? (
                  <div className="svp-toggle-spinner" />
                ) : (
                  <span className="svp-toggle-label">SVP</span>
                )}
              </button>
              )}
              {!casting.isCasting && <button
                className={`video-control-btn quality-btn ${currentQuality !== 'original' ? 'active' : ''}`}
                onClick={(e) => {
                  e.stopPropagation()
                  setShowQualitySelector(!showQualitySelector)
                }}
                title={`Quality: ${currentQuality === 'original' ? 'Original' : currentQuality}`}
              >
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zm-5.04-6.71l-2.75 3.54h2.79v2.71h2V13.83h2.79l-2.75-3.54zM7 9h2v2H7z"/>
                </svg>
              </button>}
              <div className="video-volume-container">
                <button
                  className="video-control-btn video-mute-btn"
                  onClick={casting.isCasting ? undefined : playback.toggleMute}
                  title={casting.isCasting ? 'Chromecast volume' : playback.isMuted ? 'Unmute (M)' : 'Mute (M)'}
                >
                  {(!casting.isCasting && (playback.isMuted || playback.volume === 0)) ? (
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>
                    </svg>
                  ) : playback.volume < 0.5 ? (
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path d="M18.5 12c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM5 9v6h4l5 5V4L9 9H5z"/>
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                    </svg>
                  )}
                </button>
                <input
                  type="range"
                  className="video-volume-slider"
                  min="0"
                  max="1"
                  step="0.05"
                  value={casting.isCasting ? (casting.castStatus?.volume ?? 1) : (playback.isMuted ? 0 : playback.volume)}
                  onChange={casting.isCasting ? ((e) => casting.castVolume(parseFloat(e.target.value))) : playback.handleVolumeChange}
                  title={`Volume: ${Math.round((casting.isCasting ? (casting.castStatus?.volume ?? 1) : (playback.isMuted ? 0 : playback.volume)) * 100)}%`}
                />
              </div>
              <button
                className={`video-control-btn video-fullscreen-btn ${isFullscreen ? 'active' : ''}`}
                onClick={handleToggleFullscreen}
                title={isFullscreen ? 'Exit fullscreen (F)' : 'Fullscreen (F)'}
              >
                {isFullscreen ? (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M8 3H5a2 2 0 0 0-2 2v3M21 8V5a2 2 0 0 0-2-2h-3M3 16v3a2 2 0 0 0 2 2h3M16 21h3a2 2 0 0 0 2-2v-3"/>
                  </svg>
                ) : (
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3"/>
                  </svg>
                )}
              </button>
              </div>
              </div>
            </div>}
            {/* Playback speed badge */}
            {playback.playbackSpeed !== 1.0 && (
              <div className="playback-speed-badge">
                {playback.playbackSpeed.toFixed(2).replace(/\.?0+$/, '')}x
              </div>
            )}
            {/* Optical flow loading indicator */}
            {streaming.opticalFlowLoading && (
              <div className="interpolate-loading">
                <div className="interpolate-loading-spinner" />
                <span>Buffering {streaming.opticalFlowConfig?.target_fps || 60} FPS...</span>
              </div>
            )}
            {/* Optical flow streaming indicator */}
            {streaming.opticalFlowStreamUrl && !streaming.opticalFlowLoading && (
              <div className="interpolate-badge">
                {streaming.opticalFlowConfig?.target_fps || 60} FPS
              </div>
            )}
            {/* Optical flow error toast */}
            {streaming.opticalFlowError && (
              <div className="interpolate-error-toast">
                {streaming.opticalFlowError}
              </div>
            )}
            {/* SVP loading indicator */}
            {streaming.svpLoading && (
              <div className="interpolate-loading svp-loading">
                <div className="interpolate-loading-spinner" />
                <span>SVP: Buffering {streaming.svpConfig?.target_fps || 60} FPS...</span>
                <button className="svp-cancel-btn" onClick={streaming.cancelSVPLoading} title="Cancel SVP">✕</button>
              </div>
            )}
            {/* SVP streaming indicator */}
            {streaming.svpStreamUrl && !streaming.svpLoading && !streaming.svpPendingSeek && (
              <div className="interpolate-badge svp-badge">
                SVP {streaming.svpConfig?.target_fps || 60} FPS
              </div>
            )}
            {/* SVP waiting for seek indicator */}
            {streaming.svpPendingSeek && (
              <div className="interpolate-loading svp-loading">
                <div className="interpolate-loading-spinner" />
                <span>Buffering to {formatTime(streaming.svpPendingSeek)}...</span>
                <button className="svp-cancel-btn" onClick={streaming.cancelSVPLoading} title="Cancel SVP">✕</button>
              </div>
            )}
            {/* SVP error toast */}
            {streaming.svpError && (
              <div className="interpolate-error-toast svp-error">
                SVP: {streaming.svpError}
              </div>
            )}
            {/* Generic stream error toast */}
            {streaming.streamError && (
              <div className="interpolate-error-toast">
                {streaming.streamError}
              </div>
            )}
            {/* Subtitle install progress */}
            {subtitles.installing && (
              <div className="subtitle-progress-badge installing">
                <div className="subtitle-progress-spinner" />
                <span>Installing faster-whisper...</span>
              </div>
            )}
            {/* Subtitle generation progress */}
            {subtitles.generating && (
              <div className="subtitle-progress-badge">
                <div className="subtitle-progress-spinner" />
                <span>Subtitles: {Math.round(subtitles.progress)}%</span>
              </div>
            )}
            {/* Subtitle error toast */}
            {subtitles.error && (
              <div className="interpolate-error-toast subtitle-error">
                {subtitles.error}
              </div>
            )}
            {/* Cast remote control overlay */}
            {casting.isCasting && (
              <div className="cast-overlay" data-curation-gesture-block onClick={(e) => e.stopPropagation()}>
                <div className="cast-overlay-icon-wrap">
                  <span className="cast-overlay-pulse" />
                  <span className="cast-overlay-pulse delay" />
                  <svg viewBox="0 0 24 24" fill="currentColor" className="cast-overlay-icon">
                    <path d="M1 18v3h3c0-1.66-1.34-3-3-3zm0-4v2c2.76 0 5 2.24 5 5h2c0-3.87-3.13-7-7-7zm18-7H5v1.63c3.96 1.28 7.09 4.41 8.37 8.37H19V7zM1 10v2c4.97 0 9 4.03 9 9h2c0-6.08-4.93-11-11-11zm20-7H3c-1.1 0-2 .9-2 2v3h2V5h18v14h-7v2h7c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/>
                  </svg>
                </div>
                <div className="cast-overlay-status">
                  {casting.castStatus?.state === 'paused' ? 'Paused on' : 'Playing on'}
                </div>
                <div className="cast-overlay-device">
                  {casting.activeDevice?.name || 'TV'}
                </div>
                {(casting.castStatus?.title || image?.original_filename) && (
                  <div className="cast-overlay-title">
                    {casting.castStatus?.title || image?.original_filename}
                  </div>
                )}
                <button className="cast-stop-btn" onClick={casting.stopCasting}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/>
                    <line x1="2" y1="20" x2="2.01" y2="20"/>
                    <line x1="17" y1="9" x2="9" y2="17"/>
                  </svg>
                  Stop Casting
                </button>
              </div>
            )}
            {/* Cast error toast */}
            {casting.castError && (
              <div className="interpolate-error-toast cast-error">
                {casting.castError}
              </div>
            )}
            {/* Resume playback toast */}
            {resumePosition && (
              <div className="resume-toast" data-curation-gesture-block onClick={(e) => e.stopPropagation()}>
                <span>Resume from {formatTime(resumePosition.position)}?</span>
                <div className="resume-toast-actions">
                  <button className="resume-toast-btn" onClick={() => {
                    playback.seekVideo(resumePosition.position - playback.currentTimeRef.current)
                    setResumePosition(null)
                  }}>Resume</button>
                  <button className="resume-toast-btn dismiss" onClick={() => setResumePosition(null)}>Start Over</button>
                </div>
              </div>
            )}
            {/* Auto-advance countdown overlay */}
            {autoAdvance.countdown !== null && (
              <div className="auto-advance-overlay" data-curation-gesture-block onClick={(e) => e.stopPropagation()}>
                {/* Next item thumbnail preview */}
                {images[currentIndex + 1] && (
                  <img
                    className="auto-advance-thumbnail"
                    src={getMediaUrl(images[currentIndex + 1].thumbnail_url)}
                    alt=""
                  />
                )}
                <div className="auto-advance-info">
                  <span className="auto-advance-text">Next in {autoAdvance.countdown}s</span>
                  <div className="auto-advance-actions">
                    <button className="auto-advance-btn cancel" onClick={autoAdvance.cancelCountdown}>Cancel</button>
                    <button className="auto-advance-btn advance" onClick={autoAdvance.advanceNow}>Play Now</button>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : imageLoadError ? (
          <div className="lightbox-unavailable missing">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <path d="M15 9l-6 6M9 9l6 6"/>
            </svg>
            <h3>Image Could Not Be Loaded</h3>
            <p>The media request failed or this image could not be decoded.</p>
            <button
              className="lightbox-confirm-cancel"
              onClick={(event) => {
                event.stopPropagation()
                setImageLoadError(false)
                setImageRetryKey(value => value + 1)
              }}
            >
              Retry
            </button>
          </div>
        ) : (
          <img
            key={`${previewUrl ? `${image.id}-preview` : image.id}-${imageRetryKey}`}
            ref={mediaRef}
            src={renderedImageUrl}
            alt=""
            className="lightbox-media"
            style={{ ...(previewUrl ? {} : getFilterStyle()), ...zoomPan.getZoomTransform() }}
            onContextMenu={handleImageContextMenu}
            onError={(event) => {
              console.warn('[Lightbox] Media load failed', {
                imageId: image.id,
                directoryId: image.directory_id,
                libraryId: image.library_id,
                source: event.currentTarget.currentSrc || event.currentTarget.src,
              })
              if (imageSourceOwnerRef.current.owns(renderedImageSource)) {
                setImageLoadError(true)
              }
            }}
          />
        )}
      </div>

      {curationMode && (
        <>
          <div className="curation-status">
            <strong>Curated {curationMode.processed}</strong>
            <span>{curationMode.remaining} queued</span>
            {curationMode.goal?.enabled && (
              <span>{curationMode.goal.cadence === 'weekly' ? 'Weekly' : 'Daily'} goal {curationMode.progress} / {curationMode.goal.target}</span>
            )}
          </div>
          <div className="curation-controls" onClick={event => event.stopPropagation()}>
            <button className="curation-action discard" onClick={curationMode.onDiscard} disabled={curationMode.busy} title="Discard (D or Left Arrow)">
              <span aria-hidden="true">✕</span> Discard
            </button>
            <button className="curation-undo" onClick={curationMode.onUndo} disabled={curationMode.busy || !curationMode.lastAction} title="Undo last action (U)">
              ↶ <span>Undo</span>
            </button>
            <button className="curation-action keep" onClick={curationMode.onKeep} disabled={curationMode.busy} title="Keep (K or Right Arrow)">
              <span aria-hidden="true">★</span> Keep
            </button>
          </div>
        </>
      )}

      {!isVideoFile && (
        <div className="lightbox-counter">
          {currentIndex + 1} / {total}
        </div>
      )}

      {/* Copy feedback toast */}
      {copyFeedback && (
        <div className={`lightbox-copy-toast ${copyFeedback}`}>
          {copyFeedback === 'success' ? 'Copied to clipboard!' : 'Cannot copy this file'}
        </div>
      )}

      {/* Mobile favorite button - bottom center like camera shutter */}
      <button
        className={`lightbox-mobile-favorite ${isFavorited ? 'active' : ''}`}
        onClick={handleToggleFavorite}
        disabled={processing}
      >
        <svg viewBox="0 0 24 24" fill={isFavorited ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2">
          <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
        </svg>
      </button>

      {/* Delete confirmation dialog */}
      {showDeleteConfirm && (
        <div className="lightbox-confirm-overlay" onClick={() => setShowDeleteConfirm(false)}>
          <div className="lightbox-confirm-dialog" onClick={e => e.stopPropagation()}>
            <h3>Delete Image?</h3>
            <p>This will permanently delete the file from your filesystem. This action cannot be undone.</p>
            <div className="lightbox-confirm-actions">
              <button
                ref={cancelBtnRef}
                className="lightbox-confirm-cancel"
                onClick={() => setShowDeleteConfirm(false)}
                disabled={processing}
              >
                Cancel
              </button>
              <button
                ref={deleteBtnRef}
                className="lightbox-confirm-delete"
                onClick={handleDelete}
                disabled={processing}
              >
                {processing ? 'Deleting...' : 'Delete'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* SVP side menu */}
      {isVideoFile && svpInstalled && (
        <SVPSideMenu
          isOpen={showSVPMenu}
          onClose={async () => {
            setShowSVPMenu(false)
            const playbackIntent = streaming.capturePlaybackIntent()
            const generation = ++svpToggleGenerationRef.current
            const imageKey = currentImageKey
            try {
              const newConfig = await getSVPConfig()
              if (generation !== svpToggleGenerationRef.current
                  || imageKey !== activeImageKeyRef.current) return
              const configChanged = JSON.stringify(newConfig) !== JSON.stringify(streaming.svpConfig)
              svpDesiredEnabledRef.current = Boolean(newConfig.enabled)
              streaming.setSvpConfig(newConfig)
              if (!configChanged) return

              streaming.setSvpError(null)
              if (!newConfig.enabled) {
                setCurrentQuality('original')
                localStorage.setItem('video_quality_preference', 'original')
                await streaming.handleQualityChange('original', playbackIntent)
              } else if (!streaming.nativeSvpPlayback && image && isVideo(image.filename)) {
                await streaming.stopSVP()
                if (generation !== svpToggleGenerationRef.current
                    || imageKey !== activeImageKeyRef.current) return
                await streaming.startSVPStream(playbackIntent.position, playbackIntent, true, null, true)
              }
            } catch (err) {
              if (generation !== svpToggleGenerationRef.current
                  || imageKey !== activeImageKeyRef.current) return
              console.error('Failed to reload SVP config:', err)
            }
          }}
          image={image}
        />
      )}

      {/* Quality selector */}
      {isVideoFile && !(streaming.nativeSvpPlayback && streaming.svpConfig?.enabled) && (
        <QualitySelector
          isOpen={showQualitySelector}
          onClose={() => setShowQualitySelector(false)}
          currentQuality={currentQuality}
          onQualityChange={handleQualityChange}
          sourceResolution={streaming.sourceResolution}
        />
      )}

      {/* Subtitle language/task menu */}
      {whisperInstalled && showSubtitleMenu && (
        <>
          <div className="subtitle-menu-popup" onClick={(e) => e.stopPropagation()}>
            <div className="subtitle-menu-header">
              <span>Subtitles</span>
              <button
                className={`subtitle-menu-toggle ${subtitles.subtitlesEnabled ? 'active' : ''}`}
                onClick={subtitles.toggleSubtitles}
              >
                {subtitles.subtitlesEnabled ? 'On' : 'Off'}
              </button>
              <button className="subtitle-menu-close" onClick={() => setShowSubtitleMenu(false)} aria-label="Close subtitle settings">×</button>
            </div>
            <div className="subtitle-menu-section">
              <div className="subtitle-menu-label">Source Language</div>
              <div className="subtitle-menu-options">
                {[
                  { value: '', label: 'Auto-detect' },
                  { value: 'ja', label: 'Japanese' },
                  { value: 'en', label: 'English' },
                  { value: 'zh', label: 'Chinese' },
                  { value: 'ko', label: 'Korean' },
                  { value: 'de', label: 'German' },
                  { value: 'fr', label: 'French' },
                  { value: 'es', label: 'Spanish' },
                  { value: 'ru', label: 'Russian' },
                ].map(lang => (
                  <button
                    key={lang.value}
                    className={`subtitle-menu-option ${subtitles.subtitleLanguage === lang.value ? 'active' : ''}`}
                    onClick={() => {
                      const newLang = lang.value
                      const currentTask = subtitles.subtitleTask
                      setShowSubtitleMenu(false)
                      if (newLang !== subtitles.subtitleLanguage) {
                        subtitles.restartWithSettings(newLang, currentTask)
                      }
                    }}
                  >
                    <span>{lang.label}</span>
                    {subtitles.subtitleLanguage === lang.value && (
                      <svg className="subtitle-menu-check" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                      </svg>
                    )}
                  </button>
                ))}
              </div>
            </div>
            <div className="subtitle-menu-section">
              <div className="subtitle-menu-label">Output</div>
              <div className="subtitle-menu-options">
                {[
                  { value: 'translate', label: 'Translate to English' },
                  { value: 'transcribe', label: 'Transcribe (original language)' },
                ].map(t => (
                  <button
                    key={t.value}
                    className={`subtitle-menu-option ${subtitles.subtitleTask === t.value ? 'active' : ''}`}
                    onClick={() => {
                      const newTask = t.value
                      const currentLang = subtitles.subtitleLanguage
                      setShowSubtitleMenu(false)
                      if (newTask !== subtitles.subtitleTask) {
                        subtitles.restartWithSettings(currentLang, newTask)
                      }
                    }}
                  >
                    <span>{t.label}</span>
                    {subtitles.subtitleTask === t.value && (
                      <svg className="subtitle-menu-check" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
                      </svg>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="subtitle-menu-backdrop" onClick={() => setShowSubtitleMenu(false)} />
        </>
      )}

      {contextMenu && (
        <ContextMenu
          position={contextMenu}
          onClose={() => setContextMenu(null)}
          items={[
            {
              label: 'Copy Image',
              onClick: handleCopyImage
            },
            {
              label: 'Paste Image',
              onClick: handlePasteImage,
              disabled: !image?.directory_id
            }
          ]}
        />
      )}
    </div>
  )
}

export default Lightbox
