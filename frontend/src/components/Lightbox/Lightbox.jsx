import { useEffect, useCallback, useState, useRef, useMemo } from 'react'
import { getMediaUrl, getSVPConfig, updateSVPConfig, stopSVPStream, stopInterpolatedStream, getPlaybackPosition, fetchCollections, addToCollection, createCollection, getShareNetworkInfo } from '../../api'
import { getDesktopAPI } from '../../tauriAPI'
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

function Lightbox({ images, currentIndex, total, onClose, onNav, onTagClick, onImageUpdate, onSidebarHover, sidebarOpen, onDelete }) {
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
  const [adjustments, setAdjustments] = useState({ brightness: 0, contrast: 0, gamma: 0 })
  const [applyingAdjustments, setApplyingAdjustments] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [generatingPreview, setGeneratingPreview] = useState(false)

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

  // Refs
  const mediaRef = useRef(null)
  const containerRef = useRef(null)

  const image = images[currentIndex]

  // UI visibility and fullscreen hook
  const {
    showUI,
    isFullscreen,
    resetHideTimer,
    handleMouseMove,
    handleToggleFullscreen
  } = useUIVisibility(containerRef)

  // Video streaming hook
  const streaming = useVideoStreaming(mediaRef, image, currentQuality)

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
    restartTranscodeFromPosition: streaming.restartTranscodeFromPosition
  }, image?.id)

  // Zoom and pan hook
  const zoomPan = useZoomPan(mediaRef, containerRef, resetHideTimer, image)

  // Timeline preview hook (for video thumbnail preview on hover)
  const timelinePreview = useTimelinePreview(
    image?.id,
    image?.directory_id,
    playback.duration
  )

  // Whisper subtitle hook
  const subtitles = useWhisperSubtitles(mediaRef, image)

  // Auto-advance hook
  const autoAdvance = useAutoAdvance(mediaRef, {
    onNav,
    currentIndex,
    totalImages: images.length,
    isVideoFile: isVideo(image?.original_filename),
  })

  // Share stream hook (host side)
  const shareStream = useShareStream(mediaRef, {
    imageId: image?.id,
    directoryId: image?.directory_id,
    isVideoFile: isVideo(image?.original_filename),
  })

  // Cast session hook (Chromecast / DLNA)
  const casting = useCastSession(mediaRef, image)

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
  useEffect(() => {
    setAdjustments({ brightness: 0, contrast: 0, gamma: 0 })
    setShowAdjustments(false)
    setPreviewUrl(null)
    zoomPan.resetZoom()
    playback.resetPlaybackState()
    subtitles.stopSubtitlesStream()
    setShowSubtitleMenu(false)
  }, [image?.id])

  // Check for resume position when opening a video
  useEffect(() => {
    if (!image || !isVideo(image.filename)) return
    setResumePosition(null)
    clearTimeout(resumeTimerRef.current)

    getPlaybackPosition(image.id).then(data => {
      if (data.position > 10 && !data.completed) {
        setResumePosition(data)
        // Auto-dismiss after 5s
        resumeTimerRef.current = setTimeout(() => setResumePosition(null), 5000)
      }
    }).catch(() => {})

    return () => clearTimeout(resumeTimerRef.current)
  }, [image?.id])

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
      const { toggleFavorite } = await import('../../api')
      const result = await toggleFavorite(image.id)
      setIsFavorited(result.is_favorite)
      // Update parent state so the change persists when navigating
      if (onImageUpdate) {
        onImageUpdate(image.id, { is_favorite: result.is_favorite })
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
      const { deleteImage } = await import('../../api')
      await deleteImage(image.id, true, image.directory_id) // true = delete from filesystem, pass directory_id if available
      setShowDeleteConfirm(false)

      // Notify parent to remove the image and navigate
      if (onDelete) {
        onDelete(image.id)
      }
    } catch (err) {
      console.error('Failed to delete image:', err)
      alert('Failed to delete image: ' + err.message)
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

  // Handle click on video to play/pause
  const handleVideoClick = useCallback((e) => {
    if (!isVideo(image?.original_filename)) return
    // Don't toggle play if clicking on controls
    if (e.target.closest('.lightbox-video-controls')) return
    playback.toggleVideoPlay()
    resetHideTimer()
  }, [image?.original_filename, playback, resetHideTimer])

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
    setCurrentQuality(qualityId)
    localStorage.setItem('video_quality_preference', qualityId)
    await streaming.handleQualityChange(qualityId, playback.setCurrentTime)
  }, [streaming, playback])

  // Toggle SVP on/off
  const handleToggleSVP = useCallback(async () => {
    const newEnabled = !streaming.svpConfig?.enabled
    try {
      const updatedConfig = await updateSVPConfig({ enabled: newEnabled })
      streaming.setSvpConfig(updatedConfig)

      if (newEnabled) {
        // Auto-start effect watches svpConfig.enabled and will start the stream
      } else {
        // Stop SVP stream
        if (streaming.svpStreamUrl) {
          await stopSVPStream()
          streaming.setSvpStreamUrl(null)
        }
        streaming.setSvpError(null)
        streaming.setSvpLoading(false)
        streaming.svpStartingRef.current = false
      }
    } catch (err) {
      console.error('Failed to toggle SVP:', err)
    }
  }, [streaming, image])

  // Generate preview of adjustments
  const handleGeneratePreview = useCallback(async () => {
    if (!image || generatingPreview) return

    // Check if any adjustments were made
    if (adjustments.brightness === 0 && adjustments.contrast === 0 && adjustments.gamma === 0) {
      return
    }

    setGeneratingPreview(true)
    try {
      const { previewImageAdjustments } = await import('../../api')
      const result = await previewImageAdjustments(image.id, {
        brightness: adjustments.brightness,
        contrast: adjustments.contrast,
        gamma: adjustments.gamma
      })

      // Add cache buster to prevent browser caching
      setPreviewUrl(`${result.preview_url}?t=${Date.now()}`)
    } catch (err) {
      console.error('Failed to generate preview:', err)
      alert('Failed to generate preview: ' + err.message)
    }
    setGeneratingPreview(false)
  }, [image, adjustments, generatingPreview])

  // Discard preview and go back to CSS filter mode
  const handleDiscardPreview = useCallback(async () => {
    if (!image) return

    try {
      const { discardImagePreview } = await import('../../api')
      await discardImagePreview(image.id)
    } catch (err) {
      console.error('Failed to discard preview:', err)
    }
    setPreviewUrl(null)
  }, [image])

  // Apply adjustments to file
  const handleApplyAdjustments = useCallback(async () => {
    if (!image || applyingAdjustments) return

    // Check if any adjustments were made
    if (adjustments.brightness === 0 && adjustments.contrast === 0 && adjustments.gamma === 0) {
      return
    }

    setApplyingAdjustments(true)
    try {
      const { applyImageAdjustments, discardImagePreview } = await import('../../api')
      const result = await applyImageAdjustments(image.id, {
        brightness: adjustments.brightness,
        contrast: adjustments.contrast,
        gamma: adjustments.gamma
      })

      // Clean up preview cache
      try {
        await discardImagePreview(image.id)
      } catch (e) {
        // Ignore cleanup errors
      }

      // Force reload the image by updating the URL with a cache buster
      if (onImageUpdate) {
        onImageUpdate(image.id, {
          url: `${image.url.split('?')[0]}?t=${Date.now()}`,
          thumbnail_url: `${image.thumbnail_url.split('?')[0]}?t=${Date.now()}`
        })
      }

      // Reset adjustments and preview after applying
      setAdjustments({ brightness: 0, contrast: 0, gamma: 0 })
      setPreviewUrl(null)
      setShowAdjustments(false)
    } catch (err) {
      console.error('Failed to apply adjustments:', err)
      alert('Failed to apply adjustments: ' + err.message)
    }
    setApplyingAdjustments(false)
  }, [image, adjustments, applyingAdjustments, onImageUpdate])

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

    // Gamma: exponential mapping (same as backend)
    // slider -100 to +100 → exponent 3.0 to 0.33
    const gammaExponent = Math.pow(3.0, -adjustments.gamma / 100)

    // Build filter string: brightness and contrast via CSS, gamma via SVG
    const filters = []

    if (adjustments.brightness !== 0 || adjustments.contrast !== 0) {
      filters.push(`brightness(${cssBrightness}) contrast(${cssContrast})`)
    }

    if (adjustments.gamma !== 0) {
      const svgFilter = `url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg"><filter id="g"><feComponentTransfer><feFuncR type="gamma" exponent="${gammaExponent}"/><feFuncG type="gamma" exponent="${gammaExponent}"/><feFuncB type="gamma" exponent="${gammaExponent}"/></feComponentTransfer></filter></svg>#g')`
      filters.push(svgFilter)
    }

    return {
      filter: filters.join(' ')
    }
  }

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return

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

      // VLC-like video controls
      if (isVideoFile && mediaRef.current) {
        switch (e.key) {
          case ' ':
            e.preventDefault()
            playback.toggleVideoPlay()
            return
          case 'ArrowLeft':
            e.preventDefault()
            if (e.ctrlKey || e.metaKey) {
              playback.seekVideo(-30) // Ctrl+Left: -30s
            } else if (e.shiftKey) {
              playback.seekVideo(-1) // Shift+Left: -1s
            } else {
              playback.seekVideo(-5) // Left: -5s
            }
            return
          case 'ArrowRight':
            e.preventDefault()
            if (e.ctrlKey || e.metaKey) {
              playback.seekVideo(30) // Ctrl+Right: +30s
            } else if (e.shiftKey) {
              playback.seekVideo(1) // Shift+Right: +1s
            } else {
              playback.seekVideo(5) // Right: +5s
            }
            return
          case 'ArrowUp':
            e.preventDefault()
            playback.adjustVolume(0.05) // Volume +5%
            return
          case 'ArrowDown':
            e.preventDefault()
            playback.adjustVolume(-0.05) // Volume -5%
            return
          case 'm':
          case 'M':
            e.preventDefault()
            playback.toggleMute()
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
            playback.increaseSpeed() // Speed +0.25x
            return
          case '-':
          case '[':
            e.preventDefault()
            playback.decreaseSpeed() // Speed -0.25x
            return
          case 'Backspace':
            e.preventDefault()
            playback.resetSpeed() // Reset to 1.0x
            return
          case 'e':
          case 'E':
            e.preventDefault()
            playback.frameAdvance() // Frame advance (when paused)
            return
          case 'c':
          case 'C':
            e.preventDefault()
            subtitles.toggleSubtitles()
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
  }, [onNav, onClose, handleToggleFavorite, handleCopyImage, handleDelete, showDeleteConfirm, playback, image?.original_filename, handleToggleFullscreen, subtitles])

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
    // Don't navigate if we handled this as a touch gesture or if touch moved
    if (zoomPan.touchMoved.current || zoomPan.touchHandled.current) {
      zoomPan.touchMoved.current = false
      zoomPan.touchHandled.current = false
      return
    }

    // Don't navigate if zoomed in
    if (zoomPan.zoom.scale > 1) return

    // Don't navigate on videos (navigation uses buttons only)
    if (isVideo(image?.original_filename)) return

    // Don't navigate if clicking on interactive elements (but allow video area for navigation)
    if (e.target.closest('.lightbox-toolbar, .lightbox-counter, .lightbox-confirm-overlay, .lightbox-video-controls')) return

    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const width = rect.width

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

  // Handle touch end with sidebar callback
  const handleTouchEndWithSidebar = useCallback((e) => {
    zoomPan.handleTouchEnd(e, onSidebarHover, sidebarOpen)
  }, [zoomPan, onSidebarHover, sidebarOpen])

  // Handle loaded metadata with source resolution setter
  const handleLoadedMetadataWithResolution = useCallback(() => {
    playback.handleLoadedMetadata(streaming.setSourceResolution)
  }, [playback.handleLoadedMetadata, streaming.setSourceResolution])

  // Handle video canplay event - ensure video plays even if autoPlay is blocked
  // Also reset hide timer to ensure auto-hide works on mobile/Capacitor
  // Also check if browser can decode the video codec (fallback to transcode if not)
  const handleVideoCanPlay = useCallback((e) => {
    e.target.play().catch(() => {})
    resetHideTimer()
    streaming.checkCodecFallback(e.target)
  }, [resetHideTimer, streaming.checkCodecFallback])

  // Handle video context menu
  const handleVideoContextMenu = useCallback((e) => {
    e.preventDefault()
    const desktopAPI = getDesktopAPI()
    if (desktopAPI?.showImageContextMenu) {
      desktopAPI.showImageContextMenu({
        imageUrl: getMediaUrl(image?.url),
        filePath: image?.file_path,
        isVideo: true
      })
    }
  }, [image?.url, image?.file_path])

  // Determine if we should play the video directly (no streaming)
  const shouldPlayDirect = useMemo(() => {
    return streaming.svpConfigLoaded
      && !streaming.svpStreamUrl
      && !streaming.opticalFlowStreamUrl
      && !streaming.transcodeStreamUrl
      && !streaming.svpLoading
      && !streaming.codecFallbackActive
      && currentQuality === 'original'
      && (!streaming.svpConfig?.enabled || streaming.svpError)
      && (!streaming.opticalFlowConfig?.enabled || streaming.opticalFlowError)
  }, [
    streaming.svpConfigLoaded, streaming.svpStreamUrl, streaming.opticalFlowStreamUrl,
    streaming.transcodeStreamUrl, streaming.svpLoading, streaming.codecFallbackActive,
    currentQuality, streaming.svpConfig?.enabled, streaming.svpError,
    streaming.opticalFlowConfig?.enabled, streaming.opticalFlowError
  ])

  if (!image) return null

  const isVideoFile = isVideo(image.original_filename)
  const fileStatus = image.file_status || 'available'
  const isUnavailable = fileStatus !== 'available'

  return (
    <div
      className={`lightbox ${!showUI ? 'ui-hidden' : ''} ${zoomPan.zoom.scale > 1 ? 'zoomed' : ''} ${isFullscreen ? 'fullscreen' : ''} ${isVideoFile ? 'lightbox-video' : ''}`}
      onClick={handleNavClick}
      onDoubleClick={handleDoubleClick}
      onMouseMove={(e) => { handleMouseMove(); zoomPan.handleMouseMoveDrag(e); }}
      onMouseDown={zoomPan.handleMouseDown}
      onMouseUp={zoomPan.handleMouseUp}
      onMouseLeave={zoomPan.handleMouseUp}
      onTouchStart={zoomPan.handleTouchStart}
      onTouchMove={(e) => { zoomPan.handleTouchMove(e); zoomPan.handleTouchMoveZoom(e); }}
      onTouchEnd={(e) => { handleTouchEndWithSidebar(e); zoomPan.handleTouchEndZoom(); }}
      ref={containerRef}
    >
      {/* Top toolbar */}
      <div className="lightbox-toolbar">
        <button
          className="lightbox-btn lightbox-menu"
          onClick={() => onSidebarHover && onSidebarHover(!sidebarOpen)}
          title="Toggle sidebar"
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
        {isVideo(image?.original_filename) && casting.castConfig?.enabled && (
          <div className="lightbox-cast-container">
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
              <div className="cast-device-picker" onClick={(e) => e.stopPropagation()}>
                <div className="cast-picker-header">
                  <span>Cast to</span>
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
                    {casting.devicesLoading ? 'Scanning...' : 'No devices found'}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        {isVideo(image?.original_filename) && (
          <div className="lightbox-share-container">
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
          className="lightbox-btn lightbox-delete"
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
          <div className="lightbox-adjust-container">
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
                    onChange={e => {
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
                    onChange={e => {
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
                    onChange={e => {
                      setAdjustments(prev => ({ ...prev, gamma: parseInt(e.target.value) }))
                      if (previewUrl) setPreviewUrl(null) // Clear stale preview
                    }}
                  />
                </div>
                <div className="adjustment-actions">
                  <button
                    className="adjustment-reset"
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
                    disabled={generatingPreview || (adjustments.brightness === 0 && adjustments.contrast === 0 && adjustments.gamma === 0)}
                  >
                    {generatingPreview ? 'Loading...' : previewUrl ? 'CSS' : 'Preview'}
                  </button>
                  <button
                    className="adjustment-apply"
                    onClick={handleApplyAdjustments}
                    disabled={applyingAdjustments || (adjustments.brightness === 0 && adjustments.contrast === 0 && adjustments.gamma === 0)}
                  >
                    {applyingAdjustments ? 'Saving...' : 'Apply'}
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
        {isVideoFile && (
          <button
            className="lightbox-btn lightbox-svp"
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
        {isVideoFile ? (
          /* Videos: cycle display mode button */
          <button
            className={`lightbox-btn lightbox-display-mode ${playback.videoDisplayMode !== 'fit' ? 'active' : ''}`}
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
            className={`lightbox-btn lightbox-fullscreen ${isFullscreen ? 'active' : ''}`}
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
        <button className="lightbox-btn lightbox-close" onClick={onClose} title="Close (Esc)">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      </div>

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
          <div className="lightbox-video-container">
            <video
              key={image.id}
              ref={mediaRef}
              src={shouldPlayDirect ? getMediaUrl(image.url) : undefined}
              autoPlay
              playsInline
              loop={!autoAdvance.isEnabled}
              className={`lightbox-media video-display-${playback.videoDisplayMode} ${streaming.svpStreamUrl ? 'svp-streaming' : streaming.opticalFlowStreamUrl ? 'interpolated-streaming' : streaming.transcodeStreamUrl ? 'transcode-streaming' : ''}`}
              style={zoomPan.getZoomTransform()}
              onClick={handleVideoClick}
              onPlay={playback.handleVideoPlay}
              onPause={playback.handleVideoPause}
              onTimeUpdate={playback.handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadataWithResolution}
              onCanPlay={handleVideoCanPlay}
              onContextMenu={handleVideoContextMenu}
            />
            {/* Custom video controls */}
            <div
              className="lightbox-video-controls"
              onClick={(e) => e.stopPropagation()}
              onTouchStart={(e) => e.stopPropagation()}
              onTouchMove={(e) => e.stopPropagation()}
              onTouchEnd={(e) => e.stopPropagation()}
            >
              {/* Centered playback controls */}
              <div className="video-playback-controls">
                <button
                  className="video-nav-btn"
                  onClick={() => onNav(-1)}
                  title="Previous (Left Arrow)"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 6h2v12H6V6zm3.5 6l8.5 6V6l-8.5 6z"/>
                  </svg>
                </button>
                <button
                  className="video-play-btn-center"
                  onClick={playback.toggleVideoPlay}
                  title={playback.isPlaying ? 'Pause (Space)' : 'Play (Space)'}
                >
                  {playback.isPlaying ? (
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                    </svg>
                  ) : (
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path d="M8 5v14l11-7z"/>
                    </svg>
                  )}
                </button>
                <button
                  className="video-nav-btn"
                  onClick={() => onNav(1)}
                  title="Next (Right Arrow)"
                >
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M16 6v12h2V6h-2zm-3.5 6l-8.5 6V6l8.5 6z"/>
                  </svg>
                </button>
              </div>
              {/* Timeline row */}
              <div className="video-controls-row">
              <span className="video-time">{formatTime(playback.currentTime)}</span>
              <div
                ref={playback.timelineRef}
                className={`video-timeline ${!playback.duration ? 'loading' : ''}`}
                onMouseDown={playback.handleSeekStart}
                onMouseMove={(e) => {
                  playback.handleSeekMove(e)
                  timelinePreview.handleTimelineHover(e)
                }}
                onMouseUp={playback.handleSeekEnd}
                onMouseLeave={(e) => {
                  playback.handleSeekEnd(e)
                  timelinePreview.handleTimelineHoverEnd()
                }}
                onTouchStart={playback.handleSeekTouchStart}
                onTouchMove={playback.handleSeekTouchMove}
                onTouchEnd={playback.handleSeekTouchEnd}
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
                    className="video-timeline-progress"
                    style={{ width: `${playback.duration ? (playback.currentTime / playback.duration) * 100 : 0}%` }}
                  />
                  <div
                    className="video-timeline-playhead"
                    style={{ left: `${playback.duration ? (playback.currentTime / playback.duration) * 100 : 0}%` }}
                  />
                </div>
              </div>
              <span className="video-time">{formatTime(playback.duration, true)}</span>
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
                >
                  <svg viewBox="0 0 12 8" fill="currentColor">
                    <path d="M1.41 7.41L6 2.83l4.59 4.58L12 6 6 0 0 6l1.41 1.41z"/>
                  </svg>
                </button>
              </div>
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
              <button
                className="video-control-btn quality-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  setShowQualitySelector(!showQualitySelector)
                }}
                title="Quality"
              >
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zm-5.04-6.71l-2.75 3.54h2.79v2.71h2V13.83h2.79l-2.75-3.54zM7 9h2v2H7z"/>
                </svg>
              </button>
              <div className="video-volume-container">
                <button
                  className="video-control-btn video-mute-btn"
                  onClick={playback.toggleMute}
                  title={playback.isMuted ? 'Unmute (M)' : 'Mute (M)'}
                >
                  {playback.isMuted || playback.volume === 0 ? (
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
                  value={playback.isMuted ? 0 : playback.volume}
                  onChange={playback.handleVolumeChange}
                  title={`Volume: ${Math.round((playback.isMuted ? 0 : playback.volume) * 100)}%`}
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
              <div className="cast-overlay" onClick={(e) => e.stopPropagation()}>
                <div className="cast-overlay-header">
                  <svg viewBox="0 0 24 24" fill="currentColor" className="cast-overlay-icon">
                    <path d="M1 18v3h3c0-1.66-1.34-3-3-3zm0-4v2c2.76 0 5 2.24 5 5h2c0-3.87-3.13-7-7-7zm18-7H5v1.63c3.96 1.28 7.09 4.41 8.37 8.37H19V7zM1 10v2c4.97 0 9 4.03 9 9h2c0-6.08-4.93-11-11-11zm20-7H3c-1.1 0-2 .9-2 2v3h2V5h18v14h-7v2h7c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2z"/>
                  </svg>
                  <span>Casting to TV</span>
                </div>
                <div className="cast-overlay-controls">
                  <button
                    className="cast-control-btn"
                    onClick={() => casting.castStatus?.state === 'playing' ? casting.castPause() : casting.castResume()}
                  >
                    {casting.castStatus?.state === 'playing' ? (
                      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/></svg>
                    ) : (
                      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                    )}
                  </button>
                </div>
                {/* Cast timeline */}
                {casting.castStatus?.duration > 0 && (
                  <div className="cast-timeline-row">
                    <span className="cast-time">{formatTime(casting.castStatus.current_time || 0)}</span>
                    <div
                      className="cast-timeline"
                      onClick={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect()
                        const pct = (e.clientX - rect.left) / rect.width
                        casting.castSeek(pct * casting.castStatus.duration)
                      }}
                    >
                      <div className="cast-timeline-track">
                        <div
                          className="cast-timeline-progress"
                          style={{ width: `${(casting.castStatus.current_time / casting.castStatus.duration) * 100}%` }}
                        />
                      </div>
                    </div>
                    <span className="cast-time">{formatTime(casting.castStatus.duration)}</span>
                  </div>
                )}
                {/* Cast volume */}
                <div className="cast-volume-row">
                  <svg viewBox="0 0 24 24" fill="currentColor" className="cast-volume-icon">
                    <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
                  </svg>
                  <input
                    type="range"
                    className="cast-volume-slider"
                    min="0"
                    max="1"
                    step="0.05"
                    value={casting.castStatus?.volume ?? 1}
                    onChange={(e) => casting.castVolume(parseFloat(e.target.value))}
                  />
                </div>
                <button className="cast-stop-btn" onClick={casting.stopCasting}>
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
              <div className="resume-toast" onClick={(e) => e.stopPropagation()}>
                <span>Resume from {formatTime(resumePosition.position)}?</span>
                <div className="resume-toast-actions">
                  <button className="resume-toast-btn" onClick={() => {
                    playback.seekVideo(resumePosition.position - playback.currentTime)
                    setResumePosition(null)
                  }}>Resume</button>
                  <button className="resume-toast-btn dismiss" onClick={() => setResumePosition(null)}>Start Over</button>
                </div>
              </div>
            )}
            {/* Auto-advance countdown overlay */}
            {autoAdvance.countdown !== null && (
              <div className="auto-advance-overlay" onClick={(e) => e.stopPropagation()}>
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
        ) : (
          <img
            key={previewUrl ? `${image.id}-preview` : image.id}
            ref={mediaRef}
            src={getMediaUrl(previewUrl || image.url)}
            alt=""
            className="lightbox-media"
            style={{ ...(previewUrl ? {} : getFilterStyle()), ...zoomPan.getZoomTransform() }}
            onContextMenu={(e) => {
              e.preventDefault()
              const desktopAPI = getDesktopAPI()
              if (desktopAPI?.showImageContextMenu) {
                desktopAPI.showImageContextMenu({
                  imageUrl: getMediaUrl(image.url),
                  filePath: image.file_path,
                  isVideo: false
                })
              }
            }}
          />
        )}
      </div>

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
      {isVideoFile && (
        <SVPSideMenu
          isOpen={showSVPMenu}
          onClose={async () => {
            setShowSVPMenu(false)
            // Reload SVP config in case it was changed in the menu
            try {
              const newConfig = await getSVPConfig()
              const configChanged = JSON.stringify(newConfig) !== JSON.stringify(streaming.svpConfig)
              streaming.setSvpConfig(newConfig)

              // If config changed, restart the stream with new settings
              if (configChanged) {
                // Stop current stream
                if (streaming.svpStreamUrl) {
                  await stopSVPStream()
                  streaming.setSvpStreamUrl(null)
                }
                // Clear error (e.g., "fps already at target" with old target)
                streaming.setSvpError(null)
                streaming.setSvpLoading(false)
                streaming.svpStartingRef.current = false

                // Start new stream if enabled
                if (newConfig.enabled && image && isVideo(image.filename)) {
                  // Small delay to ensure state is cleared
                  setTimeout(() => {
                    streaming.startSVPStreamRef.current?.()
                  }, 100)
                }
              }
            } catch (err) {
              console.error('Failed to reload SVP config:', err)
            }
          }}
          image={image}
        />
      )}

      {/* Quality selector */}
      {isVideoFile && (
        <QualitySelector
          isOpen={showQualitySelector}
          onClose={() => setShowQualitySelector(false)}
          currentQuality={currentQuality}
          onQualityChange={handleQualityChange}
          sourceResolution={streaming.sourceResolution}
        />
      )}

      {/* Subtitle language/task menu */}
      {showSubtitleMenu && (
        <>
          <div className="subtitle-menu-popup" onClick={(e) => e.stopPropagation()}>
            <div className="subtitle-menu-header">Subtitles</div>
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
    </div>
  )
}

export default Lightbox
