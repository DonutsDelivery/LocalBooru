import { useEffect, useCallback, useState, useRef } from 'react'
import Hls from 'hls.js'
import { getMediaUrl, getOpticalFlowConfig, playVideoInterpolated, stopInterpolatedStream, getSVPConfig, playVideoSVP, stopSVPStream, playVideoTranscode, stopTranscodeStream } from '../api'
import SVPSideMenu from './SVPSideMenu'
import QualitySelector from './QualitySelector'
import './Lightbox.css'

// Check if filename is a video
const isVideo = (filename) => {
  if (!filename) return false
  const ext = filename.toLowerCase().split('.').pop()
  return ['webm', 'mp4', 'mov'].includes(ext)
}

function Lightbox({ images, currentIndex, total, onClose, onNav, onTagClick, onImageUpdate, onSidebarHover, sidebarOpen, onDelete }) {
  const [processing, setProcessing] = useState(false)
  const [isFavorited, setIsFavorited] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [copyFeedback, setCopyFeedback] = useState(null) // 'success' | 'error' | null
  const [showUI, setShowUI] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const hideUITimeout = useRef(null)
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
  const transcodeHlsRef = useRef(null)

  // SVP side menu state
  const [showSVPMenu, setShowSVPMenu] = useState(false)

  // Quality selector state
  const [showQualitySelector, setShowQualitySelector] = useState(false)
  const [currentQuality, setCurrentQuality] = useState(() => {
    // Load quality preference from localStorage on init
    return localStorage.getItem('video_quality_preference') || 'original'
  })
  const [sourceResolution, setSourceResolution] = useState(null)

  // Video player state
  const [isPlaying, setIsPlaying] = useState(true) // Start autoplaying
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isSeeking, setIsSeeking] = useState(false)
  const [videoDisplayMode, setVideoDisplayMode] = useState('fit') // 'fit' | 'fill' | 'original'
  const [videoNaturalSize, setVideoNaturalSize] = useState({ width: 0, height: 0 })
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)

  // Zoom state
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 })
  const zoomRef = useRef(zoom) // Ref to always have current zoom in callbacks
  zoomRef.current = zoom
  const mediaRef = useRef(null)
  const timelineRef = useRef(null)
  const containerRef = useRef(null)
  const isDragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const lastPinchDistance = useRef(null)

  const image = images[currentIndex]

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
  useEffect(() => {
    setAdjustments({ brightness: 0, contrast: 0, gamma: 0 })
    setShowAdjustments(false)
    setPreviewUrl(null)
    setZoom({ scale: 1, x: 0, y: 0 })
    setIsPlaying(true)
    setCurrentTime(0)
    setDuration(0)
    setIsSeeking(false)
    setVideoDisplayMode('fit')
    setVideoNaturalSize({ width: 0, height: 0 })
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
    // Cleanup HLS instances
    if (hlsRef.current) {
      hlsRef.current.destroy()
      hlsRef.current = null
    }
    if (svpHlsRef.current) {
      svpHlsRef.current.destroy()
      svpHlsRef.current = null
    }
    // Stop backend streams when navigating to a non-video image
    // (For video-to-video navigation, startSVPStream() handles stopping the old stream)
    if (image && !isVideo(image.filename)) {
      stopSVPStream().catch(() => {})
      stopInterpolatedStream().catch(() => {})
    }
  }, [image?.id])

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


  // Auto-hide UI after inactivity
  const resetHideTimer = useCallback(() => {
    setShowUI(true)
    if (hideUITimeout.current) {
      clearTimeout(hideUITimeout.current)
    }
    hideUITimeout.current = setTimeout(() => {
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

  // Fullscreen toggle handler
  const handleToggleFullscreen = useCallback(async () => {
    if (!containerRef.current) return

    try {
      if (!document.fullscreenElement) {
        await containerRef.current.requestFullscreen()
      } else {
        await document.exitFullscreen()
      }
    } catch (err) {
      console.error('Fullscreen error:', err)
    }
  }, [])

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  // Track favorite state for current image
  useEffect(() => {
    if (image) {
      setIsFavorited(image.is_favorite || false)
    }
  }, [image?.id, image?.is_favorite])

  // Touch/swipe handling
  const touchStartX = useRef(null)
  const touchStartY = useRef(null)
  const touchMoved = useRef(false)
  const touchHandled = useRef(false)

  const handleTouchStart = useCallback((e) => {
    touchStartX.current = e.touches[0].clientX
    touchStartY.current = e.touches[0].clientY
    touchMoved.current = false
    touchHandled.current = false
    resetHideTimer() // Show UI on touch
  }, [resetHideTimer])

  const handleTouchMove = useCallback((e) => {
    if (touchStartX.current === null) return
    const deltaX = Math.abs(e.touches[0].clientX - touchStartX.current)
    const deltaY = Math.abs(e.touches[0].clientY - touchStartY.current)
    // Mark as moved if finger moved more than 10px
    if (deltaX > 10 || deltaY > 10) {
      touchMoved.current = true
    }
  }, [])

  const handleTouchEnd = useCallback((e) => {
    if (touchStartX.current === null) return

    const touchEndX = e.changedTouches[0].clientX
    const touchEndY = e.changedTouches[0].clientY
    const deltaX = touchEndX - touchStartX.current
    const deltaY = touchEndY - touchStartY.current

    // Only handle swipe for sidebar when not zoomed
    if (zoom.scale <= 1) {
      // Require a minimum swipe of 50px and horizontal movement must be dominant
      if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 50) {
        touchHandled.current = true

        // Swipe right = open sidebar, swipe left = close sidebar
        // Image navigation is done by tapping left/right sides
        if (deltaX > 0 && !sidebarOpen) {
          onSidebarHover && onSidebarHover(true)
        } else if (deltaX < 0 && sidebarOpen) {
          onSidebarHover && onSidebarHover(false)
        }
      }
    }

    touchStartX.current = null
    touchStartY.current = null
  }, [onNav, onSidebarHover, sidebarOpen, zoom.scale])

  // Helper to check if zoom is at default
  const isZoomDefault = useCallback(() => {
    return zoom.scale === 1 && zoom.x === 0 && zoom.y === 0
  }, [zoom])

  // Calculate scale to fill the viewport with the image
  const calculateFillScale = useCallback(() => {
    if (!mediaRef.current || !containerRef.current) return null

    const media = mediaRef.current
    const container = containerRef.current
    const containerRect = container.getBoundingClientRect()

    // Get natural dimensions (for images) or video dimensions
    const naturalWidth = media.naturalWidth || media.videoWidth || media.offsetWidth
    const naturalHeight = media.naturalHeight || media.videoHeight || media.offsetHeight

    if (!naturalWidth || !naturalHeight || !containerRect.width || !containerRect.height) return null

    const containerWidth = containerRect.width
    const containerHeight = containerRect.height
    const containerAspect = containerWidth / containerHeight
    const imageAspect = naturalWidth / naturalHeight

    // With object-fit: contain, calculate displayed size at scale 1
    let displayedWidth, displayedHeight
    if (imageAspect > containerAspect) {
      // Width fills container, height is letterboxed
      displayedWidth = containerWidth
      displayedHeight = containerWidth / imageAspect
    } else {
      // Height fills container, width is letterboxed
      displayedHeight = containerHeight
      displayedWidth = containerHeight * imageAspect
    }

    // Calculate scale to make both dimensions fill container
    const scaleForWidth = containerWidth / displayedWidth
    const scaleForHeight = containerHeight / displayedHeight

    // Return the larger scale (one will be 1.0, the other > 1.0)
    return Math.max(scaleForWidth, scaleForHeight)
  }, [])

  // Restart SVP stream from a specific position (for seeking beyond buffered content)
  const restartSVPFromPosition = useCallback(async (targetTime) => {
    if (!image || !svpConfig?.enabled) return

    console.log(`[SVP] Restarting stream from ${targetTime.toFixed(1)}s`)

    // Show loading indicator
    setSvpLoading(true)
    setSvpPendingSeek(null)
    setCurrentTime(targetTime)

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
        setSvpStreamUrl(result.stream_url)
        setSvpStartOffset(targetTime)  // Track offset for timeline display
        if (result.duration) {
          setSvpTotalDuration(result.duration)
        }
      } else {
        setSvpError(result.error || 'Failed to restart SVP stream')
        setSvpLoading(false)
      }
    } catch (err) {
      console.error('SVP restart error:', err)
      setSvpError(err.message || 'Failed to restart SVP stream')
      setSvpLoading(false)
    }
  }, [image, svpConfig])

  // Seek forward/backward
  const seekVideo = useCallback((seconds) => {
    if (!mediaRef.current) return

    // For SVP streams, currentTime is in HLS time, need to convert to absolute video time
    const currentAbsoluteTime = svpStreamUrl ? mediaRef.current.currentTime + svpStartOffset : mediaRef.current.currentTime
    const newTime = Math.max(0, Math.min(duration, currentAbsoluteTime + seconds))

    // For SVP streams, check if we need to restart from a new position
    if (svpStreamUrl) {
      const bufferedEnd = svpStartOffset + svpBufferedDuration
      const bufferedStart = svpStartOffset

      if (newTime < bufferedStart - 1 || newTime > bufferedEnd + 2) {
        restartSVPFromPosition(newTime)
        return
      }

      // Seek within current stream
      const hlsTime = newTime - svpStartOffset
      setSvpPendingSeek(null)
      mediaRef.current.currentTime = Math.max(0, hlsTime)
      setCurrentTime(newTime)
      return
    }

    // Normal video seek
    setSvpPendingSeek(null)
    mediaRef.current.currentTime = newTime
    setCurrentTime(newTime)
  }, [duration, svpStreamUrl, svpBufferedDuration, svpStartOffset, restartSVPFromPosition])

  // Double-click handler: zoom to fill or reset (images), toggle display mode (videos)
  const handleDoubleClick = useCallback((e) => {
    // Don't zoom if clicking on interactive elements
    if (e.target.closest('.lightbox-toolbar, .lightbox-counter, .lightbox-confirm-overlay, .lightbox-adjustments, .lightbox-video-controls')) return

    resetHideTimer()

    // Special handling for videos - zone-based actions
    if (isVideo(image?.original_filename)) {
      const rect = e.currentTarget.getBoundingClientRect()
      const clickX = e.clientX - rect.left
      const width = rect.width

      if (clickX < width * 0.4) {
        // Left 40%: skip back 10 seconds
        seekVideo(-10)
      } else if (clickX > width * 0.6) {
        // Right 40%: skip forward 10 seconds
        seekVideo(10)
      } else {
        // Center 20%: toggle display mode (fit/original)
        setVideoDisplayMode(videoDisplayMode === 'fit' ? 'original' : 'fit')
      }
      return
    }

    // Image zoom behavior (unchanged)
    if (isZoomDefault()) {
      // Zoom to fill at click position
      const fillScale = calculateFillScale()
      if (fillScale === null) {
        // Image not ready, ignore
        return
      }
      if (fillScale <= 1.05) {
        // Image already fills or nearly fills, zoom to 2x instead
        setZoom({ scale: 2, x: 0, y: 0 })
      } else {
        setZoom({ scale: fillScale, x: 0, y: 0 })
      }
    } else {
      // Reset zoom
      setZoom({ scale: 1, x: 0, y: 0 })
    }
  }, [isZoomDefault, calculateFillScale, resetHideTimer, image?.original_filename, videoDisplayMode, videoNaturalSize, seekVideo])

  // Wheel zoom at cursor position (disabled for videos)
  const handleWheel = useCallback((e) => {
    // Don't zoom if over interactive elements
    if (e.target.closest('.lightbox-toolbar, .lightbox-adjustments, .lightbox-confirm-overlay')) return

    // Disable wheel zoom for videos
    if (isVideo(image?.original_filename)) {
      e.preventDefault()
      return
    }

    e.preventDefault()
    resetHideTimer()

    // Use ref to always get current zoom value (avoids stale closure)
    const currentZoom = zoomRef.current

    // Normalize delta across browsers and input devices
    // Use sign only, apply consistent zoom step
    const direction = e.deltaY < 0 ? 1 : -1
    const zoomStep = 0.05 // 5% zoom per scroll step
    const newScale = Math.min(Math.max(currentZoom.scale * (1 + direction * zoomStep), 1), 10)

    if (newScale === 1) {
      // Reset to default when zoomed out fully
      setZoom({ scale: 1, x: 0, y: 0 })
      return
    }

    // Zoom toward cursor position
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect()
      const cursorX = e.clientX - rect.left - rect.width / 2
      const cursorY = e.clientY - rect.top - rect.height / 2

      const scaleFactor = newScale / currentZoom.scale
      const newX = cursorX - (cursorX - currentZoom.x) * scaleFactor
      const newY = cursorY - (cursorY - currentZoom.y) * scaleFactor

      setZoom({ scale: newScale, x: newX, y: newY })
    } else {
      setZoom({ scale: newScale, x: currentZoom.x, y: currentZoom.y })
    }
  }, [resetHideTimer, image?.original_filename]) // Removed zoom from dependencies - using ref instead

  // Mouse drag for panning when zoomed
  const handleMouseDown = useCallback((e) => {
    if (zoom.scale <= 1) return
    if (e.target.closest('.lightbox-toolbar, .lightbox-adjustments, .lightbox-confirm-overlay')) return

    isDragging.current = true
    dragStart.current = { x: e.clientX - zoom.x, y: e.clientY - zoom.y }
    e.preventDefault()
  }, [zoom])

  const handleMouseMoveDrag = useCallback((e) => {
    if (!isDragging.current) return

    const newX = e.clientX - dragStart.current.x
    const newY = e.clientY - dragStart.current.y
    setZoom(prev => ({ ...prev, x: newX, y: newY }))
  }, [])

  const handleMouseUp = useCallback(() => {
    isDragging.current = false
  }, [])

  // Pinch-to-zoom for touch (disabled for videos)
  const handleTouchMoveZoom = useCallback((e) => {
    // Disable pinch zoom for videos
    if (isVideo(image?.original_filename)) return

    if (e.touches.length === 2) {
      // Pinch zoom
      const touch1 = e.touches[0]
      const touch2 = e.touches[1]
      const distance = Math.hypot(touch2.clientX - touch1.clientX, touch2.clientY - touch1.clientY)

      if (lastPinchDistance.current !== null) {
        const delta = (distance - lastPinchDistance.current) * 0.01
        const newScale = Math.min(Math.max(zoom.scale * (1 + delta), 1), 10)

        if (newScale === 1) {
          setZoom({ scale: 1, x: 0, y: 0 })
        } else {
          // Zoom toward center of pinch
          if (containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect()
            const centerX = (touch1.clientX + touch2.clientX) / 2 - rect.left - rect.width / 2
            const centerY = (touch1.clientY + touch2.clientY) / 2 - rect.top - rect.height / 2

            const scaleFactor = newScale / zoom.scale
            const newX = centerX - (centerX - zoom.x) * scaleFactor
            const newY = centerY - (centerY - zoom.y) * scaleFactor

            setZoom({ scale: newScale, x: newX, y: newY })
          } else {
            setZoom(prev => ({ ...prev, scale: newScale }))
          }
        }
      }

      lastPinchDistance.current = distance
      touchHandled.current = true
      e.preventDefault()
    } else if (e.touches.length === 1 && zoom.scale > 1) {
      // Single finger pan when zoomed
      if (touchStartX.current !== null) {
        const deltaX = e.touches[0].clientX - touchStartX.current
        const deltaY = e.touches[0].clientY - touchStartY.current

        if (Math.abs(deltaX) > 5 || Math.abs(deltaY) > 5) {
          setZoom(prev => ({
            ...prev,
            x: prev.x + deltaX,
            y: prev.y + deltaY
          }))
          touchStartX.current = e.touches[0].clientX
          touchStartY.current = e.touches[0].clientY
          touchMoved.current = true
        }
      }
    }
  }, [zoom, image?.original_filename])

  const handleTouchEndZoom = useCallback(() => {
    lastPinchDistance.current = null
  }, [])

  // Attach wheel listener with passive: false
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    container.addEventListener('wheel', handleWheel, { passive: false })
    return () => container.removeEventListener('wheel', handleWheel)
  }, [handleWheel])

  // Toggle favorite
  const handleToggleFavorite = useCallback(async () => {
    if (processing || !image) return
    setProcessing(true)

    const wasActive = isFavorited
    setIsFavorited(!wasActive)

    try {
      const { toggleFavorite } = await import('../api')
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
      const { deleteImage } = await import('../api')
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
      // Use Electron API if available
      if (window.electronAPI?.copyImageToClipboard) {
        const result = await window.electronAPI.copyImageToClipboard(getMediaUrl(image.url))
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

  // Start optical flow interpolation for video (called automatically when enabled)
  const startInterpolatedStream = useCallback(async () => {
    if (!image || !opticalFlowConfig?.enabled || !isVideo(image.filename)) return
    if (opticalFlowStreamUrl || opticalFlowLoading) return // Already active or starting

    setOpticalFlowLoading(true)
    setOpticalFlowError(null)

    try {
      const result = await playVideoInterpolated(image.file_path)

      if (result.success && result.stream_url) {
        setOpticalFlowStreamUrl(result.stream_url)
      } else {
        setOpticalFlowError(result.error || 'Failed to start interpolated playback')
      }
    } catch (err) {
      console.error('Optical flow error:', err)
      setOpticalFlowError(err.message || 'Failed to start interpolated playback')
    }

    setOpticalFlowLoading(false)
  }, [image, opticalFlowConfig, opticalFlowStreamUrl, opticalFlowLoading])

  // Refs to access latest callbacks without adding them to effect dependencies
  const startSVPStreamRef = useRef(null)
  const startInterpolatedStreamRef = useRef(null)

  // Auto-start interpolated stream when video opens
  // Priority: SVP (if enabled) > Optical Flow (if enabled)
  useEffect(() => {
    if (image && isVideo(image.filename)) {
      console.log('[SVP Auto-start] Checking...', {
        enabled: svpConfig?.enabled,
        svpConfig: svpConfig
      })
      // Prefer SVP if enabled
      if (svpConfig?.enabled) {
        console.log('[SVP Auto-start] Starting SVP stream...')
        startSVPStreamRef.current()
      }
      // Fall back to optical flow if enabled
      else if (opticalFlowConfig?.enabled) {
        startInterpolatedStreamRef.current()
      }
    }
  }, [image?.id, svpConfig?.enabled, opticalFlowConfig?.enabled])

  // Setup HLS player when optical flow stream is active
  useEffect(() => {
    if (!opticalFlowStreamUrl || !mediaRef.current) return

    const video = mediaRef.current

    if (Hls.isSupported()) {
      // Cleanup previous instance
      if (hlsRef.current) {
        hlsRef.current.destroy()
      }

      // Clear video src to prevent dual playback with HLS MediaSource
      video.pause()
      video.removeAttribute('src')
      video.load()

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
  }, [opticalFlowStreamUrl])

  // Setup HLS player when transcode stream is active (fallback when no interpolation)
  useEffect(() => {
    if (!transcodeStreamUrl || !mediaRef.current) return

    const video = mediaRef.current

    if (Hls.isSupported()) {
      // Cleanup previous instance
      if (transcodeHlsRef.current) {
        transcodeHlsRef.current.destroy()
      }

      // Clear video src to prevent dual playback with HLS MediaSource
      video.pause()
      video.removeAttribute('src')
      video.load()

      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
        backBufferLength: 30
      })

      // Use getMediaUrl to handle dev mode (different ports for frontend/backend)
      hls.loadSource(getMediaUrl(transcodeStreamUrl))
      hls.attachMedia(video)

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(() => {})
      })

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          console.error('Transcode HLS fatal error:', data)
          setTranscodeStreamUrl(null)
        }
      })

      transcodeHlsRef.current = hls
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari/iOS native HLS support
      video.src = getMediaUrl(transcodeStreamUrl)
      video.addEventListener('loadedmetadata', () => {
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
  }, [transcodeStreamUrl])

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
      // Stop backend streams on unmount
      stopSVPStream().catch(() => {})
      stopInterpolatedStream().catch(() => {})
      stopTranscodeStream().catch(() => {})
    }
  }, [])

  // Start SVP interpolation for video (called manually via button)
  const startSVPStream = useCallback(async () => {
    console.log('[startSVPStream] Called', { image: image?.id, enabled: svpConfig?.enabled, isVideo: isVideo(image?.filename) })
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

    // Stop any existing optical flow stream
    if (opticalFlowStreamUrl) {
      setOpticalFlowStreamUrl(null)
      await stopInterpolatedStream()
    }

    setSvpLoading(true)
    setSvpError(null)

    console.log('[startSVPStream] Calling API with path:', image.file_path)
    try {
      const result = await playVideoSVP(image.file_path)
      console.log('[startSVPStream] API result:', result)

      if (result.success && result.stream_url) {
        setSvpStreamUrl(result.stream_url)
        setSvpStartOffset(0)  // Starting from beginning
        // Store the known total duration from API for proper timeline display
        if (result.duration) {
          setSvpTotalDuration(result.duration)
        }
      } else {
        setSvpError(result.error || 'Failed to start SVP playback')
      }
    } catch (err) {
      console.error('SVP error:', err)
      setSvpError(err.message || 'Failed to start SVP playback')
      setSvpLoading(false)  // Only set loading false on error here
    } finally {
      svpStartingRef.current = false
    }
    // Note: svpLoading stays true until MANIFEST_PARSED fires in the useEffect
  }, [image, svpConfig, svpStreamUrl, svpLoading, opticalFlowStreamUrl])

  // Update refs after callbacks are defined (used by auto-start effect)
  startSVPStreamRef.current = startSVPStream
  startInterpolatedStreamRef.current = startInterpolatedStream

  // Toggle video play/pause
  const toggleVideoPlay = useCallback(() => {
    if (!mediaRef.current) return
    const video = mediaRef.current
    if (video.paused) {
      video.play().catch(() => {})
    } else {
      video.pause()
    }
  }, [])

  // Handle click anywhere on video to play/pause
  const handleVideoClick = useCallback((e) => {
    if (!isVideo(image?.original_filename)) return
    // Don't toggle play if clicking on controls
    if (e.target.closest('.lightbox-video-controls')) return
    toggleVideoPlay()
    resetHideTimer()
  }, [image?.original_filename, toggleVideoPlay, resetHideTimer])

  // Handle quality change
  const handleQualityChange = useCallback(async (qualityId) => {
    console.log('[Lightbox] Quality change requested:', qualityId)
    setCurrentQuality(qualityId)
    localStorage.setItem('video_quality_preference', qualityId)

    if (!mediaRef.current) {
      console.log('[Lightbox] No mediaRef available')
      return
    }

    try {
      if (qualityId === 'original') {
        console.log('[Lightbox] Switching to original quality')
        // Stop streams and load direct video
        if (svpStreamUrl) {
          await stopSVPStream()
          setSvpStreamUrl(null)
        }
        if (opticalFlowStreamUrl) {
          await stopInterpolatedStream()
          setOpticalFlowStreamUrl(null)
        }
        if (transcodeStreamUrl) {
          await stopTranscodeStream()
          setTranscodeStreamUrl(null)
        }

        if (mediaRef.current) {
          const currentTime = mediaRef.current.currentTime
          mediaRef.current.src = getMediaUrl(image.url)
          mediaRef.current.currentTime = currentTime
          mediaRef.current.play().catch(() => {})
        }
      } else {
        // Restart stream with new quality
        const currentTime = mediaRef.current?.currentTime || 0
        console.log('[Lightbox] Restarting stream with quality:', qualityId, 'at time:', currentTime)
        console.log('[Lightbox] SVP enabled/ready:', svpConfig?.enabled, svpConfig?.status?.ready)
        console.log('[Lightbox] OpticalFlow enabled:', opticalFlowConfig?.enabled)

        // Check if SVP is currently playing
        if (svpStreamUrl) {
          console.log('[Lightbox] Restarting SVP stream with quality')
          await stopSVPStream()
          const result = await playVideoSVP(image.file_path, currentTime, qualityId)
          console.log('[Lightbox] SVP play result:', result)
          if (result.success) {
            setSvpStreamUrl(result.stream_url)
            if (result.duration) setSvpTotalDuration(result.duration)
          } else {
            console.error('[Lightbox] SVP play failed:', result.error)
          }
        } else if (opticalFlowStreamUrl) {
          console.log('[Lightbox] Restarting OpticalFlow stream with quality')
          await stopInterpolatedStream()
          const result = await playVideoInterpolated(image.file_path, currentTime, qualityId)
          console.log('[Lightbox] OpticalFlow play result:', result)
          if (result.success) {
            setOpticalFlowStreamUrl(result.stream_url)
          } else {
            console.error('[Lightbox] OpticalFlow play failed:', result.error)
          }
        } else {
          // Neither SVP nor OpticalFlow is currently playing
          console.log('[Lightbox] No active stream, starting new one with quality')
          // Try SVP first (if enabled), fall back to OpticalFlow (if enabled), or use transcode
          if (svpConfig?.enabled) {
            // SVP is enabled, try to use it
            console.log('[Lightbox] Starting new SVP stream with quality')
            const result = await playVideoSVP(image.file_path, currentTime, qualityId)
            console.log('[Lightbox] SVP play result:', result)
            if (result.success) {
              setSvpStreamUrl(result.stream_url)
              if (result.duration) setSvpTotalDuration(result.duration)
            } else {
              console.error('[Lightbox] SVP play failed:', result.error)
              alert('Failed to start SVP stream: ' + result.error)
            }
          } else if (opticalFlowConfig?.enabled) {
            // OpticalFlow is enabled, try to use it
            console.log('[Lightbox] Starting new OpticalFlow stream with quality')
            const result = await playVideoInterpolated(image.file_path, currentTime, qualityId)
            console.log('[Lightbox] OpticalFlow play result:', result)
            if (result.success) {
              setOpticalFlowStreamUrl(result.stream_url)
            } else {
              console.error('[Lightbox] OpticalFlow play failed:', result.error)
              alert('Failed to start OpticalFlow stream: ' + result.error)
            }
          } else {
            // Neither SVP nor OpticalFlow enabled, use simple transcode
            console.log('[Lightbox] Using transcode (FFmpeg only) for quality change')
            if (transcodeStreamUrl) {
              await stopTranscodeStream()
              setTranscodeStreamUrl(null)
            }
            const result = await playVideoTranscode(image.file_path, currentTime, qualityId)
            console.log('[Lightbox] Transcode play result:', result)
            if (result.success) {
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
  }, [image?.url, image?.file_path, svpStreamUrl, opticalFlowStreamUrl, svpConfig, opticalFlowConfig])

  // Sync play state with video element events
  const handleVideoPlay = useCallback(() => {
    setIsPlaying(true)
    // Ensure source resolution is set (fallback for HLS streams where it might not be available in onLoadedMetadata)
    if (mediaRef.current && (!sourceResolution || !sourceResolution.width || !sourceResolution.height)) {
      const width = mediaRef.current.videoWidth
      const height = mediaRef.current.videoHeight
      if (width > 0 && height > 0) {
        console.log('[Lightbox] Setting resolution on play:', width, 'x', height)
        setSourceResolution({
          width,
          height
        })
      }
    }
  }, [sourceResolution])

  const handleVideoPause = useCallback(() => {
    setIsPlaying(false)
  }, [])

  // Update current time as video plays
  const handleTimeUpdate = useCallback(() => {
    if (!mediaRef.current || isSeeking) return
    // Don't update time display while waiting for pending seek
    if (svpPendingSeek) return
    // Add offset for SVP streams that started from a seek position
    const actualTime = svpStreamUrl ? mediaRef.current.currentTime + svpStartOffset : mediaRef.current.currentTime
    setCurrentTime(actualTime)
  }, [isSeeking, svpPendingSeek, svpStreamUrl, svpStartOffset])

  // Get duration and natural size when video metadata loads
  const handleLoadedMetadata = useCallback(() => {
    if (!mediaRef.current) return
    // For SVP/HLS streams, use the known total duration from API if available
    // This allows the timeline to show the full video length even while segments are being generated
    if (svpTotalDuration && svpStreamUrl) {
      setDuration(svpTotalDuration)
    } else {
      setDuration(mediaRef.current.duration)
    }

    const width = mediaRef.current.videoWidth
    const height = mediaRef.current.videoHeight

    setVideoNaturalSize({
      width,
      height
    })

    // Store source resolution for quality selector
    // For HLS streams, dimensions might not be available immediately
    if (width > 0 && height > 0) {
      console.log('[Lightbox] Setting source resolution:', width, 'x', height)
      setSourceResolution({
        width,
        height
      })
    } else {
      console.log('[Lightbox] Video dimensions not available yet, will retry on play')
    }
  }, [svpTotalDuration, svpStreamUrl])

  // Handle seeking via timeline
  const handleSeek = useCallback((e) => {
    if (!mediaRef.current || !duration || !timelineRef.current) return
    const rect = timelineRef.current.getBoundingClientRect()
    const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const newTime = percent * duration  // Absolute video time

    // For SVP streams, check if we need to restart from a new position
    if (svpStreamUrl) {
      // Calculate the actual buffered range in absolute video time
      const bufferedEnd = svpStartOffset + svpBufferedDuration
      const bufferedStart = svpStartOffset

      // Need to restart if seeking outside the current buffered range
      if (newTime < bufferedStart - 1 || newTime > bufferedEnd + 2) {
        console.log(`[SVP] Seeking to ${newTime.toFixed(1)}s, buffered range: ${bufferedStart.toFixed(1)}-${bufferedEnd.toFixed(1)}s. Restarting stream...`)
        restartSVPFromPosition(newTime)
        return
      }

      // Seek within current stream (convert to HLS stream time)
      const hlsTime = newTime - svpStartOffset
      setSvpPendingSeek(null)
      mediaRef.current.currentTime = Math.max(0, hlsTime)
      setCurrentTime(newTime)
      return
    }

    // Normal video seek (no SVP)
    setSvpPendingSeek(null)
    mediaRef.current.currentTime = newTime
    setCurrentTime(newTime)
  }, [duration, svpStreamUrl, svpBufferedDuration, svpStartOffset, restartSVPFromPosition])

  const handleSeekStart = useCallback((e) => {
    setIsSeeking(true)
    handleSeek(e)
  }, [handleSeek])

  const handleSeekMove = useCallback((e) => {
    if (!isSeeking) return
    handleSeek(e)
  }, [isSeeking, handleSeek])

  const handleSeekEnd = useCallback(() => {
    setIsSeeking(false)
  }, [])

  // Format time as MM:SS
  const formatTime = (seconds) => {
    if (!seconds || !isFinite(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Handle volume change
  const handleVolumeChange = useCallback((e) => {
    const newVolume = parseFloat(e.target.value)
    setVolume(newVolume)
    if (mediaRef.current) {
      mediaRef.current.volume = newVolume
      setIsMuted(newVolume === 0)
    }
  }, [])

  // Toggle mute
  const toggleMute = useCallback(() => {
    if (!mediaRef.current) return
    if (isMuted) {
      mediaRef.current.muted = false
      setIsMuted(false)
    } else {
      mediaRef.current.muted = true
      setIsMuted(true)
    }
  }, [isMuted])

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

  // Setup HLS player when SVP stream URL is available
  // Keep normal video playing until HLS is ready, then switch
  useEffect(() => {
    if (!svpStreamUrl || !mediaRef.current) return

    const video = mediaRef.current
    let cancelled = false

    if (Hls.isSupported()) {
      // Cleanup previous instance
      if (svpHlsRef.current) {
        svpHlsRef.current.destroy()
      }

      // Clear video src to prevent dual playback with HLS MediaSource
      video.pause()
      video.removeAttribute('src')
      video.load()

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
        video.play().catch(() => {})
        setSvpLoading(false)
      })
    } else {
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
  }, [svpStreamUrl])

  // Generate preview of adjustments
  const handleGeneratePreview = useCallback(async () => {
    if (!image || generatingPreview) return

    // Check if any adjustments were made
    if (adjustments.brightness === 0 && adjustments.contrast === 0 && adjustments.gamma === 0) {
      return
    }

    setGeneratingPreview(true)
    try {
      const { previewImageAdjustments } = await import('../api')
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
      const { discardImagePreview } = await import('../api')
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
      const { applyImageAdjustments, discardImagePreview } = await import('../api')
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
          handleToggleFavorite()
          break
        case ' ':
          // Space toggles video play/pause
          if (isVideo(image?.original_filename) && mediaRef.current) {
            e.preventDefault()
            toggleVideoPlay()
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
  }, [onNav, onClose, handleToggleFavorite, handleCopyImage, handleDelete, showDeleteConfirm, toggleVideoPlay, image?.original_filename])

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
    if (touchMoved.current || touchHandled.current) {
      touchMoved.current = false
      touchHandled.current = false
      return
    }

    // Don't navigate if zoomed in
    if (zoom.scale > 1) return

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

  // Get transform style for zoomed media
  const getZoomTransform = () => {
    if (zoom.scale === 1 && zoom.x === 0 && zoom.y === 0) {
      return {}
    }
    return {
      transform: `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})`,
      cursor: zoom.scale > 1 ? 'grab' : 'default'
    }
  }

  if (!image) return null

  const isVideoFile = isVideo(image.original_filename)
  const fileStatus = image.file_status || 'available'
  const isUnavailable = fileStatus !== 'available'

  return (
    <div
      className={`lightbox ${!showUI ? 'ui-hidden' : ''} ${zoom.scale > 1 ? 'zoomed' : ''} ${isFullscreen ? 'fullscreen' : ''}`}
      onClick={handleNavClick}
      onDoubleClick={handleDoubleClick}
      onMouseMove={(e) => { handleMouseMove(); handleMouseMoveDrag(e); }}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchStart={handleTouchStart}
      onTouchMove={(e) => { handleTouchMove(e); handleTouchMoveZoom(e); }}
      onTouchEnd={(e) => { handleTouchEnd(e); handleTouchEndZoom(); }}
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
              src={svpConfigLoaded && !svpStreamUrl && !opticalFlowStreamUrl && !transcodeStreamUrl && !svpLoading && !svpConfig?.enabled && !opticalFlowConfig?.enabled ? getMediaUrl(image.url) : undefined}
              autoPlay
              playsInline
              loop
              className={`lightbox-media video-display-${videoDisplayMode} ${svpStreamUrl ? 'svp-streaming' : opticalFlowStreamUrl ? 'interpolated-streaming' : transcodeStreamUrl ? 'transcode-streaming' : ''}`}
              style={getZoomTransform()}
              onClick={handleVideoClick}
              onPlay={handleVideoPlay}
              onPause={handleVideoPause}
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onCanPlay={(e) => {
                // Ensure video plays even if autoPlay is blocked
                e.target.play().catch(() => {})
              }}
              onContextMenu={(e) => {
                e.preventDefault()
                if (window.electronAPI?.showImageContextMenu) {
                  window.electronAPI.showImageContextMenu({
                    imageUrl: getMediaUrl(image.url),
                    filePath: image.file_path,
                    isVideo: true
                  })
                }
              }}
            />
            {/* Custom video controls */}
            <div
              className="lightbox-video-controls"
              onClick={(e) => e.stopPropagation()}
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
                  onClick={toggleVideoPlay}
                  title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
                >
                  {isPlaying ? (
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
              <span className="video-time">{formatTime(currentTime)}</span>
              <div
                ref={timelineRef}
                className="video-timeline"
                onMouseDown={handleSeekStart}
                onMouseMove={handleSeekMove}
                onMouseUp={handleSeekEnd}
                onMouseLeave={handleSeekEnd}
              >
                <div className="video-timeline-track">
                  {/* Buffer indicator for SVP streams - shows how much is available */}
                  {/* Buffer indicator for SVP streams - shows the buffered range */}
                  {svpStreamUrl && svpBufferedDuration > 0 && duration > 0 && (
                    <div
                      className="video-timeline-buffer"
                      style={{
                        left: `${(svpStartOffset / duration) * 100}%`,
                        width: `${(svpBufferedDuration / duration) * 100}%`
                      }}
                    />
                  )}
                  <div
                    className="video-timeline-progress"
                    style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
                  />
                  <div
                    className="video-timeline-playhead"
                    style={{ left: `${duration ? (currentTime / duration) * 100 : 0}%` }}
                  />
                </div>
              </div>
              <span className="video-time">{formatTime(duration)}</span>
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
                  onClick={toggleMute}
                  title={isMuted ? 'Unmute' : 'Mute'}
                >
                  {isMuted || volume === 0 ? (
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"/>
                    </svg>
                  ) : volume < 0.5 ? (
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
                  value={isMuted ? 0 : volume}
                  onChange={handleVolumeChange}
                  title={`Volume: ${Math.round((isMuted ? 0 : volume) * 100)}%`}
                />
              </div>
              </div>
            </div>
            {/* Optical flow loading indicator */}
            {opticalFlowLoading && (
              <div className="interpolate-loading">
                <div className="interpolate-loading-spinner" />
                <span>Buffering {opticalFlowConfig?.target_fps || 60} FPS...</span>
              </div>
            )}
            {/* Optical flow streaming indicator */}
            {opticalFlowStreamUrl && !opticalFlowLoading && (
              <div className="interpolate-badge">
                {opticalFlowConfig?.target_fps || 60} FPS
              </div>
            )}
            {/* Optical flow error toast */}
            {opticalFlowError && (
              <div className="interpolate-error-toast">
                {opticalFlowError}
              </div>
            )}
            {/* SVP loading indicator */}
            {svpLoading && (
              <div className="interpolate-loading svp-loading">
                <div className="interpolate-loading-spinner" />
                <span>SVP: Buffering {svpConfig?.target_fps || 60} FPS...</span>
              </div>
            )}
            {/* SVP streaming indicator */}
            {svpStreamUrl && !svpLoading && !svpPendingSeek && (
              <div className="interpolate-badge svp-badge">
                SVP {svpConfig?.target_fps || 60} FPS
              </div>
            )}
            {/* SVP waiting for seek indicator */}
            {svpPendingSeek && (
              <div className="interpolate-loading svp-loading">
                <div className="interpolate-loading-spinner" />
                <span>Buffering to {formatTime(svpPendingSeek)}...</span>
              </div>
            )}
            {/* SVP error toast */}
            {svpError && (
              <div className="interpolate-error-toast svp-error">
                SVP: {svpError}
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
            style={{ ...(previewUrl ? {} : getFilterStyle()), ...getZoomTransform() }}
            onContextMenu={(e) => {
              e.preventDefault()
              if (window.electronAPI?.showImageContextMenu) {
                window.electronAPI.showImageContextMenu({
                  imageUrl: getMediaUrl(image.url),
                  filePath: image.file_path,
                  isVideo: false
                })
              }
            }}
          />
        )}
      </div>

      <div className="lightbox-counter">
        {currentIndex + 1} / {total}
      </div>

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
              const config = await getSVPConfig()
              setSvpConfig(config)
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
          sourceResolution={sourceResolution}
        />
      )}
    </div>
  )
}

export default Lightbox
