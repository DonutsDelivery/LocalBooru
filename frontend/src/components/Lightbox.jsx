import { useEffect, useCallback, useState, useRef } from 'react'
import { getMediaUrl } from '../api'
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
  const hideUITimeout = useRef(null)

  // Image adjustment state (Gwenview-style ranges)
  // All sliders: -100 to +100 (0 = no change)
  const [showAdjustments, setShowAdjustments] = useState(false)
  const [adjustments, setAdjustments] = useState({ brightness: 0, contrast: 0, gamma: 0 })
  const [applyingAdjustments, setApplyingAdjustments] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [generatingPreview, setGeneratingPreview] = useState(false)

  // Zoom state
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 })
  const zoomRef = useRef(zoom) // Ref to always have current zoom in callbacks
  zoomRef.current = zoom
  const mediaRef = useRef(null)
  const containerRef = useRef(null)
  const isDragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const lastPinchDistance = useRef(null)

  const image = images[currentIndex]

  // Reset adjustments, preview, and zoom when changing images
  useEffect(() => {
    setAdjustments({ brightness: 0, contrast: 0, gamma: 0 })
    setShowAdjustments(false)
    setPreviewUrl(null)
    setZoom({ scale: 1, x: 0, y: 0 })
  }, [image?.id])

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

  // Double-click handler: zoom to fill or reset
  const handleDoubleClick = useCallback((e) => {
    // Don't zoom if clicking on interactive elements
    if (e.target.closest('.lightbox-toolbar, .lightbox-counter, .lightbox-confirm-overlay, .lightbox-adjustments')) return

    resetHideTimer()

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
  }, [isZoomDefault, calculateFillScale, resetHideTimer])

  // Wheel zoom at cursor position
  const handleWheel = useCallback((e) => {
    // Don't zoom if over interactive elements
    if (e.target.closest('.lightbox-toolbar, .lightbox-adjustments, .lightbox-confirm-overlay')) return

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
  }, [resetHideTimer]) // Removed zoom from dependencies - using ref instead

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

  // Pinch-to-zoom for touch
  const handleTouchMoveZoom = useCallback((e) => {
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
  }, [zoom])

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
      await deleteImage(image.id, true) // true = delete from filesystem
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

      // Ctrl+C to copy image
      if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        e.preventDefault()
        handleCopyImage()
        return
      }

      switch (e.key) {
        case 'Escape':
          if (showDeleteConfirm) {
            setShowDeleteConfirm(false)
          } else {
            onClose()
          }
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
        case 'Delete':
          setShowDeleteConfirm(true)
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    document.body.style.overflow = 'hidden'

    return () => {
      window.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = ''
    }
  }, [onNav, onClose, handleToggleFavorite, handleCopyImage, showDeleteConfirm])

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

    // Don't navigate if clicking on interactive elements
    if (e.target.closest('.lightbox-toolbar, .lightbox-counter, .lightbox-confirm-overlay, video')) return

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

  const isVideoFile = isVideo(image.filename)
  const fileStatus = image.file_status || 'available'
  const isUnavailable = fileStatus !== 'available'

  return (
    <div
      className={`lightbox ${!showUI ? 'ui-hidden' : ''} ${zoom.scale > 1 ? 'zoomed' : ''}`}
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
          <video
            key={image.id}
            ref={mediaRef}
            src={getMediaUrl(image.url)}
            controls
            autoPlay
            loop
            className="lightbox-media"
            style={getZoomTransform()}
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
                className="lightbox-confirm-cancel"
                onClick={() => setShowDeleteConfirm(false)}
                disabled={processing}
              >
                Cancel
              </button>
              <button
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
    </div>
  )
}

export default Lightbox
