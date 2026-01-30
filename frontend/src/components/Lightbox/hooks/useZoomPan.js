import { useCallback, useState, useRef, useEffect } from 'react'
import { isVideo } from '../utils/helpers'

/**
 * Hook for managing zoom and pan gestures for images
 */
export function useZoomPan(mediaRef, containerRef, resetHideTimer, image) {
  // Zoom state
  const [zoom, setZoom] = useState({ scale: 1, x: 0, y: 0 })
  const zoomRef = useRef(zoom) // Ref to always have current zoom in callbacks
  zoomRef.current = zoom
  const isDragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const lastPinchDistance = useRef(null)

  // Touch/swipe handling
  const touchStartX = useRef(null)
  const touchStartY = useRef(null)
  const touchMoved = useRef(false)
  const touchHandled = useRef(false)

  // Reset zoom when image changes
  const resetZoom = useCallback(() => {
    setZoom({ scale: 1, x: 0, y: 0 })
  }, [])

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
  }, [mediaRef, containerRef])

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
  }, [resetHideTimer, image?.original_filename, containerRef])

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

  // Touch start handler
  const handleTouchStart = useCallback((e) => {
    touchStartX.current = e.touches[0].clientX
    touchStartY.current = e.touches[0].clientY
    touchMoved.current = false
    touchHandled.current = false
    resetHideTimer() // Show UI on touch
  }, [resetHideTimer])

  // Touch move handler (for swipe detection)
  const handleTouchMove = useCallback((e) => {
    if (touchStartX.current === null) return
    const deltaX = Math.abs(e.touches[0].clientX - touchStartX.current)
    const deltaY = Math.abs(e.touches[0].clientY - touchStartY.current)
    // Mark as moved if finger moved more than 10px
    if (deltaX > 10 || deltaY > 10) {
      touchMoved.current = true
    }
  }, [])

  // Touch end handler (for sidebar toggle)
  const handleTouchEnd = useCallback((e, onSidebarHover, sidebarOpen) => {
    if (touchStartX.current === null) return

    const touchEndX = e.changedTouches[0].clientX
    const deltaX = touchEndX - touchStartX.current
    const deltaY = e.changedTouches[0].clientY - touchStartY.current

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
  }, [zoom.scale])

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
  }, [zoom, image?.original_filename, containerRef])

  const handleTouchEndZoom = useCallback(() => {
    lastPinchDistance.current = null
  }, [])

  // Attach wheel listener with passive: false
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    container.addEventListener('wheel', handleWheel, { passive: false })
    return () => container.removeEventListener('wheel', handleWheel)
  }, [handleWheel, containerRef])

  // Get transform style for zoomed media
  const getZoomTransform = useCallback(() => {
    if (zoom.scale === 1 && zoom.x === 0 && zoom.y === 0) {
      return {}
    }
    return {
      transform: `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})`,
      cursor: zoom.scale > 1 ? 'grab' : 'default'
    }
  }, [zoom])

  return {
    zoom,
    setZoom,
    resetZoom,
    isZoomDefault,
    calculateFillScale,
    handleMouseDown,
    handleMouseMoveDrag,
    handleMouseUp,
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd,
    handleTouchMoveZoom,
    handleTouchEndZoom,
    getZoomTransform,
    touchMoved,
    touchHandled
  }
}
