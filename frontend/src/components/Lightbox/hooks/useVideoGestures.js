import { useCallback, useRef, useState } from 'react'

/**
 * Handles video tap zones (seek ±10s on left/right thirds, play/pause on center)
 * and drag-to-seek gesture (touch only, horizontal drag with overlay).
 */
export function useVideoGestures(playback, resetHideTimer) {
  // Tap seek indicator: { side: 'left'|'right', amount: number, key: number }
  const [seekIndicator, setSeekIndicator] = useState(null)
  // Drag seek overlay: { amount: number } while dragging
  const [dragSeek, setDragSeek] = useState(null)

  // Touch tracking refs
  const touchStartX = useRef(0)
  const touchStartY = useRef(0)
  const isDragging = useRef(false)
  const wasDragging = useRef(false)
  const dragAmount = useRef(0)
  const indicatorTimer = useRef(null)

  // Handle click/tap on video — zone detection
  const handleVideoClick = useCallback((e) => {
    // Don't toggle play if clicking on controls or overlays
    if (e.target.closest('.lightbox-video-controls, .cast-overlay, .resume-toast, .auto-advance-overlay')) return

    // Skip if a drag just finished (touchend fires click)
    if (wasDragging.current) {
      wasDragging.current = false
      return
    }

    resetHideTimer()

    // Get click position relative to the video container
    const container = e.currentTarget
    const rect = container.getBoundingClientRect()
    const x = e.clientX - rect.left
    const pct = x / rect.width

    if (pct < 0.33) {
      // Left third — seek back 10s
      playback.seekVideo(-10)
      clearTimeout(indicatorTimer.current)
      setSeekIndicator({ side: 'left', amount: -10, key: Date.now() })
      indicatorTimer.current = setTimeout(() => setSeekIndicator(null), 600)
    } else if (pct > 0.67) {
      // Right third — seek forward 10s
      playback.seekVideo(10)
      clearTimeout(indicatorTimer.current)
      setSeekIndicator({ side: 'right', amount: 10, key: Date.now() })
      indicatorTimer.current = setTimeout(() => setSeekIndicator(null), 600)
    } else {
      // Center third — play/pause
      playback.toggleVideoPlay()
    }
  }, [playback, resetHideTimer])

  // Touch start — record position
  const handleTouchStart = useCallback((e) => {
    if (e.target.closest('.lightbox-video-controls, .cast-overlay, .resume-toast, .auto-advance-overlay')) return
    const touch = e.touches[0]
    touchStartX.current = touch.clientX
    touchStartY.current = touch.clientY
    isDragging.current = false
    dragAmount.current = 0
  }, [])

  // Touch move — detect horizontal drag and show overlay
  const handleTouchMove = useCallback((e) => {
    if (e.target.closest('.lightbox-video-controls, .cast-overlay, .resume-toast, .auto-advance-overlay')) return
    const touch = e.touches[0]
    const dx = touch.clientX - touchStartX.current
    const dy = touch.clientY - touchStartY.current

    // Only activate if horizontal movement exceeds threshold and is more horizontal than vertical
    if (!isDragging.current) {
      if (Math.abs(dx) > 20 && Math.abs(dx) > Math.abs(dy) * 1.5) {
        isDragging.current = true
      } else {
        return
      }
    }

    // Prevent sidebar swipe from activating
    e.stopPropagation()

    // Calculate seek amount: 10s steps per ~50px after threshold
    const absDx = Math.abs(dx)
    const amount = Math.sign(dx) * Math.ceil((absDx - 20) / 50 + 0.001) * 10
    dragAmount.current = amount
    setDragSeek({ amount })
  }, [])

  // Touch end — perform seek if was dragging, set flag to prevent click
  const handleTouchEnd = useCallback((e) => {
    if (isDragging.current) {
      // Perform the seek
      if (dragAmount.current !== 0) {
        playback.seekVideo(dragAmount.current)
      }
      // Clear overlay
      setDragSeek(null)
      isDragging.current = false
      // Set flag to prevent the click event that fires after touchend
      wasDragging.current = true
      resetHideTimer()
    }
  }, [playback, resetHideTimer])

  return {
    handleVideoClick,
    handleTouchStart,
    handleTouchMove,
    handleTouchEnd,
    seekIndicator,
    dragSeek,
  }
}
