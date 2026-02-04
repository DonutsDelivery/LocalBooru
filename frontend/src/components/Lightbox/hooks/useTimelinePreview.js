import { useState, useCallback, useRef, useEffect } from 'react'
import { fetchPreviewFrames, getMediaUrl } from '../../../api'

/**
 * Hook for managing timeline thumbnail preview on hover
 */
export function useTimelinePreview(imageId, directoryId, duration) {
  const [previewFrames, setPreviewFrames] = useState([])
  const [hoverTime, setHoverTime] = useState(null)
  const [hoverX, setHoverX] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const fetchedRef = useRef(false)
  const timelineRef = useRef(null)

  // Fetch preview frames when image changes
  useEffect(() => {
    if (!imageId || fetchedRef.current) return

    fetchedRef.current = true
    setIsLoading(true)

    const loadFrames = async () => {
      try {
        const data = await fetchPreviewFrames(imageId, directoryId)
        if (data.frames && data.frames.length > 0) {
          const frameUrls = data.frames.map(url => getMediaUrl(url))
          setPreviewFrames(frameUrls)
          // Preload frames
          frameUrls.forEach(url => {
            const img = new Image()
            img.src = url
          })
        } else if (data.generating) {
          // Retry after delay
          setTimeout(loadFrames, 3000)
          return
        }
      } catch (err) {
        console.warn('[TimelinePreview] Failed to load frames:', err)
      }
      setIsLoading(false)
    }

    loadFrames()
  }, [imageId, directoryId])

  // Reset when image changes
  useEffect(() => {
    fetchedRef.current = false
    setPreviewFrames([])
    setHoverTime(null)
    setIsLoading(false)
  }, [imageId])

  // Handle timeline hover - works with event target directly
  const handleTimelineHover = useCallback((e) => {
    if (!duration || previewFrames.length === 0) return

    // Get timeline element from event (works whether we're hovering the track or the container)
    const timeline = e.currentTarget
    if (!timeline) return

    const rect = timeline.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percent = Math.max(0, Math.min(1, x / rect.width))
    const time = percent * duration

    setHoverTime(time)
    setHoverX(x)
  }, [duration, previewFrames.length])

  // Clear hover state
  const handleTimelineHoverEnd = useCallback(() => {
    setHoverTime(null)
  }, [])

  // Get current preview frame based on hover time
  const getCurrentFrame = useCallback(() => {
    if (hoverTime === null || previewFrames.length === 0 || !duration) return null

    const percent = hoverTime / duration
    // Map to frame index (0 to numFrames-1)
    const frameIndex = Math.min(
      Math.floor(percent * previewFrames.length),
      previewFrames.length - 1
    )
    return previewFrames[frameIndex]
  }, [hoverTime, previewFrames, duration])

  return {
    previewFrames,
    hoverTime,
    hoverX,
    isLoading,
    timelineRef,
    handleTimelineHover,
    handleTimelineHoverEnd,
    getCurrentFrame,
    hasPreviewFrames: previewFrames.length > 0
  }
}
