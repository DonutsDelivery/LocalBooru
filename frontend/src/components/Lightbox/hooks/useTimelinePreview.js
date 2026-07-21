import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { fetchPreviewFrames, getMediaUrl } from '../../../api'
import {
  createTimelinePreviewOwner,
  shouldRetryTimelinePreview,
  timelinePreviewIdentityKey,
  timelinePreviewLocator,
} from './timelinePreviewLifecycle.js'

/**
 * Hook for managing timeline thumbnail preview on hover
 */
export function useTimelinePreview(image, duration) {
  const [previewFrames, setPreviewFrames] = useState([])
  const [hoverTime, setHoverTime] = useState(null)
  const [hoverX, setHoverX] = useState(0)
  const [isLoading, setIsLoading] = useState(false)
  const ownerRef = useRef(null)
  if (ownerRef.current === null) ownerRef.current = createTimelinePreviewOwner()
  const restartRef = useRef(null)
  const loadingRef = useRef(false)

  const locator = useMemo(() => timelinePreviewLocator(image), [image])
  const identityKey = timelinePreviewIdentityKey(locator)

  useEffect(() => {
    const owner = ownerRef.current
    owner.cancel()
    loadingRef.current = false
    // Reset visible data before starting work for the new exact media identity.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPreviewFrames([])
    setHoverTime(null)
    setIsLoading(false)
    if (!locator) {
      restartRef.current = null
      return undefined
    }

    const startLoad = () => {
      if (loadingRef.current) return
      loadingRef.current = true
      setIsLoading(true)
      const request = owner.begin(identityKey)
      let retries = 0
      const finish = () => {
        if (!owner.isCurrent(request)) return
        loadingRef.current = false
        setIsLoading(false)
      }
      const loadFrames = async () => {
        try {
          const data = await fetchPreviewFrames(locator, request.signal)
          if (!owner.isCurrent(request)) return
          if (data.frames?.length > 0) {
            const frameUrls = data.frames.map(url => getMediaUrl(url))
            setPreviewFrames(frameUrls)
            frameUrls.forEach(url => {
              const img = new Image()
              img.src = url
            })
            finish()
          } else if (shouldRetryTimelinePreview(data.generating, retries)) {
            retries += 1
            owner.schedule(request, loadFrames, 3000)
          } else {
            finish()
          }
        } catch (error) {
          if (!owner.isCurrent(request) || error?.name === 'AbortError' || error?.code === 'ERR_CANCELED') return
          console.warn('[TimelinePreview] Failed to load frames:', error)
          finish()
        }
      }

      loadFrames()
    }

    restartRef.current = startLoad
    startLoad()
    return () => {
      restartRef.current = null
      loadingRef.current = false
      owner.cancel()
    }
  }, [identityKey])

  const handleTimelineHover = useCallback((e) => {
    if (previewFrames.length === 0) {
      if (!loadingRef.current) restartRef.current?.()
      return
    }
    if (!duration) return

    const timeline = e.currentTarget
    if (!timeline) return

    const rect = timeline.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percent = Math.max(0, Math.min(1, x / rect.width))
    setHoverTime(percent * duration)
    setHoverX(x)
  }, [duration, previewFrames.length])

  const handleTimelineHoverEnd = useCallback(() => {
    setHoverTime(null)
  }, [])

  const getCurrentFrame = useCallback(() => {
    if (hoverTime === null || previewFrames.length === 0 || !duration) return null

    const percent = hoverTime / duration
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
    handleTimelineHover,
    handleTimelineHoverEnd,
    getCurrentFrame,
    hasPreviewFrames: previewFrames.length > 0
  }
}
