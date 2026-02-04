import { useState, useRef, useEffect, useCallback } from 'react'
import { getMediaUrl, fetchPreviewFrames } from '../api'
import './MediaItem.css'

// Check if filename is a video
const isVideo = (filename) => {
  if (!filename) return false
  const ext = filename.toLowerCase().split('.').pop()
  return ['webm', 'mp4', 'mov'].includes(ext)
}

function MediaItem({ image, onClick, isSelectable = false, isSelected = false, onSelect }) {
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(false)
  const [localRating] = useState(image?.rating)

  // Preview frames state
  const [previewFrames, setPreviewFrames] = useState([])
  const [currentFrame, setCurrentFrame] = useState(-1) // -1 means show thumbnail
  const [previewLoaded, setPreviewLoaded] = useState(false)
  const frameIntervalRef = useRef(null)
  const previewFetchedRef = useRef(false)
  const isHoveringRef = useRef(false) // Track hover state for late-loading frames
  const previewFramesRef = useRef([]) // Ref for closure-safe access in intervals

  // Compute derived values (safe before hooks)
  const thumbnailUrl = image?.thumbnail_url ? getMediaUrl(image.thumbnail_url) : ''
  const isVideoFile = isVideo(image?.original_filename)
  const fileStatus = image?.file_status || 'available'

  // Determine if we should use preview frames for hover animation
  const usePreviewFrames = isVideoFile && previewLoaded && previewFrames.length > 0

  // Fetch preview frames for videos - only when needed (on hover)
  const fetchFramesIfNeeded = useCallback(async () => {
    if (!image || !isVideo(image.original_filename) || previewFetchedRef.current) return
    previewFetchedRef.current = true

    try {
      const data = await fetchPreviewFrames(image.id, image.directory_id)
      if (data.frames && data.frames.length > 0) {
        const frameUrls = data.frames.map(url => getMediaUrl(url))

        // Preload frames and only keep ones that successfully load
        const loadResults = await Promise.all(frameUrls.map(url => {
          return new Promise((resolve) => {
            const img = new Image()
            img.onload = () => resolve(url)
            img.onerror = () => resolve(null) // Return null for failed loads
            img.src = url
          })
        }))

        // Filter out failed frames
        const validFrames = loadResults.filter(url => url !== null)

        // Only enable slideshow if we have valid frames
        if (validFrames.length > 0) {
          previewFramesRef.current = validFrames
          setPreviewFrames(validFrames)
          setPreviewLoaded(true)
        }
      }
    } catch (err) {
      // Silently fail - preview frames are optional
    }
  }, [image])

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current)
      }
    }
  }, [])

  const handleMouseEnter = useCallback(() => {
    isHoveringRef.current = true

    // If frames already loaded from previous hover, start slideshow
    const frames = previewFramesRef.current
    if (previewLoaded && frames.length > 0 && !frameIntervalRef.current) {
      setCurrentFrame(0)
      frameIntervalRef.current = setInterval(() => {
        setCurrentFrame(prev => (prev + 1) % previewFramesRef.current.length)
      }, 600)
    }
    // If not yet fetched, start fetching in background (slideshow will work on next hover)
    else if (isVideoFile && !previewFetchedRef.current && thumbnailUrl && loaded && !error) {
      fetchFramesIfNeeded()
    }
  }, [previewLoaded, isVideoFile, fetchFramesIfNeeded, thumbnailUrl, loaded, error])

  const handleMouseLeave = useCallback(() => {
    isHoveringRef.current = false
    // Stop frame slideshow and reset
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current)
      frameIntervalRef.current = null
    }
    setCurrentFrame(-1) // Back to thumbnail
  }, [])

  // Note: Slideshow only starts on subsequent hovers after frames are loaded
  // This prevents flickering issues during initial frame fetch

  // Handle click - either select or open lightbox
  const handleClick = (e) => {
    if (isSelectable) {
      e.preventDefault()
      e.stopPropagation()
      onSelect?.(image.id)
    } else {
      onClick?.()
    }
  }

  // Handle checkbox click specifically
  const handleCheckboxClick = (e) => {
    e.preventDefault()
    e.stopPropagation()
    onSelect?.(image.id)
  }

  const handleLoadError = () => {
    setError(true)
    // Always mark as loaded so we show the appropriate placeholder
    setLoaded(true)
  }

  // Get the current display image
  const getCurrentDisplaySrc = () => {
    const frames = previewFramesRef.current
    if (currentFrame >= 0 && frames[currentFrame]) {
      return frames[currentFrame]
    }
    return thumbnailUrl
  }

  // Render file status overlay
  const renderStatusOverlay = () => {
    if (fileStatus === 'drive_offline') {
      return (
        <div className="file-status-overlay drive-offline">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
            <path d="M3 3v5h5"/>
            <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>
            <path d="M16 16h5v5"/>
          </svg>
          <span>Drive Offline</span>
        </div>
      )
    }
    if (fileStatus === 'missing') {
      return (
        <div className="file-status-overlay file-missing">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M15 9l-6 6M9 9l6 6"/>
          </svg>
          <span>File Missing</span>
        </div>
      )
    }
    return null
  }

  // Guard against missing image data - AFTER all hooks
  if (!image || !image.thumbnail_url) {
    return (
      <div className="media-item media-error">
        <div className="error-placeholder">Image unavailable</div>
      </div>
    )
  }

  // Only show error for truly broken items, not just missing thumbnails
  if (error && fileStatus !== 'available') {
    return (
      <div className="media-item media-error" onClick={handleClick}>
        <div className="error-placeholder">Failed to load</div>
      </div>
    )
  }

  return (
    <div
      className={`media-item ${loaded ? 'loaded' : 'loading'} ${fileStatus !== 'available' ? 'unavailable' : ''} ${isSelectable ? 'selectable' : ''} ${isSelected ? 'selected' : ''}`}
      data-image-id={image.id}
      onClick={handleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Selection checkbox - always visible in selection mode */}
      {isSelectable && (
        <div className="selection-checkbox" onClick={handleCheckboxClick}>
          {isSelected ? (
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm-9 14l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            </svg>
          )}
        </div>
      )}

      {/* Always use thumbnail images in grid - never load actual videos */}
      <img
        src={getCurrentDisplaySrc()}
        alt=""
        loading="lazy"
        onLoad={() => setLoaded(true)}
        onError={handleLoadError}
      />
      {/* Video indicator overlay */}
      {isVideoFile && (
        <div className="video-indicator">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
          </svg>
        </div>
      )}

      {!loaded && <div className="loading-placeholder" />}

      {/* Show placeholder icon when thumbnail is still generating (loaded but src failed) */}
      {loaded && error && fileStatus === 'available' && (
        <div className="thumbnail-generating">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21,15 16,10 5,21"/>
          </svg>
        </div>
      )}

      {/* Rating badge */}
      <span className={`rating-badge rating-${localRating}`}>
        {localRating?.toUpperCase()}
      </span>

      {/* Favorite indicator */}
      {image.is_favorite && (
        <span className="favorite-indicator" title="Favorite">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
        </span>
      )}

      {/* File status overlay */}
      {renderStatusOverlay()}
    </div>
  )
}

export default MediaItem
