import { useState, useRef, useEffect } from 'react'
import './MediaItem.css'

// Check if filename is a video
const isVideo = (filename) => {
  if (!filename) return false
  const ext = filename.toLowerCase().split('.').pop()
  return ['webm', 'mp4', 'mov'].includes(ext)
}

function MediaItem({ image, onClick }) {
  const [loaded, setLoaded] = useState(false)
  const [error, setError] = useState(false)
  const [localRating, setLocalRating] = useState(image?.rating)
  const [isShortVideo, setIsShortVideo] = useState(image?.duration != null ? image.duration <= 10 : false)

  const videoRef = useRef()

  // Guard against missing image data
  if (!image || !image.thumbnail_url) {
    return (
      <div className="media-item media-error">
        <div className="error-placeholder">Image unavailable</div>
      </div>
    )
  }

  const thumbnailUrl = image.thumbnail_url
  const isVideoFile = isVideo(image.filename)
  const fileStatus = image.file_status || 'available'

  // Handle video duration check for autoplay
  const handleVideoLoaded = () => {
    setLoaded(true)
    if (videoRef.current) {
      const duration = image?.duration != null ? image.duration : videoRef.current.duration
      if (duration <= 10) {
        setIsShortVideo(true)
        videoRef.current.play().catch(() => {})
      }
    }
  }

  // Auto-start short videos if duration is known from API
  useEffect(() => {
    if (isShortVideo && videoRef.current && loaded) {
      videoRef.current.play().catch(() => {})
    }
  }, [isShortVideo, loaded])

  const handleMouseEnter = () => {
    if (videoRef.current && !isShortVideo) {
      videoRef.current.play().catch(() => {})
    }
  }

  const handleMouseLeave = () => {
    if (videoRef.current && !isShortVideo) {
      videoRef.current.pause()
      videoRef.current.currentTime = 0
    }
  }

  const handleLoadError = () => {
    setError(true)
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

  if (error && fileStatus === 'available') {
    return (
      <div className="media-item media-error" onClick={onClick}>
        <div className="error-placeholder">Failed to load</div>
      </div>
    )
  }

  return (
    <div
      className={`media-item ${loaded ? 'loaded' : 'loading'} ${fileStatus !== 'available' ? 'unavailable' : ''}`}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {isVideoFile ? (
        <>
          <video
            ref={videoRef}
            src={image.url}
            poster={thumbnailUrl}
            muted
            loop
            playsInline
            preload="metadata"
            onLoadedMetadata={handleVideoLoaded}
            onError={handleLoadError}
          />
          {!isShortVideo && (
            <div className="video-indicator">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
              </svg>
            </div>
          )}
        </>
      ) : (
        <img
          src={thumbnailUrl}
          alt=""
          loading="lazy"
          onLoad={() => setLoaded(true)}
          onError={handleLoadError}
        />
      )}

      {!loaded && <div className="loading-placeholder" />}

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
