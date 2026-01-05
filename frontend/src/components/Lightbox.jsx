import { useEffect, useCallback, useState, useRef } from 'react'
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

  const image = images[currentIndex]

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
  }, [])

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

    touchStartX.current = null
    touchStartY.current = null
  }, [onNav, onSidebarHover, sidebarOpen])

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
        const result = await window.electronAPI.copyImageToClipboard(image.url)
        if (result.success) {
          setCopyFeedback('success')
        } else {
          throw new Error(result.error)
        }
      } else {
        // Fallback for browser - fetch and copy
        const response = await fetch(image.url)
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

    // Don't navigate if clicking on interactive elements
    if (e.target.closest('.lightbox-toolbar, .lightbox-counter, .lightbox-confirm-overlay, video')) return

    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const width = rect.width

    // Left 40% = previous (but not the sidebar hover zone which is ~100px)
    // Right 40% = next
    // Middle 20% = do nothing (where the image is)
    if (clickX < width * 0.4 && clickX > 100) {
      onNav(-1)
    } else if (clickX > width * 0.6) {
      onNav(1)
    }
  }

  if (!image) return null

  const isVideoFile = isVideo(image.filename)
  const fileStatus = image.file_status || 'available'
  const isUnavailable = fileStatus !== 'available'

  return (
    <div
      className="lightbox"
      onClick={handleNavClick}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
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
            src={image.url}
            controls
            autoPlay
            loop
            className="lightbox-media"
            onContextMenu={(e) => {
              e.preventDefault()
              if (window.electronAPI?.showImageContextMenu) {
                window.electronAPI.showImageContextMenu({
                  imageUrl: image.url,
                  filePath: image.file_path,
                  isVideo: true
                })
              }
            }}
          />
        ) : (
          <img
            key={image.id}
            src={image.url}
            alt=""
            className="lightbox-media"
            onContextMenu={(e) => {
              e.preventDefault()
              if (window.electronAPI?.showImageContextMenu) {
                window.electronAPI.showImageContextMenu({
                  imageUrl: image.url,
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
