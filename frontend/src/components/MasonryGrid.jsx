import { useEffect, useRef, useCallback } from 'react'
import Masonry from 'react-masonry-css'
import MediaItem from './MediaItem'
import './MasonryGrid.css'

const defaultBreakpoints = {
  default: 8,
  2400: 7,
  1800: 6,
  1400: 5,
  1200: 4,
  900: 3,
  600: 2
}

const largeBreakpoints = {
  default: 5,
  1800: 4,
  1400: 3,
  900: 2,
  600: 1
}

function MasonryGrid({
  images,
  onImageClick,
  onLoadMore,
  loading,
  hasMore,
  user,
  onImageUpdate,
  showStatus = false,
  largeImages = false,
  isSelectable = false,
  selectedImages = new Set(),
  onSelectImage
}) {
  const breakpointColumns = largeImages ? largeBreakpoints : defaultBreakpoints
  const observerRef = useRef()
  const loadMoreRef = useRef()

  // Infinite scroll observer
  const handleObserver = useCallback((entries) => {
    const [entry] = entries
    if (entry.isIntersecting && hasMore && !loading) {
      onLoadMore()
    }
  }, [hasMore, loading, onLoadMore])

  useEffect(() => {
    const option = {
      root: null,
      rootMargin: '400px',
      threshold: 0
    }
    observerRef.current = new IntersectionObserver(handleObserver, option)

    if (loadMoreRef.current) {
      observerRef.current.observe(loadMoreRef.current)
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect()
      }
    }
  }, [handleObserver])

  if (!images.length && !loading) {
    return (
      <div className="masonry-empty">
        <p>No images found</p>
      </div>
    )
  }

  return (
    <div className="masonry-container">
      <Masonry
        breakpointCols={breakpointColumns}
        className="masonry-grid"
        columnClassName="masonry-column"
      >
        {images.map((image) => (
          <MediaItem
            key={`${image.id}-${image.is_favorite}`}
            image={image}
            onClick={() => onImageClick(image.id)}
            user={user}
            onRatingChange={onImageUpdate}
            onReject={onImageUpdate}
            showStatus={showStatus}
            isSelectable={isSelectable}
            isSelected={selectedImages.has(image.id)}
            onSelect={onSelectImage}
          />
        ))}
      </Masonry>

      <div ref={loadMoreRef} className="load-more-trigger">
        {loading && <div className="loading-spinner" />}
      </div>
    </div>
  )
}

export default MasonryGrid
