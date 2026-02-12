import { useEffect, useRef, useMemo, useState } from 'react'
import MediaItem from './MediaItem'
import FolderItem from './FolderItem'
import { getColumnCount } from '../utils/gridLayout'
import './MasonryGrid.css'

// Distribute images into columns based on aspect ratio to balance heights
// Returns both items and normalized heights for each column
function distributeToColumns(images, columnCount) {
  const columns = Array.from({ length: columnCount }, () => ({ items: [], height: 0 }))

  // Debug: count how many images have dimensions
  let withDimensions = 0
  let withoutDimensions = 0

  for (const image of images) {
    // Find the shortest column
    let shortestIdx = 0
    let shortestHeight = columns[0].height

    for (let i = 1; i < columns.length; i++) {
      if (columns[i].height < shortestHeight) {
        shortestHeight = columns[i].height
        shortestIdx = i
      }
    }

    // Calculate estimated height based on aspect ratio
    // Use a default aspect ratio of 4:3 (landscape) if dimensions unavailable
    const hasDimensions = image.width && image.height && image.width > 0 && image.height > 0
    if (hasDimensions) {
      withDimensions++
    } else {
      withoutDimensions++
    }

    const aspectRatio = hasDimensions
      ? image.width / image.height
      : 4/3  // Default to landscape if unknown

    // Normalized height (assuming column width of 1)
    const itemHeight = 1 / aspectRatio

    columns[shortestIdx].items.push(image)
    columns[shortestIdx].height += itemHeight
  }

  return columns
}

// Calculate how many skeleton placeholders each column needs to match the tallest
function getSkeletonCounts(columns) {
  if (columns.length === 0) return []

  const maxHeight = Math.max(...columns.map(col => col.height))
  const avgItemHeight = 1 // Assume square-ish skeleton items

  return columns.map(col => {
    const gap = maxHeight - col.height
    return Math.ceil(gap / avgItemHeight)
  })
}

function MasonryGrid({
  images,
  onImageClick,
  onFolderClick,
  onLoadMore,
  loading,
  hasMore,
  user,
  onImageUpdate,
  showStatus = false,
  largeImages = false,
  isSelectable = false,
  selectedImages = new Set(),
  onSelectImage,
  tileSize = 3
}) {
  const [columnCount, setColumnCount] = useState(() => getColumnCount(window.innerWidth, tileSize))
  const observerRef = useRef()
  const loadMoreRef = useRef()

  // Use refs for observer callback values to avoid recreating the observer
  const hasMoreRef = useRef(hasMore)
  const loadingRef = useRef(loading)
  const onLoadMoreRef = useRef(onLoadMore)
  const loadingTimeoutRef = useRef(null)

  // Keep refs in sync
  useEffect(() => {
    hasMoreRef.current = hasMore
  }, [hasMore])

  useEffect(() => {
    loadingRef.current = loading
  }, [loading])

  useEffect(() => {
    onLoadMoreRef.current = onLoadMore
  }, [onLoadMore])

  // Update column count on resize
  useEffect(() => {
    const handleResize = () => {
      setColumnCount(getColumnCount(window.innerWidth, tileSize))
    }

    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [tileSize])

  // Update column count when tileSize changes
  useEffect(() => {
    setColumnCount(getColumnCount(window.innerWidth, tileSize))
  }, [tileSize])

  // Distribute images across columns based on aspect ratios
  const columnData = useMemo(
    () => distributeToColumns(images, columnCount),
    [images, columnCount]
  )

  // Calculate skeleton counts to fill shorter columns when loading
  const skeletonCounts = useMemo(
    () => (loading && hasMore) ? getSkeletonCounts(columnData) : columnData.map(() => 0),
    [columnData, loading, hasMore]
  )

  // Infinite scroll observer - uses refs to avoid recreating observer on state changes
  useEffect(() => {
    const handleObserver = (entries) => {
      const [entry] = entries
      if (entry.isIntersecting && hasMoreRef.current && !loadingRef.current) {
        // Debounce to prevent rapid re-triggering during layout shifts
        if (loadingTimeoutRef.current) {
          clearTimeout(loadingTimeoutRef.current)
        }
        loadingTimeoutRef.current = setTimeout(() => {
          // Double-check state hasn't changed during debounce
          if (hasMoreRef.current && !loadingRef.current) {
            onLoadMoreRef.current()
          }
        }, 50)
      }
    }

    const option = {
      root: null,
      rootMargin: '1500px', // Load well ahead to hide column height differences
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
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current)
      }
    }
  }, []) // Empty deps - observer created once, uses refs for current values

  if (!images.length && !loading) {
    return (
      <div className="masonry-empty">
        <p>No images found</p>
      </div>
    )
  }

  // Calculate max column width based on tile size
  const maxColumnWidth = useMemo(() => {
    const widths = {
      1: 200,   // smallest tiles
      2: 250,
      3: 300,   // default
      4: 450,
      5: 600    // largest tiles
    }
    return widths[tileSize] || 300
  }, [tileSize])

  return (
    <div className="masonry-container" style={{ '--max-column-width': `${maxColumnWidth}px` }}>
      <div className="masonry-grid">
        {columnData.map((column, colIdx) => (
          <div key={colIdx} className="masonry-column">
            {column.items.map((image) => (
              <div key={image._isFolder ? `folder-${image.path}` : `${image.id}-${image.is_favorite}`}>
                {image._isFolder ? (
                  <FolderItem folder={image} onClick={() => onFolderClick(image.path)} />
                ) : (
                  <MediaItem
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
                )}
              </div>
            ))}
            {/* Skeleton placeholders to fill shorter columns while loading */}
            {skeletonCounts[colIdx] > 0 && Array.from({ length: skeletonCounts[colIdx] }).map((_, i) => (
              <div key={`skeleton-${i}`} className="masonry-skeleton" />
            ))}
          </div>
        ))}
      </div>

      <div ref={loadMoreRef} className="load-more-trigger">
        {loading && <div className="loading-spinner" />}
      </div>
    </div>
  )
}

export default MasonryGrid
