import { useState, useEffect, useMemo } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { fetchDirectories, getFileDimensions } from '../../api'
import PromptSection from './PromptSection'
import FilterControls, { ALL_RATINGS, MIN_AGE_LIMIT, MAX_AGE_LIMIT, RESOLUTION_OPTIONS, ORIENTATION_OPTIONS, DURATION_OPTIONS } from './FilterControls'
import TagSearch from './TagSearch'
import '../Sidebar.css'

function Sidebar({
  tags,
  onTagClick,
  collapsed,
  mobileOpen,
  onClose,
  currentTags,
  selectedImage,
  onSearch,
  initialTags,
  initialRating,
  initialFavoritesOnly,
  initialDirectoryId,
  initialMinAge,
  initialMaxAge,
  initialSort,
  initialTimeframe,
  initialFilename,
  initialResolution,
  initialOrientation,
  initialDuration,
  total,
  stats,
  lightboxMode,
  lightboxHover,
  onMouseLeave
}) {
  const location = useLocation()
  const isGalleryPage = location.pathname === '/'
  const [hovering, setHovering] = useState(false)
  const [directories, setDirectories] = useState([])
  const [selectedDirectory, setSelectedDirectory] = useState(initialDirectoryId || null)
  const [appVersion, setAppVersion] = useState(null)
  const [updateStatus, setUpdateStatus] = useState(null)
  const [selectedRatings, setSelectedRatings] = useState(() => {
    if (initialRating) {
      let ratings = initialRating.split(',').filter(r => ALL_RATINGS.includes(r))
      if (ratings.length > 0) return ratings
    }
    return [...ALL_RATINGS]
  })
  const [sortBy, setSortBy] = useState(initialSort || 'newest')
  const [favoritesOnly, setFavoritesOnly] = useState(initialFavoritesOnly || false)
  const [minAge, setMinAge] = useState(initialMinAge || null)
  const [maxAge, setMaxAge] = useState(initialMaxAge || null)
  const [timeframe, setTimeframe] = useState(initialTimeframe || null)
  const [filenameSearch, setFilenameSearch] = useState(initialFilename || '')
  const [resolution, setResolution] = useState(initialResolution || null)
  const [orientation, setOrientation] = useState(initialOrientation || null)
  const [duration, setDuration] = useState(initialDuration || null)
  const [fetchedDimensions, setFetchedDimensions] = useState(null)
  const [filtersExpanded, setFiltersExpanded] = useState(() => {
    const saved = localStorage.getItem('filtersExpanded')
    return saved !== null ? JSON.parse(saved) : false
  })

  // Load directories
  useEffect(() => {
    fetchDirectories().then(data => {
      setDirectories(data.directories || [])
    }).catch(console.error)
  }, [])

  // Persist filters expanded state
  useEffect(() => {
    localStorage.setItem('filtersExpanded', JSON.stringify(filtersExpanded))
  }, [filtersExpanded])

  // Get app version and listen for updates (Electron only)
  useEffect(() => {
    if (window.electronAPI) {
      window.electronAPI.getVersion().then(setAppVersion).catch(console.error)
      const unsubscribe = window.electronAPI.onUpdaterStatus((status) => {
        setUpdateStatus(status)
      })
      return () => unsubscribe?.()
    }
  }, [])

  useEffect(() => {
    if (initialRating) {
      let ratings = initialRating.split(',').filter(r => ALL_RATINGS.includes(r))
      if (ratings.length > 0) setSelectedRatings(ratings)
    }
    setFavoritesOnly(initialFavoritesOnly || false)
    setSelectedDirectory(initialDirectoryId || null)
    setMinAge(initialMinAge || null)
    setMaxAge(initialMaxAge || null)
    setTimeframe(initialTimeframe || null)
    setFilenameSearch(initialFilename || '')
    setResolution(initialResolution || null)
    setOrientation(initialOrientation || null)
    setDuration(initialDuration || null)
    if (initialSort) setSortBy(initialSort)
  }, [initialRating, initialFavoritesOnly, initialDirectoryId, initialMinAge, initialMaxAge, initialSort, initialTimeframe, initialFilename, initialResolution, initialOrientation, initialDuration])

  // Fetch dimensions for selected image when it changes
  useEffect(() => {
    if (!selectedImage || !selectedImage.file_path) {
      setFetchedDimensions(null)
      return
    }

    getFileDimensions(selectedImage.file_path)
      .then(result => {
        if (result.success) {
          setFetchedDimensions({
            width: result.width,
            height: result.height
          })
        } else {
          setFetchedDimensions(null)
        }
      })
      .catch(err => {
        console.error('Failed to fetch dimensions:', err)
        setFetchedDimensions(null)
      })
  }, [selectedImage?.id, selectedImage?.file_path])

  const isVisible = !collapsed || hovering || mobileOpen

  // Memoize activeTags to prevent unnecessary re-renders and useEffect triggers
  const activeTags = useMemo(() =>
    currentTags ? currentTags.split(',').map(t => t.trim()) : []
  , [currentTags])

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration)
  }

  const handleClear = () => {
    setFilenameSearch('')
    setSelectedRatings([...ALL_RATINGS])
    setSortBy('newest')
    setFavoritesOnly(false)
    setSelectedDirectory(null)
    setMinAge(null)
    setMaxAge(null)
    setTimeframe(null)
    setResolution(null)
    setOrientation(null)
    setDuration(null)
    onSearch('', ALL_RATINGS.join(','), 'newest', false, null, null, null, null, '', null, null, null)
  }

  const handleSortChange = (newSortBy) => {
    setSortBy(newSortBy)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, newSortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration)
  }

  const handleDirectoryChange = (dirId) => {
    const newDirId = dirId === '' ? null : parseInt(dirId)
    setSelectedDirectory(newDirId)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, newDirId, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration)
  }

  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, newTimeframe, filenameSearch, resolution, orientation, duration)
  }

  const handleResolutionChange = (newResolution) => {
    setResolution(newResolution)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, newResolution, orientation, duration)
  }

  const handleOrientationChange = (newOrientation) => {
    setOrientation(newOrientation)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, newOrientation, duration)
  }

  const handleDurationChange = (newDuration) => {
    setDuration(newDuration)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, newDuration)
  }

  const toggleRating = (rating) => {
    setSelectedRatings(prev => {
      const isSelected = prev.includes(rating)
      if (isSelected) {
        const newRatings = prev.filter(r => r !== rating)
        if (newRatings.length === 0) return [...ALL_RATINGS]
        return newRatings
      }
      return [...prev, rating]
    })
  }

  const toggleFavorites = () => {
    const newFavOnly = !favoritesOnly
    setFavoritesOnly(newFavOnly)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, newFavOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration)
  }

  const handleFilenameSearchClear = () => {
    setFilenameSearch('')
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, '', resolution, orientation, duration)
  }

  const handleAgeChange = () => {
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration)
  }

  return (
    <aside
      className={`sidebar ${collapsed ? 'collapsed' : ''} ${isVisible ? 'visible' : ''} ${mobileOpen ? 'mobile-open' : ''} ${lightboxMode ? 'lightbox-mode' : ''} ${lightboxHover ? 'lightbox-hover' : ''}`}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => {
        setHovering(false)
        if (onMouseLeave) onMouseLeave()
      }}
    >
      {mobileOpen && (
        <button className="sidebar-close" onClick={onClose}>
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      )}

      <div className="sidebar-content">
        {/* Version info */}
        {appVersion && (
          <div className="version-info">
            <span className="version-number">v{appVersion}</span>
            {updateStatus?.status === 'available' && (
              <button
                className="update-badge"
                onClick={() => window.electronAPI?.downloadUpdate()}
                title={`Update to v${updateStatus.version}`}
              >
                Update
              </button>
            )}
            {updateStatus?.status === 'downloading' && (
              <span className="update-badge downloading">
                {Math.round(updateStatus.progress || 0)}%
              </span>
            )}
            {updateStatus?.status === 'downloaded' && (
              <button
                className="update-badge ready"
                onClick={() => window.electronAPI?.installUpdate()}
                title="Click to restart and install"
              >
                Restart
              </button>
            )}
          </div>
        )}

        <nav className="sidebar-nav">
          <NavLink to="/" end className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M4 8h4V4H4v4zm6 12h4v-4h-4v4zm-6 0h4v-4H4v4zm0-6h4v-4H4v4zm6 0h4v-4h-4v4zm6-10v4h4V4h-4zm-6 4h4V4h-4v4zm6 6h4v-4h-4v4zm0 6h4v-4h-4v4z"/>
            </svg>
            <span className="nav-text">Gallery</span>
            {stats && <span className="nav-count">{stats.total_images.toLocaleString()}</span>}
          </NavLink>
          <NavLink to="/directories" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
            </svg>
            Directories
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
            </svg>
            Settings
          </NavLink>
        </nav>

        {/* Search Section - only show on gallery page */}
        {isGalleryPage && <div className="sidebar-section search-section">
          <form onSubmit={handleSearchSubmit}>
            {/* Collapsible Filters Header */}
            <button
              type="button"
              className={`filters-toggle ${filtersExpanded ? 'expanded' : ''}`}
              onClick={() => setFiltersExpanded(!filtersExpanded)}
            >
              <svg className="filters-toggle-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/>
              </svg>
              <span>Filters</span>
              {total > 0 && <span className="filters-count">{total.toLocaleString()}</span>}
              <svg className="filters-chevron" viewBox="0 0 24 24" fill="currentColor">
                <path d="M7 10l5 5 5-5z"/>
              </svg>
            </button>

            {/* Collapsible Filters Content */}
            <div className={`filters-content ${filtersExpanded ? 'expanded' : ''}`}>
              <div className="filters-inner">
                <FilterControls
                  directories={directories}
                  selectedDirectory={selectedDirectory}
                  onDirectoryChange={handleDirectoryChange}
                  filenameSearch={filenameSearch}
                  setFilenameSearch={setFilenameSearch}
                  onFilenameSearchClear={handleFilenameSearchClear}
                  onSearchSubmit={handleSearchSubmit}
                  favoritesOnly={favoritesOnly}
                  onToggleFavorites={toggleFavorites}
                  selectedRatings={selectedRatings}
                  onToggleRating={toggleRating}
                  minAge={minAge}
                  maxAge={maxAge}
                  setMinAge={setMinAge}
                  setMaxAge={setMaxAge}
                  onAgeChange={handleAgeChange}
                  sortBy={sortBy}
                  onSortChange={handleSortChange}
                  timeframe={timeframe}
                  onTimeframeChange={handleTimeframeChange}
                  resolution={resolution}
                  onResolutionChange={handleResolutionChange}
                  orientation={orientation}
                  onOrientationChange={handleOrientationChange}
                  duration={duration}
                  onDurationChange={handleDurationChange}
                  total={0}
                />
              </div>
            </div>

            <div className="search-controls">
              <button type="submit" className="search-button">
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                </svg>
                Search
              </button>
              <button type="button" className="clear-button" onClick={handleClear}>
                Clear
              </button>
            </div>
          </form>
        </div>}

        {/* Selected Image Info */}
        {isGalleryPage && selectedImage && (
          <div className="sidebar-section image-info">
            <h3>Image Info</h3>
            <div className="info-grid">
              <span className="info-label">ID</span>
              <span className="info-value">#{selectedImage.id}</span>

              {fetchedDimensions ? (
                <>
                  <span className="info-label">Resolution</span>
                  <span className="info-value">{fetchedDimensions.width}x{fetchedDimensions.height}</span>
                </>
              ) : null}

              <span className="info-label">Rating</span>
              <span className={`info-value rating-${selectedImage.rating}`}>
                {selectedImage.rating}
              </span>

              {selectedImage.created_at && (
                <>
                  <span className="info-label">Added</span>
                  <span className="info-value">
                    {new Date(selectedImage.created_at).toLocaleDateString()}
                  </span>
                </>
              )}

              {selectedImage.file_size && (
                <>
                  <span className="info-label">File Size</span>
                  <span className="info-value">
                    {selectedImage.file_size >= 1024 * 1024
                      ? `${(selectedImage.file_size / (1024 * 1024)).toFixed(1)} MB`
                      : `${(selectedImage.file_size / 1024).toFixed(0)} KB`}
                  </span>
                </>
              )}

              {selectedImage.file_path && (
                <>
                  <span className="info-label">Format</span>
                  <span className="info-value">
                    {selectedImage.file_path.split('.').pop()?.toUpperCase()}
                  </span>
                </>
              )}

              {selectedImage.directory_name && (
                <>
                  <span className="info-label">Directory</span>
                  <span className="info-value">{selectedImage.directory_name}</span>
                </>
              )}

              {selectedImage.file_path && (
                <>
                  <span className="info-label">Path</span>
                  <span className="info-value file-path" title={selectedImage.file_path}>
                    {selectedImage.file_path}
                  </span>
                </>
              )}

              {selectedImage.file_status && selectedImage.file_status !== 'available' && (
                <>
                  <span className="info-label">Status</span>
                  <span className={`info-value status-${selectedImage.file_status}`}>
                    {selectedImage.file_status === 'drive_offline' ? 'Drive Offline' : 'File Missing'}
                  </span>
                </>
              )}

              {selectedImage.num_faces !== null && selectedImage.num_faces !== undefined && (
                <>
                  <span className="info-label">Faces</span>
                  <span className="info-value">{selectedImage.num_faces}</span>

                  {selectedImage.min_age && (
                    <>
                      <span className="info-label">Age</span>
                      <span className="info-value">
                        {selectedImage.min_age === selectedImage.max_age
                          ? `${selectedImage.min_age}`
                          : `${selectedImage.min_age}-${selectedImage.max_age}`}
                      </span>
                    </>
                  )}
                </>
              )}
            </div>

            {selectedImage.tags?.length > 0 && (
              <div className="image-tags">
                <h4>Tags</h4>
                <div className="tag-list">
                  {selectedImage.tags.map(tag => (
                    <button
                      key={tag.name}
                      className={`tag tag-${tag.category}`}
                      onClick={() => onTagClick(tag.name)}
                    >
                      {tag.name.replace(/_/g, ' ')}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* AI Generation Prompts */}
            {(selectedImage.prompt || selectedImage.negative_prompt) && (
              <div className="image-prompts">
                <h4>AI Generation</h4>

                {selectedImage.prompt && (
                  <PromptSection
                    label="Positive"
                    text={selectedImage.prompt}
                    isNegative={false}
                  />
                )}

                {selectedImage.negative_prompt && (
                  <PromptSection
                    label="Negative"
                    text={selectedImage.negative_prompt}
                    isNegative={true}
                  />
                )}

                {/* Generation parameters */}
                {(selectedImage.model_name || selectedImage.seed || selectedImage.steps || selectedImage.cfg_scale || selectedImage.sampler) && (
                  <div className="prompt-params">
                    {selectedImage.model_name && (
                      <span className="param-tag">Model: {selectedImage.model_name}</span>
                    )}
                    {selectedImage.seed && (
                      <span className="param-tag">Seed: {selectedImage.seed}</span>
                    )}
                    {selectedImage.steps && (
                      <span className="param-tag">Steps: {selectedImage.steps}</span>
                    )}
                    {selectedImage.cfg_scale && (
                      <span className="param-tag">CFG: {selectedImage.cfg_scale}</span>
                    )}
                    {selectedImage.sampler && (
                      <span className="param-tag">Sampler: {selectedImage.sampler}</span>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Tag Browser */}
        {isGalleryPage && !selectedImage && (
          <TagSearch
            tags={tags}
            activeTags={activeTags}
            onTagClick={onTagClick}
          />
        )}

        {/* Support link */}
        <div className="sidebar-support">
          <a href="https://ko-fi.com/donutsdelivery" target="_blank" rel="noopener noreferrer">
            <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
            </svg>
            Support
          </a>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
