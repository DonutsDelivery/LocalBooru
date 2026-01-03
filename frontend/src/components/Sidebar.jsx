import { useState, useEffect, useRef } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { fetchDirectories } from '../api'
import './Sidebar.css'

// Collapsible prompt section with copy button
function PromptSection({ label, text, isNegative }) {
  const [expanded, setExpanded] = useState(false)
  const [copied, setCopied] = useState(false)

  const isLong = text.length > 150
  const displayText = expanded || !isLong ? text : text.slice(0, 150) + '...'

  const handleCopy = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className={`prompt-section ${isNegative ? 'negative' : 'positive'}`}>
      <div className="prompt-header">
        <span className="prompt-label">{label}</span>
        <button className="copy-btn" onClick={handleCopy} title="Copy to clipboard">
          {copied ? (
            <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
              <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/>
            </svg>
          )}
        </button>
      </div>
      <div className={`prompt-text ${expanded ? 'expanded' : ''}`}>
        {displayText}
      </div>
      {isLong && (
        <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  )
}

// Rating definitions for LocalBooru (all ratings shown)
const ALL_RATINGS = ['pg', 'pg13', 'r', 'x', 'xxx']

// Local storage key for persistent filters
const FILTER_STORAGE_KEY = 'localbooru_filters'

// Sort options
const SORT_OPTIONS = [
  { value: 'newest', label: 'Newest' },
  { value: 'oldest', label: 'Oldest' },
  { value: 'random', label: 'Random' }
]

// Age range constants
const MIN_AGE_LIMIT = 0
const MAX_AGE_LIMIT = 80

// Timeframe options
const TIMEFRAME_OPTIONS = [
  { value: null, label: 'All Time' },
  { value: 'today', label: 'Today' },
  { value: 'week', label: 'This Week' },
  { value: 'month', label: 'This Month' },
  { value: 'year', label: 'This Year' }
]

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
  total,
  stats,
  lightboxMode,
  lightboxHover,
  onMouseLeave
}) {
  const location = useLocation()
  const isGalleryPage = location.pathname === '/'
  const [hovering, setHovering] = useState(false)
  const [tagInput, setTagInput] = useState('')
  const [suggestionIndex, setSuggestionIndex] = useState(-1)
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

  // Load directories
  useEffect(() => {
    fetchDirectories().then(data => {
      setDirectories(data.directories || [])
    }).catch(console.error)
  }, [])

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
    if (initialSort) setSortBy(initialSort)
  }, [initialRating, initialFavoritesOnly, initialDirectoryId, initialMinAge, initialMaxAge, initialSort, initialTimeframe])

  const isVisible = !collapsed || hovering || mobileOpen
  const activeTags = currentTags ? currentTags.split(',').map(t => t.trim()) : []

  // Group tags by category
  const groupedTags = (tags || []).reduce((acc, tag) => {
    const category = tag.category || 'general'
    if (!acc[category]) acc[category] = []
    acc[category].push(tag)
    return acc
  }, {})

  const categoryOrder = ['artist', 'character', 'copyright', 'general', 'meta']

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe)
  }

  const handleClear = () => {
    setTagInput('')
    setSelectedRatings([...ALL_RATINGS])
    setSortBy('newest')
    setFavoritesOnly(false)
    setSelectedDirectory(null)
    setMinAge(null)
    setMaxAge(null)
    setTimeframe(null)
    onSearch('', ALL_RATINGS.join(','), 'newest', false, null, null, null, null)
  }

  const handleSortChange = (newSortBy) => {
    setSortBy(newSortBy)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, newSortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe)
  }

  const handleDirectoryChange = (dirId) => {
    const newDirId = dirId === '' ? null : parseInt(dirId)
    setSelectedDirectory(newDirId)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, newDirId, minAge, maxAge, timeframe)
  }

  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe)
    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, newTimeframe)
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
    onSearch(currentTags || '', ratingParam, sortBy, newFavOnly, selectedDirectory, minAge, maxAge, timeframe)
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
            {/* Directory Filter */}
            <div className="directory-filter">
              <select
                value={selectedDirectory || ''}
                onChange={(e) => handleDirectoryChange(e.target.value)}
                className="directory-select"
              >
                <option value="">All Directories</option>
                {directories
                  .filter(dir => dir.image_count > 0)
                  .map(dir => (
                    <option key={dir.id} value={dir.id}>
                      {dir.name} ({dir.image_count})
                    </option>
                  ))}
              </select>
            </div>

            {/* Favorites Toggle */}
            <label className="toggle-row">
              <span className="toggle-label">
                <svg viewBox="0 0 24 24" fill="currentColor" className="toggle-icon">
                  <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                </svg>
                Favorites
              </span>
              <div className={`toggle-switch ${favoritesOnly ? 'active' : ''}`} onClick={toggleFavorites}>
                <div className="toggle-knob" />
              </div>
            </label>

            {/* Rating Buttons */}
            <div className="rating-buttons">
              {ALL_RATINGS.map(rating => (
                <button
                  key={rating}
                  type="button"
                  className={`rating-btn rating-${rating} ${selectedRatings.includes(rating) ? 'active' : ''}`}
                  onClick={() => toggleRating(rating)}
                >
                  {rating.toUpperCase()}
                </button>
              ))}
            </div>

            {/* Age Range Slider */}
            <div className="age-filter">
              <div className="age-filter-header">
                <span className="age-filter-label">Age Range:</span>
                <span className="age-filter-value">
                  {minAge || MIN_AGE_LIMIT} - {maxAge ? maxAge : '80+'}
                </span>
              </div>
              <div className="age-slider-container">
                <input
                  type="range"
                  min={MIN_AGE_LIMIT}
                  max={MAX_AGE_LIMIT}
                  value={minAge || MIN_AGE_LIMIT}
                  onChange={(e) => {
                    const val = parseInt(e.target.value)
                    const newMin = val === MIN_AGE_LIMIT ? null : val
                    setMinAge(newMin)
                  }}
                  onMouseUp={() => {
                    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
                    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe)
                  }}
                  onTouchEnd={() => {
                    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
                    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe)
                  }}
                  className="age-slider age-slider-min"
                />
                <input
                  type="range"
                  min={MIN_AGE_LIMIT}
                  max={MAX_AGE_LIMIT}
                  value={maxAge || MAX_AGE_LIMIT}
                  onChange={(e) => {
                    const val = parseInt(e.target.value)
                    const newMax = val === MAX_AGE_LIMIT ? null : val
                    setMaxAge(newMax)
                  }}
                  onMouseUp={() => {
                    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
                    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe)
                  }}
                  onTouchEnd={() => {
                    const ratingParam = selectedRatings.length > 0 ? selectedRatings.join(',') : ''
                    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe)
                  }}
                  className="age-slider age-slider-max"
                />
              </div>
            </div>

            {/* Sort Controls */}
            <div className="sort-controls">
              <div className="sort-buttons">
                {SORT_OPTIONS.map(option => (
                  <button
                    key={option.value}
                    type="button"
                    className={`sort-btn ${sortBy === option.value ? 'active' : ''}`}
                    onClick={() => handleSortChange(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Timeframe Filter */}
            <div className="timeframe-filter">
              <div className="timeframe-buttons">
                {TIMEFRAME_OPTIONS.map(option => (
                  <button
                    key={option.value || 'all'}
                    type="button"
                    className={`timeframe-btn ${timeframe === option.value ? 'active' : ''}`}
                    onClick={() => handleTimeframeChange(option.value)}
                  >
                    {option.label}
                  </button>
                ))}
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

            {total > 0 && (
              <div className="search-results-count">
                {total.toLocaleString()} results
              </div>
            )}
          </form>
        </div>}

        {/* Selected Image Info */}
        {isGalleryPage && selectedImage && (
          <div className="sidebar-section image-info">
            <h3>Image Info</h3>
            <div className="info-grid">
              <span className="info-label">ID</span>
              <span className="info-value">#{selectedImage.id}</span>

              <span className="info-label">Size</span>
              <span className="info-value">{selectedImage.width}x{selectedImage.height}</span>

              <span className="info-label">Rating</span>
              <span className={`info-value rating-${selectedImage.rating}`}>
                {selectedImage.rating}
              </span>

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
          <>
            <div className="sidebar-section">
              <h3>Tags</h3>
              <div className="search-input-wrapper">
                <svg className="search-icon" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
                </svg>
                <input
                  type="text"
                  value={tagInput}
                  onChange={(e) => {
                    setTagInput(e.target.value)
                    setSuggestionIndex(-1)
                  }}
                  onKeyDown={(e) => {
                    const suggestions = (tags || []).filter(tag =>
                      tag.name.toLowerCase().includes(tagInput.toLowerCase()) &&
                      !activeTags.includes(tag.name)
                    ).slice(0, 8)

                    if (e.key === 'ArrowDown') {
                      e.preventDefault()
                      setSuggestionIndex(prev => prev < suggestions.length - 1 ? prev + 1 : 0)
                    } else if (e.key === 'ArrowUp') {
                      e.preventDefault()
                      setSuggestionIndex(prev => prev > 0 ? prev - 1 : suggestions.length - 1)
                    } else if (e.key === 'Enter' && tagInput.trim()) {
                      e.preventDefault()
                      const selectedTag = suggestionIndex >= 0 ? suggestions[suggestionIndex] : suggestions[0]
                      if (selectedTag) {
                        onTagClick(selectedTag.name)
                        setTagInput('')
                        setSuggestionIndex(-1)
                      }
                    } else if (e.key === 'Escape') {
                      setTagInput('')
                      setSuggestionIndex(-1)
                    }
                  }}
                  placeholder="Type to search tags..."
                  className="search-input"
                />
                {tagInput && (
                  <button type="button" className="search-clear" onClick={() => { setTagInput(''); setSuggestionIndex(-1) }}>
                    <svg viewBox="0 0 24 24" fill="currentColor">
                      <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                    </svg>
                  </button>
                )}
                {/* Autocomplete suggestions */}
                {tagInput.length >= 2 && (
                  <div className="tag-suggestions">
                    {(tags || [])
                      .filter(tag =>
                        tag.name.toLowerCase().includes(tagInput.toLowerCase()) &&
                        !activeTags.includes(tag.name)
                      )
                      .slice(0, 8)
                      .map((tag, index) => (
                        <button
                          key={tag.name}
                          className={`suggestion-item tag-${tag.category} ${index === suggestionIndex ? 'selected' : ''}`}
                          onClick={() => {
                            onTagClick(tag.name)
                            setTagInput('')
                            setSuggestionIndex(-1)
                          }}
                        >
                          <span className="suggestion-name">{tag.name.replace(/_/g, ' ')}</span>
                          <span className="suggestion-count">({tag.post_count})</span>
                        </button>
                      ))
                    }
                  </div>
                )}
              </div>
            </div>

            {/* Active Tags */}
            {activeTags.length > 0 && (
              <div className="sidebar-section active-tags">
                <h4>Active Filters</h4>
                <div className="tag-list">
                  {activeTags.map(tag => (
                    <button
                      key={tag}
                      className="tag active"
                      onClick={() => onTagClick(tag)}
                      title="Remove from filters"
                    >
                      {tag.replace(/_/g, ' ')}
                      <svg viewBox="0 0 24 24" fill="currentColor" className="remove-icon">
                        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                      </svg>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Tag Categories */}
            {categoryOrder.map(category => {
              const categoryTags = groupedTags[category]
              if (!categoryTags?.length) return null

              return (
                <div key={category} className="sidebar-section">
                  <h4 className={`category-header category-${category}`}>
                    {category}
                  </h4>
                  <div className="tag-list compact">
                    {categoryTags.slice(0, 30).map(tag => (
                      <button
                        key={tag.name}
                        className={`tag tag-${category} ${activeTags.includes(tag.name) ? 'active' : ''}`}
                        onClick={() => onTagClick(tag.name)}
                      >
                        <span className="tag-name">{tag.name.replace(/_/g, ' ')}</span>
                        <span className="tag-count">({tag.post_count})</span>
                      </button>
                    ))}
                  </div>
                </div>
              )
            })}
          </>
        )}
      </div>
    </aside>
  )
}

export default Sidebar
