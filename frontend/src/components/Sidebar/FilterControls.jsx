/**
 * FilterControls - Rating, age range, timeframe, and favorites toggle controls
 */
import { useState } from 'react'

// Rating definitions for LocalBooru (all ratings shown)
export const ALL_RATINGS = ['pg', 'pg13', 'r', 'x', 'xxx']

// Sort options - organized by category
export const SORT_OPTIONS = [
  { value: 'newest', label: 'Date (Newest)', group: 'Date' },
  { value: 'oldest', label: 'Date (Oldest)', group: 'Date' },
  { value: 'filename_asc', label: 'Filename (A-Z)', group: 'Name' },
  { value: 'filename_desc', label: 'Filename (Z-A)', group: 'Name' },
  { value: 'filesize_largest', label: 'File Size (Largest)', group: 'Size' },
  { value: 'filesize_smallest', label: 'File Size (Smallest)', group: 'Size' },
  { value: 'resolution_high', label: 'Resolution (Highest)', group: 'Resolution' },
  { value: 'resolution_low', label: 'Resolution (Lowest)', group: 'Resolution' },
  { value: 'duration_longest', label: 'Duration (Longest)', group: 'Duration' },
  { value: 'duration_shortest', label: 'Duration (Shortest)', group: 'Duration' },
  { value: 'random', label: 'Random', group: 'Other' }
]

// Age range constants
export const MIN_AGE_LIMIT = 0
export const MAX_AGE_LIMIT = 80

// Timeframe options
export const TIMEFRAME_OPTIONS = [
  { value: null, label: 'All Time' },
  { value: 'today', label: 'Today' },
  { value: 'week', label: 'This Week' },
  { value: 'month', label: 'This Month' },
  { value: 'year', label: 'This Year' }
]

// Resolution presets (min width/height)
export const RESOLUTION_OPTIONS = [
  { value: null, label: 'Any' },
  { value: { width: 1280, height: 720 }, label: '720p+' },
  { value: { width: 1920, height: 1080 }, label: '1080p+' },
  { value: { width: 2560, height: 1440 }, label: '1440p+' },
  { value: { width: 3840, height: 2160 }, label: '4K+' }
]

// Orientation options
export const ORIENTATION_OPTIONS = [
  { value: null, label: 'Any' },
  { value: 'landscape', label: 'Landscape' },
  { value: 'portrait', label: 'Portrait' },
  { value: 'square', label: 'Square' }
]

// Duration options (in seconds)
export const DURATION_OPTIONS = [
  { value: null, label: 'Any' },
  { value: { min: 0, max: 60 }, label: '<1m' },
  { value: { min: 60, max: 300 }, label: '1-5m' },
  { value: { min: 300, max: 1800 }, label: '5-30m' },
  { value: { min: 1800, max: null }, label: '30m+' }
]

function FilterControls({
  // Directory filter
  directories,
  selectedDirectory,
  onDirectoryChange,
  // Filename search
  filenameSearch,
  setFilenameSearch,
  onFilenameSearchClear,
  onSearchSubmit,
  // Favorites toggle
  favoritesOnly,
  onToggleFavorites,
  // Rating buttons
  selectedRatings,
  onToggleRating,
  // Age range
  minAge,
  maxAge,
  setMinAge,
  setMaxAge,
  onAgeChange,
  // Timeframe
  timeframe,
  onTimeframeChange,
  // Resolution
  resolution,
  onResolutionChange,
  orientation,
  onOrientationChange,
  // Duration
  duration,
  onDurationChange,
  // Search results
  total
}) {
  const [advancedExpanded, setAdvancedExpanded] = useState(false)

  // Count active advanced filters
  const activeAdvancedCount = [
    minAge !== null || maxAge !== null, // Age range modified
    timeframe !== null,                  // Timeframe set
    resolution !== null,                 // Resolution set
    orientation !== null,                // Orientation set
    duration !== null                    // Duration set
  ].filter(Boolean).length

  return (
    <>
      {/* Directory Filter */}
      <div className="directory-filter">
        <div className="directory-filter-list">
          <button
            type="button"
            className={`directory-filter-btn ${!selectedDirectory ? 'active' : ''}`}
            onClick={() => onDirectoryChange('')}
          >
            <svg className="directory-filter-icon" viewBox="0 0 24 24" fill="currentColor">
              <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
            </svg>
            <span className="directory-filter-name">All Directories</span>
          </button>
          {directories.map(dir => (
            <button
              key={dir.id}
              type="button"
              className={`directory-filter-btn ${selectedDirectory === dir.id ? 'active' : ''}`}
              onClick={() => onDirectoryChange(String(dir.id))}
            >
              <svg className="directory-filter-icon" viewBox="0 0 24 24" fill="currentColor">
                <path d="M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/>
              </svg>
              <span className="directory-filter-name">{dir.name}</span>
              {dir.image_count > 0 && (
                <span className="directory-filter-count">{dir.image_count}</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Filename Search */}
      <div className="filename-search">
        <div className="search-input-wrapper">
          <svg className="search-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm4 18H6V4h7v5h5v11z"/>
          </svg>
          <input
            type="text"
            value={filenameSearch}
            onChange={(e) => setFilenameSearch(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                onSearchSubmit(e)
              }
            }}
            placeholder="Search by filename..."
            className="search-input"
          />
          {filenameSearch && (
            <button
              type="button"
              className="search-clear"
              onClick={onFilenameSearchClear}
            >
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* Favorites Toggle */}
      <label className="toggle-row">
        <span className="toggle-label">
          <svg viewBox="0 0 24 24" fill="currentColor" className="toggle-icon">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
          Favorites
        </span>
        <div className={`toggle-switch ${favoritesOnly ? 'active' : ''}`} onClick={onToggleFavorites}>
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
            onClick={() => onToggleRating(rating)}
          >
            {rating.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Advanced Filters Toggle */}
      <button
        type="button"
        className={`advanced-filters-toggle ${advancedExpanded ? 'expanded' : ''}`}
        onClick={() => setAdvancedExpanded(!advancedExpanded)}
      >
        <svg className="advanced-filters-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M3 17v2h6v-2H3zM3 5v2h10V5H3zm10 16v-2h8v-2h-8v-2h-2v6h2zM7 9v2H3v2h4v2h2V9H7zm14 4v-2H11v2h10zm-6-4h2V7h4V5h-4V3h-2v6z"/>
        </svg>
        <span>More Filters</span>
        {activeAdvancedCount > 0 && (
          <span className="advanced-filters-count">{activeAdvancedCount}</span>
        )}
        <svg className="advanced-filters-chevron" viewBox="0 0 24 24" fill="currentColor">
          <path d="M7.41 8.59L12 13.17l4.59-4.58L18 10l-6 6-6-6 1.41-1.41z"/>
        </svg>
      </button>

      {/* Advanced Filters Content */}
      <div className={`advanced-filters-content ${advancedExpanded ? 'expanded' : ''}`}>
        <div className="advanced-filters-inner">
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
                onMouseUp={onAgeChange}
                onTouchEnd={onAgeChange}
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
                onMouseUp={onAgeChange}
                onTouchEnd={onAgeChange}
                className="age-slider age-slider-max"
              />
            </div>
          </div>

          {/* Timeframe Filter */}
          <div className="timeframe-filter">
            <div className="timeframe-buttons">
              {TIMEFRAME_OPTIONS.map(option => {
                const isActive = timeframe === option.value
                return (
                  <button
                    key={option.value || 'all'}
                    type="button"
                    className={`timeframe-btn ${isActive ? 'active' : ''}`}
                    onClick={() => onTimeframeChange(isActive && option.value !== null ? null : option.value)}
                  >
                    {option.label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Resolution Filter */}
          <div className="resolution-filter">
            <span className="filter-label">Min Resolution</span>
            <div className="resolution-buttons">
              {RESOLUTION_OPTIONS.map(option => {
                const isActive = option.value === null
                  ? resolution === null
                  : resolution?.width === option.value?.width && resolution?.height === option.value?.height
                return (
                  <button
                    key={option.label}
                    type="button"
                    className={`resolution-btn ${isActive ? 'active' : ''}`}
                    onClick={() => onResolutionChange(isActive && option.value !== null ? null : option.value)}
                  >
                    {option.label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Orientation Filter */}
          <div className="orientation-filter">
            <span className="filter-label">Orientation</span>
            <div className="orientation-buttons">
              {ORIENTATION_OPTIONS.map(option => {
                const isActive = orientation === option.value
                return (
                  <button
                    key={option.value || 'any'}
                    type="button"
                    className={`orientation-btn ${isActive ? 'active' : ''}`}
                    onClick={() => onOrientationChange(isActive && option.value !== null ? null : option.value)}
                  >
                    {option.label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Duration Filter (for videos) */}
          <div className="duration-filter">
            <span className="filter-label">Video Duration</span>
            <div className="duration-buttons">
              {DURATION_OPTIONS.map(option => {
                const isActive = option.value === null
                  ? duration === null
                  : duration?.min === option.value?.min && duration?.max === option.value?.max
                return (
                  <button
                    key={option.label}
                    type="button"
                    className={`duration-btn ${isActive ? 'active' : ''}`}
                    onClick={() => onDurationChange(isActive && option.value !== null ? null : option.value)}
                  >
                    {option.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>

      {total > 0 && (
        <div className="search-results-count">
          {total.toLocaleString()} results
        </div>
      )}
    </>
  )
}

export default FilterControls
