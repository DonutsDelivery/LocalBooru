/**
 * FilterControls - Rating, age range, timeframe, and favorites toggle controls
 */

// Rating definitions for LocalBooru (all ratings shown)
export const ALL_RATINGS = ['pg', 'pg13', 'r', 'x', 'xxx']

// Sort options
export const SORT_OPTIONS = [
  { value: 'newest', label: 'Newest' },
  { value: 'oldest', label: 'Oldest' },
  { value: 'random', label: 'Random' }
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
  // Sort controls
  sortBy,
  onSortChange,
  // Timeframe
  timeframe,
  onTimeframeChange,
  // Search results
  total
}) {
  return (
    <>
      {/* Directory Filter */}
      <div className="directory-filter">
        <select
          value={selectedDirectory || ''}
          onChange={(e) => onDirectoryChange(e.target.value)}
          className="directory-select"
        >
          <option value="">All Directories</option>
          {directories.map(dir => (
            <option key={dir.id} value={dir.id}>
              {dir.name}{dir.image_count > 0 ? ` (${dir.image_count})` : ''}
            </option>
          ))}
        </select>
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

      {/* Sort Controls */}
      <div className="sort-controls">
        <div className="sort-buttons">
          {SORT_OPTIONS.map(option => (
            <button
              key={option.value}
              type="button"
              className={`sort-btn ${sortBy === option.value ? 'active' : ''}`}
              onClick={() => onSortChange(option.value)}
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
              onClick={() => onTimeframeChange(option.value)}
            >
              {option.label}
            </button>
          ))}
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
