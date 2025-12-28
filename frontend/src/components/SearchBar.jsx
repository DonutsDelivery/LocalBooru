import { useState, useEffect } from 'react'
import './SearchBar.css'

function SearchBar({ onSearch, initialTags, initialRating, total }) {
  const [tags, setTags] = useState(initialTags || '')
  const [rating, setRating] = useState(initialRating || '')
  // SFW mode: when on, only show safe-rated content
  const [sfwMode, setSfwMode] = useState(() => {
    return localStorage.getItem('donutbooru_sfw_mode') === 'true'
  })

  useEffect(() => {
    setTags(initialTags || '')
    // If SFW mode is on, force rating to safe
    if (sfwMode) {
      setRating('safe')
    } else {
      setRating(initialRating || '')
    }
  }, [initialTags, initialRating, sfwMode])

  const handleSubmit = (e) => {
    e.preventDefault()
    // If SFW mode is on, always filter by safe
    onSearch(tags, sfwMode ? 'safe' : rating)
  }

  const handleClear = () => {
    setTags('')
    if (!sfwMode) {
      setRating('')
    }
    onSearch('', sfwMode ? 'safe' : '')
  }

  const toggleSfwMode = () => {
    const newMode = !sfwMode
    setSfwMode(newMode)
    localStorage.setItem('donutbooru_sfw_mode', newMode.toString())
    // Immediately search with new mode
    onSearch(tags, newMode ? 'safe' : '')
  }

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <div className="search-input-wrapper">
        <svg className="search-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
        </svg>
        <input
          type="text"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="Search tags... (comma separated)"
          className="search-input"
        />
        {(tags || rating) && (
          <button type="button" className="search-clear" onClick={handleClear}>
            <svg viewBox="0 0 24 24" fill="currentColor">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        )}
      </div>

      {/* SFW/NSFW Toggle */}
      <button
        type="button"
        className={`sfw-toggle ${sfwMode ? 'sfw-on' : 'nsfw-on'}`}
        onClick={toggleSfwMode}
        title={sfwMode ? 'SFW Mode: Only safe content shown' : 'NSFW Mode: All content shown'}
      >
        {sfwMode ? 'SFW' : 'NSFW'}
      </button>

      {/* Rating dropdown (hidden when SFW mode is on) */}
      {!sfwMode && (
        <select
          value={rating}
          onChange={(e) => setRating(e.target.value)}
          className="search-rating"
        >
          <option value="">All Ratings</option>
          <option value="safe">Safe</option>
          <option value="questionable">Questionable</option>
          <option value="explicit">Explicit</option>
        </select>
      )}

      <button type="submit" className="search-button">
        Search
      </button>

      {total > 0 && (
        <span className="search-results-count">
          {total.toLocaleString()} results
        </span>
      )}
    </form>
  )
}

export default SearchBar
