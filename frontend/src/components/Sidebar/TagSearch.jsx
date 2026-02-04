import { useState, useEffect, useMemo } from 'react'
import { searchTags } from '../../api'
import useDebounce from './hooks/useDebounce'

/**
 * TagSearch - Tag input with autocomplete suggestions
 */
function TagSearch({
  tags,
  activeTags,
  onTagClick
}) {
  const [tagInput, setTagInput] = useState('')
  const [suggestionIndex, setSuggestionIndex] = useState(-1)
  const [suggestions, setSuggestions] = useState([])

  // Debounce tag input for performance (150ms delay)
  const debouncedTagInput = useDebounce(tagInput, 150)

  // Fetch suggestions from API when debounced input changes
  useEffect(() => {
    if (debouncedTagInput.length < 2) {
      setSuggestions([])
      return
    }

    let cancelled = false
    searchTags(debouncedTagInput, 10).then(results => {
      if (!cancelled) {
        // Filter out already active tags
        const filtered = results.filter(tag => !activeTags.includes(tag.name))
        setSuggestions(filtered.slice(0, 8))
      }
    }).catch(err => {
      console.error('Tag search failed:', err)
      if (!cancelled) setSuggestions([])
    })

    return () => { cancelled = true }
  }, [debouncedTagInput, activeTags])

  // Group tags by category (memoized)
  const groupedTags = useMemo(() => (tags || []).reduce((acc, tag) => {
    const category = tag.category || 'general'
    if (!acc[category]) acc[category] = []
    acc[category].push(tag)
    return acc
  }, {}), [tags])

  const categoryOrder = ['artist', 'character', 'copyright', 'general', 'meta']

  return (
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
              // Use API suggestions for keyboard navigation
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
          {/* Autocomplete suggestions from API */}
          <div className={`tag-suggestions ${suggestions.length > 0 ? 'visible' : ''}`}>
            {suggestions.map((tag, index) => (
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
            ))}
          </div>
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
  )
}

export default TagSearch
