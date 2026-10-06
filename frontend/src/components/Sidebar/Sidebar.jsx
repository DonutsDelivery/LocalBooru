import { useState, useEffect, useMemo, useCallback } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { fetchCollections, fetchDirectories, getFileDimensions, getSavedSearches, createSavedSearch, deleteSavedSearch } from '../../api'
import PromptSection from './PromptSection'
import FilterControls, { ALL_RATINGS, MIN_AGE_LIMIT, MAX_AGE_LIMIT, RESOLUTION_OPTIONS, ORIENTATION_OPTIONS, DURATION_OPTIONS, SORT_OPTIONS } from './FilterControls'
import TagSearch from './TagSearch'
import SidebarNavigation from '../SidebarNavigation'
import { collectionGalleryTarget } from '../../utils/collectionState'
import FamilyModeLock from '../FamilyModeLock'
import PersistentMusicPlayer from '../Music/PersistentMusicPlayer'
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
  initialLibraryId,
  initialMinAge,
  initialMaxAge,
  initialSort,
  initialTimeframe,
  initialFilename,
  initialResolution,
  initialOrientation,
  initialDuration,
  initialGroupByFolders,
  mediaType = 'image',
  initialWatchedStatus,
  onToggleGroupByFolders,
  total,
  stats,
  lightboxMode,
  lightboxHover,
  onMouseLeave,
  settingsTab,
  onSettingsTabChange,
  onFamilyModeChange,
  collectionItems,
  currentCollection,
  onCreateCollection,
  onRenameCollection,
  filterScope = 'local',
  availableRatings = ALL_RATINGS,
  supportedSortValues,
  defaultSort = 'newest',
  sidebarActions,
  sidebarCollections,
  searchSuggestions
}) {
  const location = useLocation()
  const isGalleryPage = location.pathname === '/' || location.pathname === '/videos'
  const isCollectionPage = location.pathname === '/collections' || location.pathname.startsWith('/collections/') || location.pathname.startsWith('/shared/')
  const isSearchPage = isGalleryPage || (isCollectionPage && Boolean(onSearch))
  const isRemote = filterScope === 'remote'
  const galleryReturnTo = collectionGalleryTarget(location, mediaType)
  const collectionNavigationState = { galleryReturnTo }
  const galleryLabel = mediaType === 'video' ? 'Videos' : 'Images'
  const isSettingsPage = location.pathname === '/settings' || location.pathname.startsWith('/settings/')
  const [hovering, setHovering] = useState(false)
  const [directories, setDirectories] = useState([])
  const [directoriesMediaType, setDirectoriesMediaType] = useState(null)
  const [selectedDirectory, setSelectedDirectory] = useState(initialDirectoryId || null)
  const [selectedLibrary, setSelectedLibrary] = useState(initialLibraryId || null)
  const [selectedRatings, setSelectedRatings] = useState(() => {
    if (initialRating) {
      let ratings = initialRating.split(',').filter(r => availableRatings.includes(r))
      if (ratings.length > 0) return ratings
    }
    return [...availableRatings]
  })
  const [sortBy, setSortBy] = useState(initialSort || defaultSort)
  const [favoritesOnly, setFavoritesOnly] = useState(initialFavoritesOnly || false)
  const [minAge, setMinAge] = useState(initialMinAge || null)
  const [maxAge, setMaxAge] = useState(initialMaxAge || null)
  const [timeframe, setTimeframe] = useState(initialTimeframe || null)
  const [filenameSearch, setFilenameSearch] = useState(initialFilename || '')
  const [resolution, setResolution] = useState(initialResolution || null)
  const [orientation, setOrientation] = useState(initialOrientation || null)
  const [duration, setDuration] = useState(initialDuration || null)
  const [watchedStatus, setWatchedStatus] = useState(initialWatchedStatus || null)
  const [collections, setCollections] = useState([])
  const [fetchedDimensions, setFetchedDimensions] = useState(null)
  const [copiedField, setCopiedField] = useState(null)
  const [filtersExpanded, setFiltersExpanded] = useState(() => {
    const saved = localStorage.getItem('filtersExpanded')
    return saved !== null ? JSON.parse(saved) : false
  })

  // Family mode state
  const [familyLocked, setFamilyLocked] = useState(false)
  const SFW_RATINGS = ['pg', 'pg13']

  // Saved searches state
  const [savedSearches, setSavedSearches] = useState([])
  const [showSaveInput, setShowSaveInput] = useState(false)
  const [saveSearchName, setSaveSearchName] = useState('')

  // Load directories
  const refreshDirectories = useCallback(requestConfig => {
    if (isRemote) return Promise.resolve()
    return fetchDirectories(false, null, mediaType, requestConfig).then(data => {
      if (requestConfig?.signal?.aborted) return
      setDirectories(data.directories || [])
      setDirectoriesMediaType(mediaType)
    }).catch(error => { if (!requestConfig?.signal?.aborted) console.error(error) })
  }, [mediaType, isRemote])

  useEffect(() => {
    refreshDirectories()
  }, [refreshDirectories])

  useEffect(() => {
    if (mediaType !== 'image') return
    let request = null
    const created = () => {
      request?.abort()
      request = new AbortController()
      refreshDirectories({ signal: request.signal })
    }
    const changingServer = () => request?.abort()
    window.addEventListener('donut-create-directory-created', created)
    window.addEventListener('donut-create-server-changing', changingServer)
    return () => {
      request?.abort()
      window.removeEventListener('donut-create-directory-created', created)
      window.removeEventListener('donut-create-server-changing', changingServer)
    }
  }, [mediaType, refreshDirectories])

  useEffect(() => {
    if (!isCollectionPage || collectionItems) return
    let active = true
    fetchCollections(mediaType).then(data => {
      if (active) setCollections(data.collections || [])
    }).catch(console.error)
    return () => { active = false }
  }, [isCollectionPage, collectionItems, mediaType])

  // Load saved searches
  useEffect(() => {
    if (isRemote) return
    getSavedSearches().then(data => {
      setSavedSearches((data.searches || []).filter(search => !search.filters?.remote_source_id && (search.filters?.media_type || 'image') === mediaType))
    }).catch(console.error)
  }, [mediaType, isRemote])

  // Persist filters expanded state
  useEffect(() => {
    localStorage.setItem('filtersExpanded', JSON.stringify(filtersExpanded))
  }, [filtersExpanded])

  useEffect(() => {
    const ratings = initialRating?.split(',').filter(r => availableRatings.includes(r)) || []
    setSelectedRatings(ratings.length ? ratings : [...availableRatings])
    setFavoritesOnly(initialFavoritesOnly || false)
    setSelectedDirectory(initialDirectoryId || null)
    setSelectedLibrary(initialLibraryId || null)
    setMinAge(initialMinAge || null)
    setMaxAge(initialMaxAge || null)
    setTimeframe(initialTimeframe || null)
    setFilenameSearch(initialFilename || '')
    setResolution(initialResolution || null)
    setOrientation(initialOrientation || null)
    setDuration(initialDuration || null)
    setWatchedStatus(initialWatchedStatus || null)
    setSortBy(initialSort || defaultSort)
  }, [initialRating, initialFavoritesOnly, initialDirectoryId, initialLibraryId, initialMinAge, initialMaxAge, initialSort, initialTimeframe, initialFilename, initialResolution, initialOrientation, initialDuration, initialWatchedStatus, availableRatings, defaultSort])

  // Fetch dimensions for selected image when it changes
  useEffect(() => {
    if (selectedImage?._remote) {
      setFetchedDimensions(selectedImage.width > 0 && selectedImage.height > 0
        ? { width: selectedImage.width, height: selectedImage.height } : null)
      return
    }
    if (!selectedImage || !selectedImage.file_path) {
      setFetchedDimensions(null)
      return
    }

    getFileDimensions(selectedImage.file_path)
      .then(result => {
        if (result.success) {
          setFetchedDimensions({
            width: result.width,
            height: result.height,
            fps: result.fps || null
          })
        } else {
          setFetchedDimensions(null)
        }
      })
      .catch(err => {
        console.error('Failed to fetch dimensions:', err)
        setFetchedDimensions(null)
      })
  }, [selectedImage?.id, selectedImage?.file_path, selectedImage?._remote, selectedImage?.width, selectedImage?.height])

  const copyInfoValue = useCallback((label, value) => {
    navigator.clipboard.writeText(String(value)).then(() => {
      setCopiedField(label)
      setTimeout(() => setCopiedField(null), 1500)
    }).catch(() => {})
  }, [])

  const isVisible = !collapsed || hovering || mobileOpen

  // Handle family mode lock state changes
  const handleFamilyLockChange = useCallback((isLocked) => {
    setFamilyLocked(isLocked)
    refreshDirectories()
    if (isLocked) {
      // Force SFW-only ratings and trigger search
      setSelectedRatings(SFW_RATINGS.filter(rating => availableRatings.includes(rating)))
      if (onSearch) {
        onSearch(currentTags || '', SFW_RATINGS.filter(rating => availableRatings.includes(rating)).join(','), sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, selectedLibrary)
      }
    } else {
      // Restore all ratings on unlock
      setSelectedRatings([...availableRatings])
      if (onSearch) {
        onSearch(currentTags || '', availableRatings.join(','), sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, selectedLibrary)
      }
    }
  }, [refreshDirectories, onSearch, currentTags, sortBy, favoritesOnly, selectedDirectory, selectedLibrary, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, availableRatings])

  // Memoize activeTags to prevent unnecessary re-renders and useEffect triggers
  const activeTags = useMemo(() =>
    currentTags ? currentTags.split(',').map(t => t.trim()) : []
  , [currentTags])

  // Get effective ratings (filter out NSFW when family mode locked)
  const getEffectiveRatings = () => {
    const allowed = selectedRatings.filter(rating => availableRatings.includes(rating))
    const ratings = familyLocked
      ? allowed.filter(r => !['r', 'x', 'xxx'].includes(r))
      : allowed
    return ratings.length > 0 ? ratings : (familyLocked ? SFW_RATINGS.filter(rating => availableRatings.includes(rating)) : [...availableRatings])
  }

  const handleSearchSubmit = (e) => {
    e.preventDefault()
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, selectedLibrary)
  }

  const handleClear = () => {
    setFilenameSearch('')
    const defaultRatings = familyLocked ? SFW_RATINGS.filter(rating => availableRatings.includes(rating)) : [...availableRatings]
    setSelectedRatings(defaultRatings)
    setSortBy(defaultSort)
    setFavoritesOnly(false)
    setSelectedDirectory(null)
    setSelectedLibrary(null)
    setMinAge(null)
    setMaxAge(null)
    setTimeframe(null)
    setResolution(null)
    setOrientation(null)
    setDuration(null)
    setWatchedStatus(null)
    if (initialGroupByFolders && onToggleGroupByFolders) {
      onToggleGroupByFolders()
    }
    onSearch('', defaultRatings.join(','), defaultSort, false, null, null, null, null, '', null, null, null, null, null)
  }

  // Save current filters as a named search
  const handleSaveSearch = async () => {
    if (!saveSearchName.trim()) return
    const filters = {
      media_type: mediaType,
      tags: currentTags || '',
      rating: selectedRatings.join(','),
      sort: sortBy,
      favorites_only: favoritesOnly,
      directory_id: selectedDirectory,
      library_id: selectedLibrary,
      min_age: minAge,
      max_age: maxAge,
      timeframe,
      filename: filenameSearch,
      orientation,
      watched_status: watchedStatus,
    }
    // Only include non-null resolution/duration
    if (resolution) filters.resolution = resolution
    if (duration) filters.duration = duration

    try {
      const result = await createSavedSearch(saveSearchName.trim(), filters)
      setSavedSearches(prev => [...prev, result.search])
      setShowSaveInput(false)
      setSaveSearchName('')
    } catch (e) {
      console.error('Failed to save search:', e)
    }
  }

  // Apply a saved search
  const handleApplySavedSearch = (search) => {
    const f = search.filters || {}
    const ratings = f.rating || availableRatings.join(',')
    setSelectedRatings(ratings.split(',').filter(r => availableRatings.includes(r)))
    setSortBy(f.sort || defaultSort)
    setFavoritesOnly(f.favorites_only || false)
    setSelectedDirectory(f.directory_id || null)
    setSelectedLibrary(f.library_id || null)
    setMinAge(f.min_age || null)
    setMaxAge(f.max_age || null)
    setTimeframe(f.timeframe || null)
    setFilenameSearch(f.filename || '')
    setResolution(f.resolution || null)
    setOrientation(f.orientation || null)
    setDuration(f.duration || null)
    setWatchedStatus(f.watched_status || null)
    onSearch(
      f.tags || '',
      ratings,
      f.sort || defaultSort,
      f.favorites_only || false,
      f.directory_id || null,
      f.min_age || null,
      f.max_age || null,
      f.timeframe || null,
      f.filename || '',
      f.resolution || null,
      f.orientation || null,
      f.duration || null,
      f.library_id || null,
      f.watched_status || null
    )
  }

  // Delete a saved search
  const handleDeleteSavedSearch = async (e, searchId) => {
    e.stopPropagation()
    try {
      await deleteSavedSearch(searchId)
      setSavedSearches(prev => prev.filter(s => s.id !== searchId))
    } catch (e) {
      console.error('Failed to delete saved search:', e)
    }
  }

  const handleSortChange = (newSortBy) => {
    setSortBy(newSortBy)
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, newSortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, selectedLibrary)
  }

  const handleDirectoryChange = (compositeId) => {
    let newDirId = null
    let newLibId = null
    if (compositeId !== '') {
      const parts = compositeId.split(':')
      newLibId = parts[0] || null
      newDirId = parts[1] ? parseInt(parts[1]) : null
    }
    setSelectedDirectory(newDirId)
    setSelectedLibrary(newLibId)
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, newDirId, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, newLibId)
  }

  // A saved gallery URL may still reference a directory whose media type was changed.
  useEffect(() => {
    if (!isGalleryPage || directoriesMediaType !== mediaType || !initialDirectoryId) return
    const selectedDirectoryExists = directories.some(dir =>
      String(dir.id) === String(initialDirectoryId) &&
      (!initialLibraryId || dir.library_id === initialLibraryId)
    )
    if (!selectedDirectoryExists) handleDirectoryChange('')
  }, [directories, directoriesMediaType, mediaType, isGalleryPage, initialDirectoryId, initialLibraryId])

  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe)
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, newTimeframe, filenameSearch, resolution, orientation, duration, selectedLibrary)
  }

  const handleResolutionChange = (newResolution) => {
    setResolution(newResolution)
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, newResolution, orientation, duration, selectedLibrary)
  }

  const handleOrientationChange = (newOrientation) => {
    setOrientation(newOrientation)
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, newOrientation, duration, selectedLibrary)
  }

  const handleDurationChange = (newDuration) => {
    setDuration(newDuration)
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, newDuration, selectedLibrary)
  }

  const toggleRating = (rating) => {
    setSelectedRatings(prev => {
      const isSelected = prev.includes(rating)
      if (isSelected) {
        const newRatings = prev.filter(r => r !== rating)
        if (newRatings.length === 0) return familyLocked ? SFW_RATINGS.filter(rating => availableRatings.includes(rating)) : [...availableRatings]
        return newRatings
      }
      return [...prev, rating]
    })
  }

  const toggleFavorites = () => {
    const newFavOnly = !favoritesOnly
    setFavoritesOnly(newFavOnly)
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, newFavOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, selectedLibrary)
  }

  const handleFilenameSearchClear = () => {
    setFilenameSearch('')
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, '', resolution, orientation, duration, selectedLibrary)
  }

  const handleAgeChange = () => {
    const ratingParam = getEffectiveRatings().join(',')
    onSearch(currentTags || '', ratingParam, sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, selectedLibrary)
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
      {(mobileOpen || (lightboxMode && lightboxHover)) && (
        <button className="sidebar-close" onClick={onClose} aria-label="Close menu">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
          </svg>
        </button>
      )}

      <div className="sidebar-content">
        <SidebarNavigation />

        {location.pathname === '/' && mediaType === 'image' && (
          <div className="sidebar-section create-entry-section">
            <button type="button" className="sidebar-create-button" onClick={() => {
              window.dispatchEvent(new CustomEvent('donut-create-open'))
              if (onClose) onClose()
            }}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="m3 18 10-10 4 4L7 22zM13 8l3-3 4 4-3 3M6 3v4M4 5h4M19 17v4M17 19h4" /></svg>
              Create
            </button>
          </div>
        )}

        {isGalleryPage && (
          <div className="sidebar-section media-collections-section">
            <NavLink to={`/collections?media_type=${mediaType}`} state={collectionNavigationState} className="sidebar-create-button collections-entry-button" onClick={onClose}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><rect x="7" y="7" width="14" height="14" rx="2"/><path d="M17 7V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h2"/></svg>
              Collections
            </NavLink>
          </div>
        )}

        {isCollectionPage && (
          <div className="sidebar-section media-collections-section">
            <NavLink to={galleryReturnTo} className="media-collection-link" onClick={onClose}>← Back to {galleryLabel}</NavLink>
            <h3>{mediaType === 'video' ? 'Video' : 'Image'} Collections</h3>
            <NavLink end to={`/collections?media_type=${mediaType}`} state={collectionNavigationState} className="media-collection-link" onClick={onClose}>All Collections</NavLink>
            {onCreateCollection && <button type="button" className="media-collection-link collection-sidebar-action" onClick={() => { onCreateCollection(); onClose?.() }}>＋ New Collection</button>}
            {onRenameCollection && <button type="button" className="media-collection-link collection-sidebar-action" onClick={() => { onRenameCollection(); onClose?.() }}>Rename Collection</button>}
            {(collectionItems || collections).map(item => {
              const collection = currentCollection?.id === item.id ? currentCollection : item
              return <NavLink
                key={collection.id}
                to={`/collections/${collection.id}?media_type=${mediaType}`}
                state={collectionNavigationState}
                className="media-collection-link"
                onClick={onClose}
              >
                <span>{collection.name}</span>
                <span>{collection.item_count}</span>
              </NavLink>
            })}
          </div>
        )}

        {sidebarActions && <div className="sidebar-section collection-sidebar-actions">{sidebarActions}</div>}

        {/* Settings Navigation */}
        {isSettingsPage && onSettingsTabChange && (
          <div className="sidebar-section settings-nav">
            {[['directories','Watch Directories'], ['general','General'], ['video','Video'], ['network','Network'],
              ['servers','Servers'], ['fediverse','Fediverse & Nodes'], ['mobile','Mobile'], ['data','Data'], ['addons','Addons'], ['addon-settings','Add-on Settings'], ['tasks','Tasks']
            ].map(([key, label]) => (
              <button key={key}
                className={`settings-nav-item ${settingsTab === key ? 'active' : ''}`}
                onClick={() => { onSettingsTabChange(key); if (onClose) onClose() }}
              >{label}</button>
            ))}
          </div>
        )}

        {/* Saved Searches */}
        {isSearchPage && !isRemote && savedSearches.length > 0 && (
          <div className="sidebar-section saved-searches-section">
            <div className="saved-searches-chips">
              {savedSearches.map(search => (
                <button
                  key={search.id}
                  className="saved-search-chip"
                  onClick={() => handleApplySavedSearch(search)}
                  title={search.name}
                >
                  <span>{search.name}</span>
                  <span className="saved-search-delete" onClick={(e) => handleDeleteSavedSearch(e, search.id)}>
                    <svg viewBox="0 0 24 24" fill="currentColor" width="12" height="12"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Search Section - only show on gallery page */}
        {isSearchPage && <div className="sidebar-section search-section">
          <form onSubmit={handleSearchSubmit}>
            {/* Sort dropdown - always visible */}
            <div className="sort-controls">
              <span className="filter-label">Sort By</span>
              <select
                value={sortBy}
                onChange={(e) => handleSortChange(e.target.value)}
                className="sort-select"
                aria-label="Sort by"
              >
                {[...(defaultSort === 'collection' ? [{ value: 'collection', label: 'Collection Order' }] : []), ...SORT_OPTIONS].filter(option => (mediaType === 'video' || option.group !== 'Duration') && (!supportedSortValues || supportedSortValues.includes(option.value))).map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

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
                  mediaType={mediaType}
                  availableRatings={availableRatings}
                  showDirectories={!isRemote}
                  showFilename={!isRemote}
                  showFolders={isGalleryPage && Boolean(onToggleGroupByFolders)}
                  showAge={!isRemote}
                  showWatched={!isRemote}
                  directories={directories}
                  selectedDirectory={selectedDirectory}
                  selectedLibrary={selectedLibrary}
                  onDirectoryChange={handleDirectoryChange}
                  filenameSearch={filenameSearch}
                  setFilenameSearch={setFilenameSearch}
                  onFilenameSearchClear={handleFilenameSearchClear}
                  onSearchSubmit={handleSearchSubmit}
                  favoritesOnly={favoritesOnly}
                  onToggleFavorites={toggleFavorites}
                  groupByFolders={initialGroupByFolders}
                  onToggleGroupByFolders={onToggleGroupByFolders}
                  selectedRatings={selectedRatings}
                  onToggleRating={toggleRating}
                  minAge={minAge}
                  maxAge={maxAge}
                  setMinAge={setMinAge}
                  setMaxAge={setMaxAge}
                  onAgeChange={handleAgeChange}
                  timeframe={timeframe}
                  onTimeframeChange={handleTimeframeChange}
                  resolution={resolution}
                  onResolutionChange={handleResolutionChange}
                  orientation={orientation}
                  onOrientationChange={handleOrientationChange}
                  duration={duration}
                  onDurationChange={handleDurationChange}
                  watchedStatus={watchedStatus}
                  onWatchedStatusChange={(status) => {
                    setWatchedStatus(status)
                    onSearch(currentTags || '', getEffectiveRatings().join(','), sortBy, favoritesOnly, selectedDirectory, minAge, maxAge, timeframe, filenameSearch, resolution, orientation, duration, selectedLibrary, status)
                  }}
                  total={0}
                  familyLocked={familyLocked}
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
              {!isRemote && <button type="button" className="save-search-button" onClick={() => setShowSaveInput(!showSaveInput)} title="Save current filters">
                <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                  <path d="M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm-5 16c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3zm3-10H5V5h10v4z"/>
                </svg>
              </button>}
            </div>
            {!isRemote && showSaveInput && (
              <div className="save-search-input">
                <input
                  type="text"
                  placeholder="Search name..."
                  value={saveSearchName}
                  onChange={(e) => setSaveSearchName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleSaveSearch() } }}
                  autoFocus
                />
                <button onClick={handleSaveSearch} disabled={!saveSearchName.trim()}>Save</button>
              </div>
            )}
          </form>
        </div>}

        {/* Selected Image Info */}
        {(isSearchPage || selectedImage?._remote) && selectedImage && (
          <div className="sidebar-section image-info">
            <h3>Media Info</h3>
            <div className="info-grid">
              <span className="info-label">ID</span>
              <span className="info-value copyable" onClick={() => copyInfoValue('ID', selectedImage._remote ? selectedImage.remote_post_id : selectedImage.id)} title="Click to copy">
                {copiedField === 'ID' ? '✓ Copied' : `#${selectedImage._remote ? selectedImage.remote_post_id : selectedImage.id}`}
              </span>

              {selectedImage._remote && selectedImage.remote_source_name && (
                <><span className="info-label">Community</span><span className="info-value">{selectedImage.remote_source_name}</span></>
              )}

              {fetchedDimensions ? (
                <>
                  <span className="info-label">Resolution</span>
                  <span className="info-value copyable" onClick={() => copyInfoValue('Resolution', `${fetchedDimensions.width}x${fetchedDimensions.height}`)} title="Click to copy">
                    {copiedField === 'Resolution' ? '✓ Copied' : `${fetchedDimensions.width}x${fetchedDimensions.height}`}
                  </span>
                </>
              ) : null}

              {fetchedDimensions?.fps && (
                <>
                  <span className="info-label">FPS</span>
                  <span className="info-value copyable" onClick={() => copyInfoValue('FPS', Number.isInteger(fetchedDimensions.fps) ? fetchedDimensions.fps : fetchedDimensions.fps.toFixed(2))} title="Click to copy">
                    {copiedField === 'FPS' ? '✓ Copied' : (Number.isInteger(fetchedDimensions.fps) ? fetchedDimensions.fps : fetchedDimensions.fps.toFixed(2))}
                  </span>
                </>
              )}

              {mediaType === 'video' && Number.isFinite(selectedImage.duration) && (
                <>
                  <span className="info-label">Duration</span>
                  <span className="info-value">{Math.floor(selectedImage.duration / 60)}:{String(Math.floor(selectedImage.duration % 60)).padStart(2, '0')}</span>
                </>
              )}

              <span className="info-label">Rating</span>
              <span className={`info-value copyable rating-${selectedImage.rating}`} onClick={() => copyInfoValue('Rating', selectedImage.rating)} title="Click to copy">
                {copiedField === 'Rating' ? '✓ Copied' : selectedImage.rating}
              </span>

              {selectedImage.created_at && (
                <>
                  <span className="info-label">Added</span>
                  <span className="info-value copyable" onClick={() => copyInfoValue('Added', new Date(selectedImage.created_at).toLocaleDateString())} title="Click to copy">
                    {copiedField === 'Added' ? '✓ Copied' : new Date(selectedImage.created_at).toLocaleDateString()}
                  </span>
                </>
              )}

              {selectedImage.file_size && (
                <>
                  <span className="info-label">File Size</span>
                  <span className="info-value copyable" onClick={() => copyInfoValue('File Size', selectedImage.file_size >= 1024 * 1024 ? `${(selectedImage.file_size / (1024 * 1024)).toFixed(1)} MB` : `${(selectedImage.file_size / 1024).toFixed(0)} KB`)} title="Click to copy">
                    {copiedField === 'File Size' ? '✓ Copied' : (selectedImage.file_size >= 1024 * 1024
                      ? `${(selectedImage.file_size / (1024 * 1024)).toFixed(1)} MB`
                      : `${(selectedImage.file_size / 1024).toFixed(0)} KB`)}
                  </span>
                </>
              )}

              {selectedImage.file_path && (
                <>
                  <span className="info-label">Format</span>
                  <span className="info-value copyable" onClick={() => copyInfoValue('Format', selectedImage.file_path.split('.').pop()?.toUpperCase())} title="Click to copy">
                    {copiedField === 'Format' ? '✓ Copied' : selectedImage.file_path.split('.').pop()?.toUpperCase()}
                  </span>
                </>
              )}

              {selectedImage.directory_name && (
                <>
                  <span className="info-label">Directory</span>
                  <span className="info-value copyable" onClick={() => copyInfoValue('Directory', selectedImage.directory_name)} title="Click to copy">
                    {copiedField === 'Directory' ? '✓ Copied' : selectedImage.directory_name}
                  </span>
                </>
              )}

              {selectedImage.file_path && (
                <>
                  <span className="info-label">Path</span>
                  <span className="info-value copyable file-path" onClick={() => copyInfoValue('Path', selectedImage.file_path)} title="Click to copy">
                    {copiedField === 'Path' ? '✓ Copied' : selectedImage.file_path}
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

              {mediaType === 'image' && selectedImage.num_faces !== null && selectedImage.num_faces !== undefined && (
                <>
                  <span className="info-label">Faces</span>
                  <span className="info-value copyable" onClick={() => copyInfoValue('Faces', selectedImage.num_faces)} title="Click to copy">
                    {copiedField === 'Faces' ? '✓ Copied' : selectedImage.num_faces}
                  </span>

                  {selectedImage.min_age && (
                    <>
                      <span className="info-label">Age</span>
                      <span className="info-value copyable" onClick={() => copyInfoValue('Age', selectedImage.min_age === selectedImage.max_age ? selectedImage.min_age : `${selectedImage.min_age}-${selectedImage.max_age}`)} title="Click to copy">
                        {copiedField === 'Age' ? '✓ Copied' : (selectedImage.min_age === selectedImage.max_age
                          ? `${selectedImage.min_age}`
                          : `${selectedImage.min_age}-${selectedImage.max_age}`)}
                      </span>
                    </>
                  )}
                </>
              )}
            </div>

            {selectedImage._remote && selectedImage.title && (
              <div className="image-caption"><h4>Title</h4><p>{selectedImage.title}</p></div>
            )}

            {mediaType === 'image' && selectedImage.tags?.length > 0 && (
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
            {mediaType === 'image' && (selectedImage.prompt || selectedImage.negative_prompt) && (
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

        {sidebarCollections && <div className="sidebar-section sidebar-collections">{sidebarCollections}</div>}

        {/* Tag Browser */}
        {isSearchPage && (mediaType === 'image' || isCollectionPage) && !selectedImage && (
          <TagSearch
            tags={tags}
            activeTags={activeTags}
            searchSuggestions={isRemote ? searchSuggestions || null : undefined}
            onTagClick={onTagClick}
          />
        )}

        {/* Family mode lock/unlock */}
        <FamilyModeLock onStateChange={onFamilyModeChange} onLockChange={handleFamilyLockChange} />

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
      <PersistentMusicPlayer />
    </aside>
  )
}

export default Sidebar
