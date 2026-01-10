/**
 * LocalBooru - Local image library with auto-tagging
 * Simplified single-user version
 */
import { useState, useEffect, useCallback, useRef } from 'react'
import { BrowserRouter, Routes, Route, useSearchParams } from 'react-router-dom'
import MasonryGrid from './components/MasonryGrid'
import Sidebar from './components/Sidebar'
import Lightbox from './components/Lightbox'
import TitleBar from './components/TitleBar'
import ComfyUIConfigModal from './components/ComfyUIConfigModal'
import NetworkSettings from './components/NetworkSettings'
import { fetchImages, fetchTags, getLibraryStats, subscribeToLibraryEvents, updateDirectory, batchDeleteImages, batchRetag, batchAgeDetect, batchMoveImages, fetchDirectories } from './api'
import './App.css'


// Directory management page
function DirectoriesPage() {
  const [directories, setDirectories] = useState([])
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState({})
  const [pruning, setPruning] = useState({})
  const [comfyuiConfigDir, setComfyuiConfigDir] = useState(null)
  const [stats, setStats] = useState(null)

  const refreshDirectories = async () => {
    const { fetchDirectories } = await import('./api')
    const data = await fetchDirectories()
    setDirectories(data.directories || [])
  }

  useEffect(() => {
    refreshDirectories()
      .catch(console.error)
      .finally(() => setLoading(false))
    getLibraryStats().then(setStats).catch(console.error)
  }, [])

  const handleAddDirectory = async () => {
    if (window.electronAPI) {
      const path = await window.electronAPI.addDirectory()
      if (path) {
        const { addDirectory } = await import('./api')
        await addDirectory(path)
        await refreshDirectories()
      }
    } else {
      alert('Directory picker only available in Electron app')
    }
  }

  const handleAddParentDirectory = async () => {
    if (window.electronAPI) {
      const path = await window.electronAPI.addDirectory()
      if (path) {
        const { addParentDirectory } = await import('./api')
        const result = await addParentDirectory(path)
        alert(result.message)
        await refreshDirectories()
      }
    } else {
      alert('Directory picker only available in Electron app')
    }
  }

  const handleRescan = async (dirId) => {
    setScanning(prev => ({ ...prev, [dirId]: true }))
    try {
      const { scanDirectory } = await import('./api')
      await scanDirectory(dirId)
      await refreshDirectories()
    } catch (error) {
      console.error('Scan failed:', error)
      alert('Scan failed: ' + error.message)
    } finally {
      setScanning(prev => ({ ...prev, [dirId]: false }))
    }
  }

  const handleRemove = async (dirId, dirName) => {
    if (!confirm(`Remove "${dirName}" from watch list?\n\nImages will be removed from library.\nActual files on disk will NOT be deleted.`)) {
      return
    }
    try {
      const { removeDirectory } = await import('./api')
      await removeDirectory(dirId, false)
      await refreshDirectories()
    } catch (error) {
      console.error('Remove failed:', error)
      alert('Remove failed: ' + error.message)
    }
  }

  const handlePrune = async (dirId, dirName, favoritedCount) => {
    const nonFavorited = directories.find(d => d.id === dirId)?.image_count - favoritedCount
    const savedDumpsterPath = localStorage.getItem('localbooru_dumpster_path') || null
    const dumpsterInfo = savedDumpsterPath ? `\nDumpster: ${savedDumpsterPath}` : ''
    if (!confirm(`Prune "${dirName}"?\n\nThis will move ${nonFavorited} non-favorited images to the dumpster folder.\nFavorited images (${favoritedCount}) will be kept.${dumpsterInfo}`)) {
      return
    }
    setPruning(prev => ({ ...prev, [dirId]: true }))
    try {
      const { pruneDirectory } = await import('./api')
      const result = await pruneDirectory(dirId, savedDumpsterPath)
      alert(`Pruned ${result.pruned} images to:\n${result.dumpster_path}`)
      await refreshDirectories()
      getLibraryStats().then(setStats).catch(console.error)
    } catch (error) {
      console.error('Prune failed:', error)
      alert('Prune failed: ' + error.message)
    } finally {
      setPruning(prev => ({ ...prev, [dirId]: false }))
    }
  }

  return (
    <div className="app">
      <div className="main-container">
        <Sidebar stats={stats} />
        <main className="content with-sidebar">
          <div className="page directories-page">
            <h1>Watch Directories</h1>
            <p>Add folders to automatically import and tag images.</p>

            <div className="directory-buttons">
              <button onClick={handleAddDirectory} className="add-directory-btn">
                + Add Directory
              </button>
              <button onClick={handleAddParentDirectory} className="add-directory-btn">
                + Add Parent Directory
              </button>
            </div>

            {loading ? (
              <p>Loading...</p>
            ) : directories.length === 0 ? (
              <p className="empty-state">No directories added yet. Add a folder to get started!</p>
            ) : (
              <ul className="directory-list">
                {directories.map(dir => (
                  <li key={dir.id} className={`directory-item ${dir.enabled ? '' : 'disabled'}`}>
                    <div className="directory-info">
                      <strong>{dir.name}</strong>
                      <span className="directory-path">{dir.path}</span>
                      <span className="directory-stats">{dir.image_count} images</span>
                      <div className="directory-diagnostics">
                        <span className="diagnostic" title="Images with age detection">
                          Age: {dir.age_detected_pct}%
                        </span>
                        <span className="diagnostic" title="Images with booru tags">
                          Tagged: {dir.tagged_pct}%
                        </span>
                        <span className="diagnostic" title="Favorited images">
                          Favorites: {dir.favorited_count}
                        </span>
                        <button
                          className="diagnostic toggle-btn"
                          onClick={() => {
                            const newValue = !dir.auto_age_detect
                            setDirectories(dirs => dirs.map(d =>
                              d.id === dir.id ? {...d, auto_age_detect: newValue} : d
                            ))
                            updateDirectory(dir.id, { auto_age_detect: newValue })
                              .catch(err => {
                                console.error('Failed to update:', err)
                                refreshDirectories()
                              })
                          }}
                        >
                          {dir.auto_age_detect ? '☑' : '☐'} Age Detect
                        </button>
                        <button
                          className="diagnostic toggle-btn public-toggle"
                          onClick={() => {
                            const newValue = !dir.public_access
                            setDirectories(dirs => dirs.map(d =>
                              d.id === dir.id ? {...d, public_access: newValue} : d
                            ))
                            updateDirectory(dir.id, { public_access: newValue })
                              .catch(err => {
                                console.error('Failed to update:', err)
                                refreshDirectories()
                              })
                          }}
                          title="Allow public network access to this directory"
                        >
                          {dir.public_access ? '☑' : '☐'} Public
                        </button>
                      </div>
                    </div>
                    <div className="directory-actions">
                      <button
                        className="rescan-btn"
                        onClick={() => handleRescan(dir.id)}
                        disabled={scanning[dir.id]}
                      >
                        {scanning[dir.id] ? 'Scanning...' : 'Rescan'}
                      </button>
                      <button
                        className="prune-btn"
                        onClick={() => handlePrune(dir.id, dir.name || dir.path, dir.favorited_count)}
                        disabled={pruning[dir.id] || dir.image_count === 0}
                        title="Move non-favorited images to dumpster"
                      >
                        {pruning[dir.id] ? 'Pruning...' : 'Prune'}
                      </button>
                      <button
                        className="comfyui-btn"
                        onClick={() => setComfyuiConfigDir(dir)}
                        title="Configure ComfyUI metadata extraction"
                      >
                        ComfyUI
                      </button>
                      <button
                        className="remove-btn"
                        onClick={() => handleRemove(dir.id, dir.name || dir.path)}
                      >
                        Remove
                      </button>
                    </div>
                    <div className="directory-status">
                      {!dir.path_exists && <span className="warning">Path not found</span>}
                      {dir.enabled ? '✓ Active' : 'Disabled'}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </main>
      </div>

      {/* ComfyUI Configuration Modal */}
      {comfyuiConfigDir && (
        <ComfyUIConfigModal
          directoryId={comfyuiConfigDir.id}
          directoryName={comfyuiConfigDir.name || comfyuiConfigDir.path}
          onClose={() => setComfyuiConfigDir(null)}
          onSave={refreshDirectories}
        />
      )}
    </div>
  )
}

// Settings page with tabs
function SettingsPage() {
  const [activeTab, setActiveTab] = useState('general')
  const [queueStatus, setQueueStatus] = useState(null)
  const [stats, setStats] = useState(null)
  const [dumpsterPath, setDumpsterPath] = useState('')
  const [ageDetection, setAgeDetection] = useState({
    enabled: false,
    installed: false,
    installing: false,
    progress: '',
    dependencies: {}
  })

  const refreshAgeDetectionStatus = async () => {
    try {
      const { getAgeDetectionStatus } = await import('./api')
      const status = await getAgeDetectionStatus()
      setAgeDetection(status)
    } catch (e) {
      console.error('Failed to get age detection status:', e)
    }
  }

  useEffect(() => {
    import('./api').then(({ getQueueStatus }) => {
      getQueueStatus().then(setQueueStatus).catch(console.error)
    })
    getLibraryStats().then(setStats).catch(console.error)
    refreshAgeDetectionStatus()
    // Load saved dumpster path
    const saved = localStorage.getItem('localbooru_dumpster_path')
    if (saved) setDumpsterPath(saved)
  }, [])

  // Live polling for queue status when there's pending work
  useEffect(() => {
    const hasPendingWork = queueStatus?.by_status?.pending > 0 || queueStatus?.by_status?.processing > 0
    if (hasPendingWork) {
      const interval = setInterval(() => {
        import('./api').then(({ getQueueStatus }) => {
          getQueueStatus().then(setQueueStatus).catch(console.error)
        })
      }, 2000)  // Poll every 2 seconds
      return () => clearInterval(interval)
    }
  }, [queueStatus?.by_status?.pending, queueStatus?.by_status?.processing])

  // Poll for installation progress
  useEffect(() => {
    if (ageDetection.installing) {
      const interval = setInterval(refreshAgeDetectionStatus, 2000)
      return () => clearInterval(interval)
    }
  }, [ageDetection.installing])

  const handleDumpsterPathChange = (e) => {
    const path = e.target.value
    setDumpsterPath(path)
    if (path) {
      localStorage.setItem('localbooru_dumpster_path', path)
    } else {
      localStorage.removeItem('localbooru_dumpster_path')
    }
  }

  return (
    <div className="app">
      <div className="main-container">
        <Sidebar stats={stats} />
        <main className="content with-sidebar">
          <div className="page settings-page">
            <h1>Settings</h1>

            {/* Settings Tabs */}
            <div className="settings-tabs">
              <button
                className={`settings-tab ${activeTab === 'general' ? 'active' : ''}`}
                onClick={() => setActiveTab('general')}
              >
                General
              </button>
              <button
                className={`settings-tab ${activeTab === 'network' ? 'active' : ''}`}
                onClick={() => setActiveTab('network')}
              >
                Network
              </button>
            </div>

            {/* Network Tab Content */}
            {activeTab === 'network' && <NetworkSettings />}

            {/* General Tab Content */}
            {activeTab === 'general' && (
            <>
            <section>
              <h2>Age Detection (Optional)</h2>
              <p className="setting-description">
                Detect faces and estimate ages in images. Requires ~2GB of additional dependencies (PyTorch, etc).
              </p>

              <div className="age-detection-status">
                <div className="deps-status">
                  <strong>Dependencies:</strong>
                  {Object.entries(ageDetection.dependencies || {}).filter(([dep]) => !dep.endsWith('_error')).map(([dep, installed]) => (
                    <span key={dep} className={`dep-badge ${installed ? 'installed' : 'missing'}`}>
                      {dep}: {installed ? '✓' : '✗'}
                    </span>
                  ))}
                </div>
                {ageDetection.dependencies?.torch_error && (
                  <p className="error-message" style={{color: '#ff6b6b', marginTop: '8px', fontSize: '0.9em'}}>
                    {ageDetection.dependencies.torch_error}
                  </p>
                )}

                {!ageDetection.installed && !ageDetection.installing && (
                  <button
                    onClick={async () => {
                      if (!confirm('Install age detection dependencies?\n\nThis will download ~2GB of data and may take several minutes.')) return
                      try {
                        const { installAgeDetection } = await import('./api')
                        const result = await installAgeDetection()
                        console.log('Install result:', result)
                        if (!result.success) {
                          alert(result.error || 'Failed to start installation')
                        }
                        refreshAgeDetectionStatus()
                      } catch (e) {
                        console.error('Install error:', e)
                        alert('Failed to start installation: ' + e.message)
                      }
                    }}
                    className="install-btn"
                  >
                    Install Dependencies (~2GB)
                  </button>
                )}

                {ageDetection.installing && (
                  <div className="install-progress">
                    <span className="spinner"></span>
                    <span>{ageDetection.progress || 'Installing...'}</span>
                  </div>
                )}

                {ageDetection.installed && (
                  <div className="toggle-setting">
                    <label>
                      <input
                        type="checkbox"
                        checked={ageDetection.enabled}
                        onChange={async (e) => {
                          const newValue = e.target.checked
                          const { toggleAgeDetection } = await import('./api')
                          const result = await toggleAgeDetection(newValue)
                          if (result.success) {
                            setAgeDetection(prev => ({ ...prev, enabled: newValue }))
                          } else {
                            alert(result.error || 'Failed to toggle')
                          }
                        }}
                      />
                      Enable age detection on new images
                    </label>
                    {ageDetection.enabled && (
                      <button
                        onClick={async () => {
                          try {
                            const { detectAgesRetrospective } = await import('./api')
                            const result = await detectAgesRetrospective()
                            alert(result.message || `Queued ${result.queued} images for age detection`)
                          } catch (e) {
                            alert('Failed: ' + e.message)
                          }
                        }}
                        style={{ marginLeft: '1rem' }}
                      >
                        Run on existing images
                      </button>
                    )}
                  </div>
                )}
              </div>
            </section>

            <section>
              <h2>Dumpster Location</h2>
              <p className="setting-description">Where pruned (non-favorited) images are moved to. Leave empty to use default (~/.localbooru/dumpster)</p>
              <input
                type="text"
                value={dumpsterPath}
                onChange={handleDumpsterPathChange}
                placeholder="/path/to/dumpster"
                className="setting-input"
              />
            </section>

            {/* Tagging Progress - only show if there's pending work */}
            {queueStatus && (queueStatus.by_status?.pending > 0 || queueStatus.by_status?.processing > 0) && (
              <section className="tagging-progress-section">
                <h2>Tagging Progress</h2>
                <div className="tagging-progress">
                  <div className="progress-bar-container">
                    <div
                      className="progress-bar-fill"
                      style={{
                        width: `${Math.round(
                          ((queueStatus.by_status?.completed || 0) /
                          ((queueStatus.by_status?.completed || 0) + (queueStatus.by_status?.pending || 0) + (queueStatus.by_status?.processing || 0))) * 100
                        ) || 0}%`
                      }}
                    />
                  </div>
                  <div className="progress-text">
                    {queueStatus.by_status?.processing > 0 && (
                      <span className="processing-indicator">Processing...</span>
                    )}
                    <span>{(queueStatus.by_status?.pending || 0).toLocaleString()} remaining</span>
                  </div>
                </div>
              </section>
            )}

            {window.electronAPI?.isElectron && (
              <section>
                <h2>Application</h2>
                <button onClick={() => {
                  if (!confirm('Quit LocalBooru completely?\n\nThis will stop the background server and close the application.')) return
                  window.electronAPI.quitApp()
                }} className="danger-btn">
                  Quit Application
                </button>
              </section>
            )}
            </>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

function Gallery() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [images, setImages] = useState([])
  const [tags, setTags] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [total, setTotal] = useState(0)
  const [lightboxIndex, setLightboxIndex] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [lightboxSidebarHover, setLightboxSidebarHover] = useState(false)
  const [stats, setStats] = useState(null)
  const statsUpdateTimeout = useRef(null)
  const lightboxIndexRef = useRef(null)

  // Keep ref in sync with state (for use in timeouts)
  useEffect(() => {
    lightboxIndexRef.current = lightboxIndex
  }, [lightboxIndex])

  // Selection mode state
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedImages, setSelectedImages] = useState(new Set())
  const [batchActionLoading, setBatchActionLoading] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [deleteWithFiles, setDeleteWithFiles] = useState(false)
  const [showMoveModal, setShowMoveModal] = useState(false)
  const [moveDirectories, setMoveDirectories] = useState([])
  const [selectedMoveDir, setSelectedMoveDir] = useState(null)

  const currentTags = searchParams.get('tags') || ''
  const currentRating = searchParams.get('rating') || 'pg,pg13,r,x,xxx'
  const favoritesOnly = searchParams.get('favorites') === 'true'
  const currentSort = searchParams.get('sort') || 'newest'
  const currentDirectoryId = searchParams.get('directory') ? parseInt(searchParams.get('directory')) : null
  const currentMinAge = searchParams.get('min_age') ? parseInt(searchParams.get('min_age')) : null
  const currentMaxAge = searchParams.get('max_age') ? parseInt(searchParams.get('max_age')) : null
  const currentTimeframe = searchParams.get('timeframe') || null

  // Load saved filters from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('localbooru_filters')
    if (saved && !window.location.search) {
      try {
        const filters = JSON.parse(saved)
        const params = {}
        if (filters.tags) params.tags = filters.tags
        if (filters.rating && filters.rating !== 'pg,pg13,r,x,xxx') params.rating = filters.rating
        if (filters.favorites) params.favorites = 'true'
        if (filters.sort && filters.sort !== 'newest') params.sort = filters.sort
        if (filters.directory) params.directory = filters.directory
        if (filters.min_age !== null && filters.min_age !== undefined) params.min_age = filters.min_age
        if (filters.max_age !== null && filters.max_age !== undefined) params.max_age = filters.max_age
        if (Object.keys(params).length > 0) {
          setSearchParams(params)
        }
      } catch (e) {
        console.error('Failed to load saved filters:', e)
      }
    }
  }, [])

  // Save filters to localStorage when they change
  useEffect(() => {
    const filters = {
      tags: currentTags || null,
      rating: currentRating,
      favorites: favoritesOnly,
      sort: currentSort,
      directory: currentDirectoryId,
      min_age: currentMinAge,
      max_age: currentMaxAge
    }
    localStorage.setItem('localbooru_filters', JSON.stringify(filters))
  }, [currentTags, currentRating, favoritesOnly, currentSort, currentDirectoryId, currentMinAge, currentMaxAge])

  // Touch handling for mobile sidebar
  const touchStartX = useRef(null)

  const handleTouchStart = useCallback((e) => {
    touchStartX.current = e.touches[0].clientX
  }, [])

  const handleTouchEnd = useCallback((e) => {
    if (touchStartX.current === null || window.innerWidth > 1024) return
    const deltaX = e.changedTouches[0].clientX - touchStartX.current
    if (Math.abs(deltaX) > 50) {
      if (deltaX > 0 && !sidebarOpen) setSidebarOpen(true)
      if (deltaX < 0 && sidebarOpen) setSidebarOpen(false)
    }
    touchStartX.current = null
  }, [sidebarOpen])

  // Load images
  const loadImages = useCallback(async (pageNum = 1, append = false) => {
    setLoading(true)
    try {
      const result = await fetchImages({
        tags: currentTags,
        rating: currentRating,
        favorites_only: favoritesOnly,
        directory_id: currentDirectoryId,
        min_age: currentMinAge,
        max_age: currentMaxAge,
        timeframe: currentTimeframe,
        sort: currentSort,
        page: pageNum,
        per_page: 50
      })

      if (append) {
        // Deduplicate when appending to avoid showing same image twice
        setImages(prev => {
          const existingIds = new Set(prev.map(img => img.id))
          const newImages = result.images.filter(img => !existingIds.has(img.id))
          return [...prev, ...newImages]
        })
      } else {
        setImages(result.images)
      }
      setTotal(result.total)
      const loadedCount = append ? images.length + result.images.length : result.images.length
      setHasMore(loadedCount < result.total)
      setPage(pageNum)
    } catch (error) {
      console.error('Failed to load images:', error)
    }
    setLoading(false)
  }, [currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge, currentTimeframe])

  // Update a single image in the images array
  const handleImageUpdate = useCallback((imageId, updates) => {
    setImages(prev => prev.map(img =>
      img.id === imageId ? { ...img, ...updates } : img
    ))
  }, [])

  // Handle image deletion from lightbox
  const handleImageDelete = useCallback((imageId) => {
    setImages(prev => {
      const newImages = prev.filter(img => img.id !== imageId)

      // Find the current index of the deleted image
      const deletedIndex = prev.findIndex(img => img.id === imageId)

      // If there are remaining images, navigate to the next one
      if (newImages.length > 0) {
        // If we deleted the last image, go to the previous one
        const nextIndex = deletedIndex >= newImages.length ? newImages.length - 1 : deletedIndex
        setLightboxIndex(newImages[nextIndex]?.id ?? null)
      } else {
        // No images left, close lightbox
        setLightboxIndex(null)
      }

      return newImages
    })
  }, [])

  // Load tags
  const loadTags = useCallback(async () => {
    try {
      const result = await fetchTags({ per_page: 100 })
      setTags(result.tags || [])
    } catch (error) {
      console.error('Failed to load tags:', error)
    }
  }, [])

  useEffect(() => {
    loadImages(1, false)
  }, [currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge, currentTimeframe, loadImages])

  useEffect(() => {
    loadTags()
  }, [loadTags])

  useEffect(() => {
    getLibraryStats().then(setStats).catch(console.error)
  }, [])

  // Subscribe to real-time library events (debounced refresh)
  // Waits 2s after last event, then refreshes once
  // Only refreshes images when: sorted by newest, scrolled to top, and not in lightbox
  const triggerDebouncedRefresh = useCallback(() => {
    if (statsUpdateTimeout.current) {
      clearTimeout(statsUpdateTimeout.current)
    }
    statsUpdateTimeout.current = setTimeout(() => {
      // Always update stats
      getLibraryStats().then(setStats).catch(console.error)

      // Only refresh images if sorted by newest, scrolled near top, and not in lightbox
      // Use ref for lightbox check since timeout captures stale closure values
      const isAtTop = window.scrollY < 200
      const isNewest = currentSort === 'newest'
      const isInLightbox = lightboxIndexRef.current !== null

      if (isNewest && isAtTop && !isInLightbox) {
        loadImages(1, false)
      }
    }, 2000)
  }, [loadImages, currentSort])

  // On visibility change, start debounce - backlog events will keep resetting it
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        triggerDebouncedRefresh()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [triggerDebouncedRefresh])

  // SSE events also trigger the same debounce
  useEffect(() => {
    const unsubscribe = subscribeToLibraryEvents((event) => {
      if (event.type === 'image_added') {
        triggerDebouncedRefresh()
      }
    })
    return () => {
      unsubscribe()
      if (statsUpdateTimeout.current) {
        clearTimeout(statsUpdateTimeout.current)
      }
    }
  }, [triggerDebouncedRefresh])

  const handleLoadMore = () => {
    if (!loading && hasMore) {
      loadImages(page + 1, true)
    }
  }

  const handleTagClick = (tagName) => {
    const currentTagList = currentTags ? currentTags.split(',').map(t => t.trim()) : []
    let newTagList

    if (currentTagList.includes(tagName)) {
      newTagList = currentTagList.filter(t => t !== tagName)
    } else {
      newTagList = [...currentTagList, tagName]
    }

    const params = {}
    if (newTagList.length > 0) params.tags = newTagList.join(',')
    if (currentRating !== 'pg,pg13,r,x,xxx') params.rating = currentRating
    if (favoritesOnly) params.favorites = 'true'
    if (currentSort !== 'newest') params.sort = currentSort
    if (currentDirectoryId) params.directory = currentDirectoryId
    if (currentMinAge !== null) params.min_age = currentMinAge
    if (currentMaxAge !== null) params.max_age = currentMaxAge
    setSearchParams(params)
  }

  const handleSearch = (tags, rating, sort, favOnly, directoryId, minAge, maxAge, timeframe) => {
    const params = {}
    if (tags) params.tags = tags
    if (rating && rating !== 'pg,pg13,r,x,xxx') params.rating = rating
    if (favOnly) params.favorites = 'true'
    if (sort && sort !== 'newest') params.sort = sort
    if (directoryId) params.directory = directoryId
    if (minAge !== null && minAge !== undefined) params.min_age = minAge
    if (maxAge !== null && maxAge !== undefined) params.max_age = maxAge
    if (timeframe) params.timeframe = timeframe
    setSearchParams(params)
  }

  const handleImageClick = (imageId) => {
    setLightboxIndex(imageId)
    // Keep sidebar visible to show image details
  }

  const handleLightboxClose = () => {
    // Scroll to the image that was being viewed
    const imageId = lightboxIndex
    setLightboxIndex(null)

    // Use requestAnimationFrame to scroll after the lightbox closes and DOM updates
    requestAnimationFrame(() => {
      const imageElement = document.querySelector(`[data-image-id="${imageId}"]`)
      if (imageElement) {
        imageElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    })
  }

  const handleLightboxNav = async (direction) => {
    const currentIdx = images.findIndex(img => img.id === lightboxIndex)
    if (currentIdx === -1) return

    let newIndex = currentIdx + direction

    // Navigating past the end - load more if available
    if (newIndex >= images.length && hasMore && !loading) {
      // Load more images
      const nextPage = page + 1
      try {
        const result = await fetchImages({
          tags: currentTags,
          rating: currentRating,
          favorites_only: favoritesOnly,
          directory_id: currentDirectoryId,
          min_age: currentMinAge,
          max_age: currentMaxAge,
          timeframe: currentTimeframe,
          sort: currentSort,
          page: nextPage,
          per_page: 50
        })

        if (result.images.length > 0) {
          // Deduplicate to avoid showing same image twice
          const existingIds = new Set(images.map(img => img.id))
          const newImages = result.images.filter(img => !existingIds.has(img.id))

          if (newImages.length > 0) {
            setImages(prev => [...prev, ...newImages])
            const newLoadedCount = images.length + newImages.length
            setHasMore(newLoadedCount < result.total)
            setPage(nextPage)
            // Navigate to the first new image
            setLightboxIndex(newImages[0].id)
            return
          }
        }
      } catch (error) {
        console.error('Failed to load more images:', error)
      }
    }

    // Stay at boundaries - don't wrap
    if (newIndex < 0) return
    if (newIndex >= images.length) return
    setLightboxIndex(images[newIndex]?.id ?? lightboxIndex)
  }

  // Selection mode handlers
  const toggleSelectionMode = () => {
    setSelectionMode(prev => !prev)
    if (selectionMode) {
      // Exiting selection mode - clear selection
      setSelectedImages(new Set())
    }
  }

  const handleSelectImage = (imageId) => {
    setSelectedImages(prev => {
      const newSet = new Set(prev)
      if (newSet.has(imageId)) {
        newSet.delete(imageId)
      } else {
        newSet.add(imageId)
      }
      return newSet
    })
  }

  const clearSelection = () => {
    setSelectedImages(new Set())
  }

  const selectAll = () => {
    setSelectedImages(new Set(images.map(img => img.id)))
  }

  // Batch action handlers
  const handleBatchDelete = async () => {
    if (selectedImages.size === 0) return
    setBatchActionLoading(true)
    try {
      const result = await batchDeleteImages(Array.from(selectedImages), deleteWithFiles)
      console.log('Batch delete result:', result)
      // Refresh the gallery
      await loadImages(1, false)
      setSelectedImages(new Set())
      setShowDeleteConfirm(false)
      setDeleteWithFiles(false)
    } catch (error) {
      console.error('Batch delete failed:', error)
    }
    setBatchActionLoading(false)
  }

  const handleBatchRetag = async () => {
    if (selectedImages.size === 0) return
    setBatchActionLoading(true)
    try {
      const result = await batchRetag(Array.from(selectedImages))
      console.log('Batch retag result:', result)
      alert(`Queued ${result.queued} images for retagging`)
      setSelectedImages(new Set())
    } catch (error) {
      console.error('Batch retag failed:', error)
    }
    setBatchActionLoading(false)
  }

  const handleBatchAgeDetect = async () => {
    if (selectedImages.size === 0) return
    setBatchActionLoading(true)
    try {
      const result = await batchAgeDetect(Array.from(selectedImages))
      console.log('Batch age detect result:', result)
      alert(`Queued ${result.queued} images for age detection`)
      setSelectedImages(new Set())
    } catch (error) {
      console.error('Batch age detect failed:', error)
    }
    setBatchActionLoading(false)
  }

  const openMoveModal = async () => {
    try {
      const dirs = await fetchDirectories()
      setMoveDirectories(dirs)
      setSelectedMoveDir(null)
      setShowMoveModal(true)
    } catch (error) {
      console.error('Failed to fetch directories:', error)
    }
  }

  const handleBatchMove = async () => {
    if (selectedImages.size === 0 || !selectedMoveDir) return
    setBatchActionLoading(true)
    try {
      const result = await batchMoveImages(Array.from(selectedImages), selectedMoveDir)
      console.log('Batch move result:', result)
      alert(`Moved ${result.moved} images`)
      // Refresh the gallery
      await loadImages(1, false)
      setSelectedImages(new Set())
      setShowMoveModal(false)
      setSelectedMoveDir(null)
    } catch (error) {
      console.error('Batch move failed:', error)
    }
    setBatchActionLoading(false)
  }

  return (
    <div
      className={`app gallery-view ${lightboxIndex !== null ? 'lightbox-active' : ''}`}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      {/* Lightbox backdrop - must be outside main-container for correct z-index stacking */}
      {lightboxSidebarHover && (
        <div
          className="sidebar-backdrop lightbox-backdrop"
          onClick={() => setLightboxSidebarHover(false)}
          onTouchStart={(e) => {
            e.currentTarget.dataset.touchStartX = e.touches[0].clientX
          }}
          onTouchEnd={(e) => {
            const startX = parseFloat(e.currentTarget.dataset.touchStartX)
            const endX = e.changedTouches[0].clientX
            // Swipe left closes sidebar
            if (startX - endX > 50) {
              setLightboxSidebarHover(false)
            }
          }}
        />
      )}

      <div className="main-container">
        {sidebarOpen && (
          <div
            className="sidebar-backdrop"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        <Sidebar
          tags={tags}
          onTagClick={(tag) => {
            handleTagClick(tag)
            setSidebarOpen(false)
          }}
          mobileOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          currentTags={currentTags}
          selectedImage={lightboxIndex !== null ? images.find(img => img.id === lightboxIndex) : null}
          onSearch={handleSearch}
          initialTags={currentTags}
          initialRating={currentRating}
          initialFavoritesOnly={favoritesOnly}
          initialDirectoryId={currentDirectoryId}
          initialMinAge={currentMinAge}
          initialMaxAge={currentMaxAge}
          initialSort={currentSort}
          initialTimeframe={currentTimeframe}
          total={total}
          stats={stats}
          lightboxMode={lightboxIndex !== null}
          lightboxHover={lightboxSidebarHover}
          onMouseLeave={() => setLightboxSidebarHover(false)}
        />

        {!sidebarOpen && <div className="swipe-hint" />}

        <main className="content with-sidebar">
          {!loading && images.length === 0 ? (
            <div className="no-results">
              <h2>No images found</h2>
              <p>Try adjusting your search filters or add some directories to watch.</p>
            </div>
          ) : (
            <MasonryGrid
              images={images}
              onImageClick={handleImageClick}
              onLoadMore={handleLoadMore}
              loading={loading}
              hasMore={hasMore}
              onImageUpdate={loadImages}
              isSelectable={selectionMode}
              selectedImages={selectedImages}
              onSelectImage={handleSelectImage}
            />
          )}

          {/* Floating select button */}
          <button
            className={`floating-select-btn ${selectionMode ? 'active' : ''}`}
            onClick={toggleSelectionMode}
            title={selectionMode ? 'Exit selection mode' : 'Enter selection mode'}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
              {selectionMode && <path d="M9 12l2 2 4-4"/>}
            </svg>
            {selectionMode ? 'Done' : 'Select'}
          </button>
        </main>

        {/* Batch action bar - shown when images are selected */}
        {selectionMode && selectedImages.size > 0 && (
          <div className="batch-action-bar">
            <div className="batch-action-count">
              {selectedImages.size} selected
            </div>
            <div className="batch-action-buttons">
              <button
                className="batch-btn"
                onClick={handleBatchRetag}
                disabled={batchActionLoading}
                title="Re-tag selected images"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>
                  <line x1="7" y1="7" x2="7.01" y2="7"/>
                </svg>
                Retag
              </button>
              <button
                className="batch-btn"
                onClick={handleBatchAgeDetect}
                disabled={batchActionLoading}
                title="Re-detect ages in selected images"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="8" r="5"/>
                  <path d="M20 21a8 8 0 1 0-16 0"/>
                </svg>
                Age Detect
              </button>
              <button
                className="batch-btn"
                onClick={openMoveModal}
                disabled={batchActionLoading}
                title="Move selected images to another directory"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                  <path d="M12 11v6"/>
                  <path d="M9 14l3-3 3 3"/>
                </svg>
                Move
              </button>
              <button
                className="batch-btn danger"
                onClick={() => setShowDeleteConfirm(true)}
                disabled={batchActionLoading}
                title="Delete selected images"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                </svg>
                Delete
              </button>
              <button
                className="batch-btn secondary"
                onClick={clearSelection}
                disabled={batchActionLoading}
              >
                Clear
              </button>
            </div>
          </div>
        )}

        {/* Delete confirmation modal */}
        {showDeleteConfirm && (
          <div className="modal-overlay" onClick={() => setShowDeleteConfirm(false)}>
            <div className="modal-content delete-confirm-modal" onClick={e => e.stopPropagation()}>
              <h3>Delete {selectedImages.size} images?</h3>
              <p>This action cannot be undone.</p>
              <label className="delete-files-option">
                <input
                  type="checkbox"
                  checked={deleteWithFiles}
                  onChange={(e) => setDeleteWithFiles(e.target.checked)}
                />
                Also delete original files from disk
              </label>
              <div className="modal-actions">
                <button
                  className="cancel-btn"
                  onClick={() => {
                    setShowDeleteConfirm(false)
                    setDeleteWithFiles(false)
                  }}
                >
                  Cancel
                </button>
                <button
                  className="danger-btn"
                  onClick={handleBatchDelete}
                  disabled={batchActionLoading}
                >
                  {batchActionLoading ? 'Deleting...' : 'Delete'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Move to directory modal */}
        {showMoveModal && (
          <div className="modal-overlay" onClick={() => setShowMoveModal(false)}>
            <div className="modal-content move-modal" onClick={e => e.stopPropagation()}>
              <h3>Move {selectedImages.size} images</h3>
              <p>Select destination directory:</p>
              <div className="directory-list">
                {moveDirectories.map(dir => (
                  <label key={dir.id} className="directory-option">
                    <input
                      type="radio"
                      name="moveDir"
                      value={dir.id}
                      checked={selectedMoveDir === dir.id}
                      onChange={() => setSelectedMoveDir(dir.id)}
                    />
                    <span className="dir-name">{dir.name || dir.path.split('/').pop()}</span>
                    <span className="dir-path">{dir.path}</span>
                  </label>
                ))}
              </div>
              <div className="modal-actions">
                <button
                  className="cancel-btn"
                  onClick={() => {
                    setShowMoveModal(false)
                    setSelectedMoveDir(null)
                  }}
                >
                  Cancel
                </button>
                <button
                  className="primary-btn"
                  onClick={handleBatchMove}
                  disabled={batchActionLoading || !selectedMoveDir}
                >
                  {batchActionLoading ? 'Moving...' : 'Move'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {lightboxIndex !== null && (
        <Lightbox
          images={images}
          currentIndex={images.findIndex(img => img.id === lightboxIndex)}
          total={total}
          onClose={handleLightboxClose}
          onNav={handleLightboxNav}
          onTagClick={handleTagClick}
          onImageUpdate={handleImageUpdate}
          onSidebarHover={setLightboxSidebarHover}
          sidebarOpen={lightboxSidebarHover}
          onDelete={handleImageDelete}
        />
      )}
    </div>
  )
}

function App() {
  return (
    <>
      <TitleBar />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Gallery />} />
          <Route path="/directories" element={<DirectoriesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App
