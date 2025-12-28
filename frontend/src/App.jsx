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
import { fetchImages, fetchTags, getLibraryStats } from './api'
import './App.css'


// Directory management page
function DirectoriesPage() {
  const [directories, setDirectories] = useState([])
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState({})
  const [pruning, setPruning] = useState({})
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
    if (!confirm(`Remove "${dirName}" from watch list?\n\nThis will NOT delete the actual files.`)) {
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

            <button onClick={handleAddDirectory} className="add-directory-btn">
              + Add Directory
            </button>

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
    </div>
  )
}

// Settings page
function SettingsPage() {
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

            <section>
              <h2>Age Detection (Optional)</h2>
              <p className="setting-description">
                Detect faces and estimate ages in images. Requires ~2GB of additional dependencies (PyTorch, etc).
              </p>

              <div className="age-detection-status">
                <div className="deps-status">
                  <strong>Dependencies:</strong>
                  {Object.entries(ageDetection.dependencies || {}).map(([dep, installed]) => (
                    <span key={dep} className={`dep-badge ${installed ? 'installed' : 'missing'}`}>
                      {dep}: {installed ? '✓' : '✗'}
                    </span>
                  ))}
                </div>

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

            <section>
              <h2>Background Tasks</h2>
              {queueStatus && (
                <div className="queue-status">
                  <p>Pending: {queueStatus.by_status?.pending || 0}</p>
                  <p>Processing: {queueStatus.by_status?.processing || 0}</p>
                  <p>Completed: {queueStatus.by_status?.completed || 0}</p>
                  <p>Failed: {queueStatus.by_status?.failed || 0}</p>
                </div>
              )}
            </section>

            <section>
              <h2>Actions</h2>
              <button onClick={async () => {
                const { tagUntagged } = await import('./api')
                const result = await tagUntagged()
                alert(`Queued ${result.queued} images for tagging`)
              }}>
                Tag Untagged Images
              </button>

              <button onClick={async () => {
                const { verifyFiles } = await import('./api')
                await verifyFiles()
                alert('File verification queued')
              }}>
                Verify File Locations
              </button>

              <button onClick={async () => {
                const { retryFailedTasks } = await import('./api')
                const result = await retryFailedTasks()
                alert(`Retried ${result.retried} failed tasks`)
              }}>
                Retry Failed Tasks
              </button>

              <button onClick={async () => {
                if (!confirm('Clear all pending tasks from the queue?')) return
                const { clearPendingTasks } = await import('./api')
                const result = await clearPendingTasks()
                alert(`Cleared ${result.cleared} pending tasks`)
                // Refresh queue status
                const { getQueueStatus } = await import('./api')
                getQueueStatus().then(setQueueStatus).catch(console.error)
              }} className="danger-btn">
                Clear Pending Queue
              </button>
            </section>
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

  const currentTags = searchParams.get('tags') || ''
  const currentRating = searchParams.get('rating') || 'pg,pg13,r,x,xxx'
  const favoritesOnly = searchParams.get('favorites') === 'true'
  const currentSort = searchParams.get('sort') || 'newest'
  const currentDirectoryId = searchParams.get('directory') ? parseInt(searchParams.get('directory')) : null
  const currentMinAge = searchParams.get('min_age') ? parseInt(searchParams.get('min_age')) : null
  const currentMaxAge = searchParams.get('max_age') ? parseInt(searchParams.get('max_age')) : null

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
        sort: currentSort,
        page: pageNum,
        per_page: 50
      })

      if (append) {
        setImages(prev => [...prev, ...result.images])
      } else {
        setImages(result.images)
      }
      setTotal(result.total)
      setHasMore(result.images.length === 50)
      setPage(pageNum)
    } catch (error) {
      console.error('Failed to load images:', error)
    }
    setLoading(false)
  }, [currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge])

  // Update a single image in the images array
  const handleImageUpdate = useCallback((imageId, updates) => {
    setImages(prev => prev.map(img =>
      img.id === imageId ? { ...img, ...updates } : img
    ))
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
  }, [currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge, loadImages])

  useEffect(() => {
    loadTags()
  }, [loadTags])

  useEffect(() => {
    getLibraryStats().then(setStats).catch(console.error)
  }, [])

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

  const handleSearch = (tags, rating, sort, favOnly, directoryId, minAge, maxAge) => {
    const params = {}
    if (tags) params.tags = tags
    if (rating && rating !== 'pg,pg13,r,x,xxx') params.rating = rating
    if (favOnly) params.favorites = 'true'
    if (sort && sort !== 'newest') params.sort = sort
    if (directoryId) params.directory = directoryId
    if (minAge !== null && minAge !== undefined) params.min_age = minAge
    if (maxAge !== null && maxAge !== undefined) params.max_age = maxAge
    setSearchParams(params)
  }

  const handleImageClick = (imageId) => {
    setLightboxIndex(imageId)
    // Keep sidebar visible to show image details
  }

  const handleLightboxClose = () => {
    setLightboxIndex(null)
  }

  const handleLightboxNav = (direction) => {
    setLightboxIndex(currentId => {
      const currentIdx = images.findIndex(img => img.id === currentId)
      if (currentIdx === -1) return currentId
      let newIndex = currentIdx + direction
      if (newIndex < 0) newIndex = images.length - 1
      if (newIndex >= images.length) newIndex = 0
      return images[newIndex]?.id ?? currentId
    })
  }

  return (
    <div
      className={`app gallery-view ${lightboxIndex !== null ? 'lightbox-active' : ''}`}
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
    >
      <div className="main-container">
        {sidebarOpen && (
          <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
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
            />
          )}
        </main>
      </div>

      {lightboxIndex !== null && (
        <Lightbox
          images={images}
          currentIndex={images.findIndex(img => img.id === lightboxIndex)}
          onClose={handleLightboxClose}
          onNav={handleLightboxNav}
          onTagClick={handleTagClick}
          onImageUpdate={handleImageUpdate}
          onSidebarHover={setLightboxSidebarHover}
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
