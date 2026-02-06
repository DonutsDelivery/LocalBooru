/**
 * LocalBooru - Local image library with auto-tagging
 * Simplified single-user version
 */
import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { BrowserRouter, Routes, Route, useSearchParams, useNavigate, useLocation } from 'react-router-dom'
import { App as CapacitorApp } from '@capacitor/app'
import { isMobileApp } from './serverManager'
import MasonryGrid from './components/MasonryGrid'
import Sidebar from './components/Sidebar'
import Lightbox from './components/Lightbox'
import TitleBar from './components/TitleBar'
import ComfyUIConfigModal from './components/ComfyUIConfigModal'
import NetworkSettings from './components/NetworkSettings'
import ServerSettings from './components/ServerSettings'
import ServerSelectScreen from './components/ServerSelectScreen'
import MigrationSettings from './components/MigrationSettings'
import OpticalFlowSettings from './components/OpticalFlowSettings'
import QRConnect from './components/QRConnect'
import { fetchImages, fetchFolders, fetchTags, getLibraryStats, subscribeToLibraryEvents, batchDeleteImages, batchRetag, batchAgeDetect, batchMoveImages, fetchDirectories } from './api'
import DirectoriesPage from './pages/DirectoriesPage'
import './App.css'

// Column count calculation (mirrors MasonryGrid logic, extended for high-res screens)
const baseColumnCounts = {
  3840: 10, 3200: 9, 2400: 8, 1800: 7, 1400: 6, 1200: 5, 900: 4, 600: 3, 0: 2
}
const tileSizeAdjustments = { 1: 3, 2: 1, 3: 0, 4: -2, 5: -4 }
const tileWidths = { 1: 200, 2: 250, 3: 300, 4: 450, 5: 600 }

function getColumnCount(width, tileSize) {
  const adjustment = tileSizeAdjustments[tileSize] || 0
  const breakpoints = Object.keys(baseColumnCounts).map(Number).sort((a, b) => b - a)
  for (const bp of breakpoints) {
    if (width >= bp) {
      return Math.max(1, baseColumnCounts[bp] + adjustment)
    }
  }
  return Math.max(1, 2 + adjustment)
}

// Calculate how many items to load based on viewport and tile size
function calculatePerPage(tileSize) {
  const width = window.innerWidth
  const height = window.innerHeight
  const columns = getColumnCount(width, tileSize)
  const tileWidth = tileWidths[tileSize] || 300
  // Assume average aspect ratio of 1.33 (4:3), so tile height ≈ tileWidth * 0.75
  // Add some for captions/padding
  const avgTileHeight = tileWidth * 0.75 + 40
  const rows = Math.ceil(height / avgTileHeight)
  // Load enough for 2x viewport to ensure smooth scrolling
  const needed = columns * rows * 2
  // Minimum 50, maximum 400 (enough for 4K with small tiles)
  return Math.min(400, Math.max(50, needed))
}

// Settings page with tabs
function SettingsPage() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('general')
  const [queueStatus, setQueueStatus] = useState(null)
  const [queuePaused, setQueuePaused] = useState(false)
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
    import('./api').then(({ getQueueStatus, getQueuePaused }) => {
      getQueueStatus().then(setQueueStatus).catch(console.error)
      getQueuePaused().then(data => setQueuePaused(data.paused)).catch(console.error)
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
        import('./api').then(({ getQueueStatus, getQueuePaused }) => {
          getQueueStatus().then(setQueueStatus).catch(console.error)
          getQueuePaused().then(data => setQueuePaused(data.paused)).catch(console.error)
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
            <div className="page-header">
              <button className="back-btn mobile-only" onClick={() => navigate('/')} aria-label="Back to gallery">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
              </button>
              <h1>Settings</h1>
            </div>

            {/* Settings Tabs */}
            <div className="settings-tabs">
              <button
                className={`settings-tab ${activeTab === 'general' ? 'active' : ''}`}
                onClick={() => setActiveTab('general')}
              >
                General
              </button>
              <button
                className={`settings-tab ${activeTab === 'video' ? 'active' : ''}`}
                onClick={() => setActiveTab('video')}
              >
                Video
              </button>
              <button
                className={`settings-tab ${activeTab === 'network' ? 'active' : ''}`}
                onClick={() => setActiveTab('network')}
              >
                Network
              </button>
              <button
                className={`settings-tab ${activeTab === 'servers' ? 'active' : ''}`}
                onClick={() => setActiveTab('servers')}
              >
                Servers
              </button>
              <button
                className={`settings-tab ${activeTab === 'mobile' ? 'active' : ''}`}
                onClick={() => setActiveTab('mobile')}
              >
                Mobile
              </button>
              <button
                className={`settings-tab ${activeTab === 'data' ? 'active' : ''}`}
                onClick={() => setActiveTab('data')}
              >
                Data
              </button>
            </div>

            {/* Tab Contents - all rendered, visibility controlled by CSS for instant switching */}
            <div className={`settings-tab-content ${activeTab === 'video' ? 'active' : ''}`}>
              <OpticalFlowSettings />
            </div>

            <div className={`settings-tab-content ${activeTab === 'network' ? 'active' : ''}`}>
              <NetworkSettings />
            </div>

            <div className={`settings-tab-content ${activeTab === 'data' ? 'active' : ''}`}>
              <MigrationSettings />
            </div>

            <div className={`settings-tab-content ${activeTab === 'servers' ? 'active' : ''}`}>
              <ServerSettings />
            </div>

            <div className={`settings-tab-content ${activeTab === 'mobile' ? 'active' : ''}`}>
              <QRConnect />
            </div>

            <div className={`settings-tab-content ${activeTab === 'general' ? 'active' : ''}`}>
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
                    {queuePaused ? (
                      <span className="paused-indicator">Paused</span>
                    ) : queueStatus.by_status?.processing > 0 ? (
                      <span className="processing-indicator">Processing...</span>
                    ) : null}
                    <span>{(queueStatus.by_status?.pending || 0).toLocaleString()} remaining</span>
                    <button
                      className={`pause-btn ${queuePaused ? 'paused' : ''}`}
                      onClick={async () => {
                        const { pauseQueue, resumeQueue } = await import('./api')
                        if (queuePaused) {
                          await resumeQueue()
                          setQueuePaused(false)
                        } else {
                          await pauseQueue()
                          setQueuePaused(true)
                        }
                      }}
                      title={queuePaused ? 'Resume processing' : 'Pause processing'}
                    >
                      {queuePaused ? '▶ Resume' : '⏸ Pause'}
                    </button>
                  </div>
                </div>
              </section>
            )}

            {(window.electronAPI?.isElectron || window.__TAURI_INTERNALS__) && (
              <section>
                <h2>Application</h2>
                <button onClick={async () => {
                  if (!confirm('Quit LocalBooru completely?\n\nThis will stop the background server and close the application.')) return
                  if (window.electronAPI?.quitApp) {
                    window.electronAPI.quitApp()
                  } else if (window.__TAURI_INTERNALS__) {
                    const { getDesktopAPI } = await import('./tauriAPI')
                    const api = getDesktopAPI()
                    if (api?.quitApp) api.quitApp()
                  }
                }} className="danger-btn">
                  Quit Application
                </button>
              </section>
            )}
            </div>
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
  const [filtersInitialized, setFiltersInitialized] = useState(false)
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

  // Handle browser back button for lightbox (works on mobile and desktop)
  useEffect(() => {
    const handlePopState = (e) => {
      // If lightbox is open and we're going back, close it
      if (lightboxIndexRef.current !== null && !e.state?.lightbox) {
        setLightboxIndex(null)
      }
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  // Keep hasMore in sync with actual images count (fixes stale closure bugs)
  useEffect(() => {
    if (total > 0) {
      setHasMore(images.length < total)
    }
  }, [images.length, total])

  // Selection mode state
  const [selectionMode, setSelectionMode] = useState(false)
  const [selectedImages, setSelectedImages] = useState(new Set())
  const [batchActionLoading, setBatchActionLoading] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [deleteWithFiles, setDeleteWithFiles] = useState(false)
  const [showMoveModal, setShowMoveModal] = useState(false)
  const [moveDirectories, setMoveDirectories] = useState([])
  const [selectedMoveDir, setSelectedMoveDir] = useState(null)

  // Tile size state (1 = smallest/most columns, 5 = largest/fewest columns)
  const [tileSize, setTileSize] = useState(() => {
    const saved = localStorage.getItem('localbooru_tileSize')
    return saved ? parseInt(saved, 10) : 3
  })

  // Navigation jump state
  const [jumpInput, setJumpInput] = useState('')
  const [isJumping, setIsJumping] = useState(false)

  // Save tile size to localStorage
  useEffect(() => {
    localStorage.setItem('localbooru_tileSize', tileSize.toString())
  }, [tileSize])

  const currentTags = searchParams.get('tags') || ''
  const currentRating = searchParams.get('rating') || 'pg,pg13,r,x,xxx'
  const favoritesOnly = searchParams.get('favorites') === 'true'
  const currentSort = searchParams.get('sort') || 'newest'
  const currentDirectoryId = searchParams.get('directory') ? parseInt(searchParams.get('directory')) : null
  const currentMinAge = searchParams.get('min_age') ? parseInt(searchParams.get('min_age')) : null
  const currentMaxAge = searchParams.get('max_age') ? parseInt(searchParams.get('max_age')) : null
  const currentTimeframe = searchParams.get('timeframe') || null
  const currentFilename = searchParams.get('filename') || ''
  const currentOrientation = searchParams.get('orientation') || null
  // Resolution is stored as "widthxheight" in URL, e.g., "1920x1080"
  const resolutionParam = searchParams.get('resolution')
  const currentResolution = useMemo(() => {
    if (!resolutionParam) return null
    const [width, height] = resolutionParam.split('x').map(Number)
    if (width && height) return { width, height }
    return null
  }, [resolutionParam])
  // Duration is stored as "min-max" in URL, e.g., "60-300" for 1-5 minutes
  const durationParam = searchParams.get('duration')
  const currentDuration = useMemo(() => {
    if (!durationParam) return null
    const [min, max] = durationParam.split('-').map(v => v === 'null' ? null : Number(v))
    return { min: min ?? null, max: max ?? null }
  }, [durationParam])

  // Folder grouping URL params
  const groupByFolders = searchParams.get('group') === 'folders'
  const currentFolder = searchParams.get('folder') || null

  // Track if we're waiting for localStorage params to be applied to URL
  const [pendingParamsFromStorage, setPendingParamsFromStorage] = useState(false)

  // Load saved filters from localStorage on mount (intentionally runs once)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const saved = localStorage.getItem('localbooru_filters')
    const hasUrlParams = searchParams.toString().length > 0
    if (saved && !hasUrlParams) {
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
        if (filters.resolution) params.resolution = `${filters.resolution.width}x${filters.resolution.height}`
        if (filters.orientation) params.orientation = filters.orientation
        if (filters.duration) params.duration = `${filters.duration.min ?? 'null'}-${filters.duration.max ?? 'null'}`
        if (filters.groupByFolders) params.group = 'folders'
        if (Object.keys(params).length > 0) {
          setPendingParamsFromStorage(true)
          setSearchParams(params)
          return // Don't initialize yet - wait for params to be applied
        }
      } catch (e) {
        console.error('Failed to load saved filters:', e)
      }
    }
    setFiltersInitialized(true)
  }, [])

  // Initialize filters once localStorage params have been applied to URL
  useEffect(() => {
    if (pendingParamsFromStorage && searchParams.toString()) {
      setPendingParamsFromStorage(false)
      setFiltersInitialized(true)
    }
  }, [pendingParamsFromStorage, searchParams])

  // Save filters to localStorage when they change (only after initial load to avoid overwriting)
  useEffect(() => {
    if (!filtersInitialized) return
    const filters = {
      tags: currentTags || null,
      rating: currentRating,
      favorites: favoritesOnly,
      sort: currentSort,
      directory: currentDirectoryId,
      min_age: currentMinAge,
      max_age: currentMaxAge,
      resolution: currentResolution,
      orientation: currentOrientation,
      duration: currentDuration,
      groupByFolders: groupByFolders
    }
    localStorage.setItem('localbooru_filters', JSON.stringify(filters))
  }, [filtersInitialized, currentTags, currentRating, favoritesOnly, currentSort, currentDirectoryId, currentMinAge, currentMaxAge, currentResolution, currentOrientation, currentDuration, groupByFolders])

  // Touch handling for mobile sidebar
  const touchStartX = useRef(null)

  const handleTouchStart = useCallback((e) => {
    touchStartX.current = e.touches[0].clientX
  }, [])

  const handleTouchEnd = useCallback((e) => {
    if (touchStartX.current === null || window.innerWidth > 1024) return
    // Don't control gallery sidebar when lightbox is open - it has its own touch handling
    if (lightboxIndex !== null) {
      touchStartX.current = null
      return
    }
    const deltaX = e.changedTouches[0].clientX - touchStartX.current
    if (Math.abs(deltaX) > 50) {
      if (deltaX > 0 && !sidebarOpen) setSidebarOpen(true)
      if (deltaX < 0 && sidebarOpen) setSidebarOpen(false)
    }
    touchStartX.current = null
  }, [sidebarOpen, lightboxIndex])

  // Load folders for folder grouping view
  const loadFolders = useCallback(async () => {
    setLoading(true)
    try {
      const result = await fetchFolders({
        directory_id: currentDirectoryId,
        rating: currentRating,
        favorites_only: favoritesOnly,
        tags: currentTags,
      })
      const folderItems = result.folders.map(f => ({
        ...f,
        _isFolder: true,
        // Use thumbnail dimensions for masonry column balancing
        id: `folder-${f.path}`,
      }))
      setImages(folderItems)
      setTotal(result.total)
      setHasMore(false)
      setPage(1)
    } catch (error) {
      console.error('Failed to load folders:', error)
    }
    setLoading(false)
  }, [currentDirectoryId, currentRating, favoritesOnly, currentTags])

  // Load images
  const loadImages = useCallback(async (pageNum = 1, append = false) => {
    // If folder grouping is active and no specific folder selected, load folders instead
    if (groupByFolders && !currentFolder) {
      if (pageNum === 1 && !append) {
        return loadFolders()
      }
      return
    }

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
        filename: currentFilename,
        min_width: currentResolution?.width,
        min_height: currentResolution?.height,
        orientation: currentOrientation,
        min_duration: currentDuration?.min,
        max_duration: currentDuration?.max,
        import_source: currentFolder,
        sort: currentSort,
        page: pageNum,
        per_page: calculatePerPage(tileSize)
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
      // Note: hasMore is computed by useEffect based on actual images.length
      setPage(pageNum)
    } catch (error) {
      console.error('Failed to load images:', error)
    }
    setLoading(false)
  }, [currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge, currentTimeframe, currentFilename, currentResolution, currentOrientation, currentDuration, tileSize, groupByFolders, currentFolder, loadFolders])

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
    if (!filtersInitialized) return
    loadImages(1, false)
  }, [filtersInitialized, currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge, currentTimeframe, currentResolution, currentOrientation, currentDuration, groupByFolders, currentFolder, loadImages])

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

  // Jump to a specific image number in the results
  const jumpToImage = useCallback(async (targetIndex) => {
    if (targetIndex < 1 || targetIndex > total) return

    setIsJumping(true)
    const perPage = calculatePerPage(tileSize)
    const targetPage = Math.ceil(targetIndex / perPage)

    try {
      const result = await fetchImages({
        tags: currentTags,
        rating: currentRating,
        favorites_only: favoritesOnly,
        directory_id: currentDirectoryId,
        min_age: currentMinAge,
        max_age: currentMaxAge,
        timeframe: currentTimeframe,
        filename: currentFilename,
        min_width: currentResolution?.width,
        min_height: currentResolution?.height,
        orientation: currentOrientation,
        min_duration: currentDuration?.min,
        max_duration: currentDuration?.max,
        import_source: currentFolder,
        sort: currentSort,
        page: targetPage,
        per_page: perPage
      })

      setImages(result.images)
      setTotal(result.total)
      setPage(targetPage)

      // Scroll to top since we're showing a new set of images
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (error) {
      console.error('Failed to jump to image:', error)
    }
    setIsJumping(false)
  }, [currentTags, currentRating, favoritesOnly, currentDirectoryId, currentSort, currentMinAge, currentMaxAge, currentTimeframe, currentFilename, currentResolution, currentOrientation, currentDuration, total, tileSize])

  // Handle jump by offset (for +/- 100 buttons)
  const handleJumpByOffset = useCallback((offset) => {
    const perPage = calculatePerPage(tileSize)
    const currentFirstImage = (page - 1) * perPage + 1
    const targetIndex = Math.max(1, Math.min(total, currentFirstImage + offset))
    jumpToImage(targetIndex)
  }, [page, total, jumpToImage, tileSize])

  // Handle direct jump from input
  const handleJumpSubmit = useCallback((e) => {
    e.preventDefault()
    const targetIndex = parseInt(jumpInput, 10)
    if (!isNaN(targetIndex) && targetIndex >= 1 && targetIndex <= total) {
      jumpToImage(targetIndex)
      setJumpInput('')
    }
  }, [jumpInput, total, jumpToImage])

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
    if (groupByFolders) params.group = 'folders'
    if (currentFolder) params.folder = currentFolder
    setSearchParams(params)
  }

  const handleSearch = (tags, rating, sort, favOnly, directoryId, minAge, maxAge, timeframe, filename, resolution, orientation, duration) => {
    const params = {}
    if (tags) params.tags = tags
    if (rating && rating !== 'pg,pg13,r,x,xxx') params.rating = rating
    if (favOnly) params.favorites = 'true'
    if (sort && sort !== 'newest') params.sort = sort
    if (directoryId) params.directory = directoryId
    if (minAge !== null && minAge !== undefined) params.min_age = minAge
    if (maxAge !== null && maxAge !== undefined) params.max_age = maxAge
    if (timeframe) params.timeframe = timeframe
    if (filename) params.filename = filename
    if (resolution) params.resolution = `${resolution.width}x${resolution.height}`
    if (orientation) params.orientation = orientation
    if (duration) params.duration = `${duration.min ?? 'null'}-${duration.max ?? 'null'}`
    // Preserve folder grouping state across filter changes
    if (groupByFolders) params.group = 'folders'
    if (currentFolder) params.folder = currentFolder
    setSearchParams(params)
  }

  // Track which folder was entered so we can scroll back to it on exit
  const enteredFolderPathRef = useRef(null)

  const handleFolderClick = useCallback((folderPath) => {
    const urlValue = folderPath || '__unfiled__'
    enteredFolderPathRef.current = urlValue
    const params = Object.fromEntries(searchParams)
    params.folder = urlValue
    setSearchParams(params)
    requestAnimationFrame(() => {
      const el = document.querySelector('.content.with-sidebar > .masonry-container')
      if (el) el.scrollTop = 0
    })
  }, [searchParams, setSearchParams])

  const handleBackToFolders = useCallback(() => {
    const targetPath = enteredFolderPathRef.current
    const params = Object.fromEntries(searchParams)
    delete params.folder
    setSearchParams(params)
    if (targetPath) {
      // Wait for folder tiles to render, then scroll to the one we came from
      const tryScroll = () => {
        const el = document.querySelector(`[data-folder-path="${CSS.escape(targetPath)}"]`)
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        } else {
          // Tiles may not have rendered yet, retry
          requestAnimationFrame(tryScroll)
        }
      }
      requestAnimationFrame(tryScroll)
    }
  }, [searchParams, setSearchParams])

  const handleToggleGroupByFolders = useCallback(() => {
    const params = Object.fromEntries(searchParams)
    if (params.group === 'folders') {
      delete params.group
      delete params.folder
    } else {
      params.group = 'folders'
      delete params.folder
    }
    setSearchParams(params)
  }, [searchParams, setSearchParams])

  const handleImageClick = (imageId) => {
    // Push history state so back button closes lightbox
    window.history.pushState({ lightbox: true, imageId }, '')
    setLightboxIndex(imageId)
    // Keep sidebar visible to show image details
  }

  const handleLightboxClose = useCallback(() => {
    // Scroll to the image that was being viewed
    const imageId = lightboxIndex

    // Go back in history to trigger popstate which closes the lightbox
    // This ensures hardware back button and X button behave consistently
    if (window.history.state?.lightbox) {
      window.history.back()
    } else {
      // Fallback: close directly if no history state (shouldn't normally happen)
      setLightboxIndex(null)
    }

    // Use requestAnimationFrame to scroll after the lightbox closes and DOM updates
    requestAnimationFrame(() => {
      const imageElement = document.querySelector(`[data-image-id="${imageId}"]`)
      if (imageElement) {
        imageElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    })
  }, [lightboxIndex])

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
          filename: currentFilename,
          min_width: currentResolution?.width,
          min_height: currentResolution?.height,
          orientation: currentOrientation,
          min_duration: currentDuration?.min,
          max_duration: currentDuration?.max,
          import_source: currentFolder,
          sort: currentSort,
          page: nextPage,
          per_page: calculatePerPage(tileSize)
        })

        if (result.images.length > 0) {
          // Deduplicate to avoid showing same image twice
          const existingIds = new Set(images.map(img => img.id))
          const newImages = result.images.filter(img => !existingIds.has(img.id))

          if (newImages.length > 0) {
            setImages(prev => [...prev, ...newImages])
            setTotal(result.total)  // Update total in case it changed
            // Note: hasMore is computed by useEffect based on actual images.length
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
          initialFilename={currentFilename}
          initialResolution={currentResolution}
          initialOrientation={currentOrientation}
          initialDuration={currentDuration}
          initialGroupByFolders={groupByFolders}
          onToggleGroupByFolders={handleToggleGroupByFolders}
          total={total}
          stats={stats}
          lightboxMode={lightboxIndex !== null}
          lightboxHover={lightboxSidebarHover}
          onMouseLeave={() => setLightboxSidebarHover(false)}
        />

        {!sidebarOpen && <div className="swipe-hint" />}

        <main className="content with-sidebar">
          {currentFolder && (
            <div className="folder-breadcrumb">
              <button className="folder-back-btn" onClick={handleBackToFolders}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
                Folders
              </button>
              <span className="folder-breadcrumb-separator">/</span>
              <span className="folder-breadcrumb-name">{currentFolder === '__unfiled__' ? 'Unfiled' : currentFolder.split('/').pop()}</span>
            </div>
          )}
          {!loading && images.length === 0 ? (
            <div className="no-results">
              <h2>{groupByFolders && !currentFolder ? 'No folders found' : 'No images found'}</h2>
              <p>Try adjusting your search filters or add some directories to watch.</p>
            </div>
          ) : (
            <MasonryGrid
              images={images}
              onImageClick={handleImageClick}
              onFolderClick={handleFolderClick}
              onLoadMore={handleLoadMore}
              loading={loading}
              hasMore={hasMore}
              onImageUpdate={loadImages}
              isSelectable={selectionMode}
              selectedImages={selectedImages}
              onSelectImage={handleSelectImage}
              tileSize={tileSize}
            />
          )}

          {/* Floating controls: tile size slider, navigation, and select button */}
          <div className="floating-controls">
            <div className="tile-size-control" title="Adjust tile size">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tile-size-icon">
                <rect x="3" y="3" width="7" height="7" rx="1"/>
                <rect x="14" y="3" width="7" height="7" rx="1"/>
                <rect x="3" y="14" width="7" height="7" rx="1"/>
                <rect x="14" y="14" width="7" height="7" rx="1"/>
              </svg>
              <input
                type="range"
                min="1"
                max="5"
                value={tileSize}
                onChange={(e) => setTileSize(parseInt(e.target.value, 10))}
                className="tile-size-slider"
              />
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="tile-size-icon">
                <rect x="4" y="4" width="16" height="16" rx="2"/>
              </svg>
            </div>

            {/* Navigation controls */}
            {total > 50 && (
              <div className="nav-jump-control">
                <button
                  className="nav-jump-btn"
                  onClick={() => handleJumpByOffset(-100)}
                  disabled={isJumping || page === 1}
                  title="Jump back 100 images"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="18 15 12 9 6 15"/>
                  </svg>
                </button>
                <form onSubmit={handleJumpSubmit} className="nav-jump-form">
                  <span className="nav-position">
                    {((page - 1) * 50 + 1).toLocaleString()}-{Math.min(page * 50, total).toLocaleString()}
                  </span>
                  <span className="nav-separator">/</span>
                  <input
                    type="number"
                    className="nav-jump-input"
                    placeholder={total.toLocaleString()}
                    value={jumpInput}
                    onChange={(e) => setJumpInput(e.target.value)}
                    min="1"
                    max={total}
                    title="Type a number and press Enter to jump"
                  />
                </form>
                <button
                  className="nav-jump-btn"
                  onClick={() => handleJumpByOffset(100)}
                  disabled={isJumping || page * 50 >= total}
                  title="Jump forward 100 images"
                >
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="6 9 12 15 18 9"/>
                  </svg>
                </button>
              </div>
            )}

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
          </div>
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
  const [mobileReady, setMobileReady] = useState(false)
  const [showServerSetup, setShowServerSetup] = useState(false)
  const [servers, setServers] = useState([])
  const [serverStatuses, setServerStatuses] = useState({})

  // Initialize server configuration for mobile app
  useEffect(() => {
    async function initMobile() {
      const { isMobileApp, getServers, getActiveServer, setActiveServerId, pingAllServers } = await import('./serverManager')
      const { updateServerConfig } = await import('./api')

      if (isMobileApp()) {
        const serverList = await getServers()

        if (serverList.length === 0) {
          // No servers - show add server UI
          setShowServerSetup(true)
        } else {
          // Ping all servers in parallel
          const statuses = await pingAllServers(serverList)
          const onlineServers = serverList.filter(s => statuses[s.id] === 'online')

          if (onlineServers.length === 1) {
            // Exactly 1 online - auto-connect
            await setActiveServerId(onlineServers[0].id)
            await updateServerConfig()
          } else {
            // 0 or 2+ online - show selection with status
            setServers(serverList)
            setServerStatuses(statuses)
            setShowServerSetup(true)
          }
        }
      }
      setMobileReady(true)
    }
    initMobile()
  }, [])

  // Show loading while initializing mobile
  if (!mobileReady) {
    return (
      <div className="app loading-screen">
        <div className="loading-content">
          <h1>LocalBooru</h1>
          <p>Loading...</p>
        </div>
      </div>
    )
  }

  // Handle disconnect/switch server
  const handleDisconnect = () => {
    setShowServerSetup(true)
    // Re-fetch servers and their statuses
    import('./serverManager').then(async ({ getServers, pingAllServers }) => {
      const serverList = await getServers()
      setServers(serverList)
      if (serverList.length > 0) {
        const statuses = await pingAllServers(serverList)
        setServerStatuses(statuses)
      }
    })
  }

  // Show server setup/selection for mobile
  if (showServerSetup) {
    // If we have servers with statuses, show the selection screen
    if (servers.length > 0) {
      return (
        <ServerSelectScreen
          servers={servers}
          serverStatuses={serverStatuses}
          onConnect={() => setShowServerSetup(false)}
        />
      )
    }

    // Otherwise show the add server screen
    return (
      <div className="app server-setup-screen">
        <div className="server-setup-content">
          <h1>LocalBooru</h1>
          <p>Connect to a LocalBooru server to get started.</p>
          <ServerSettings onServerChange={() => {
            import('./serverManager').then(({ getActiveServer }) => {
              getActiveServer().then(server => {
                if (server) setShowServerSetup(false)
              })
            })
          }} />
        </div>
      </div>
    )
  }

  return (
    <>
      <TitleBar onSwitchServer={handleDisconnect} />
      <BrowserRouter>
        <BackButtonHandler />
        <Routes>
          <Route path="/" element={<Gallery />} />
          <Route path="/directories" element={<DirectoriesPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </BrowserRouter>
    </>
  )
}

// Handle hardware back button on mobile
function BackButtonHandler() {
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    if (!isMobileApp()) return

    const handleBackButton = CapacitorApp.addListener('backButton', ({ canGoBack }) => {
      // If there's history to go back to, use it
      if (canGoBack) {
        window.history.back()
      } else if (location.pathname !== '/') {
        // On a sub-page with no history, navigate to home
        navigate('/')
      } else {
        // On home with no history - minimize app (Android default behavior)
        CapacitorApp.minimizeApp()
      }
    })

    return () => {
      handleBackButton.then(listener => listener.remove())
    }
  }, [navigate, location.pathname])

  return null
}

export default App
