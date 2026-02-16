/**
 * LocalBooru API client - supports both local and multi-server mode
 */
import axios from 'axios'
import { isMobileApp, getActiveServer } from './serverManager'
import { validateServerCertificate, isHttps } from './sslPinning'

// Current server config (cached for synchronous access)
let currentServerUrl = null
let currentServerAuth = null
let currentCertFingerprint = null  // TLS certificate fingerprint for pinning
let certValidated = false  // Whether certificate has been validated this session

// Detect if running in Tauri
function isTauriApp() {
  return typeof window !== 'undefined' && window.__TAURI_INTERNALS__ !== undefined
}

// Get API URL - same origin when served from backend, fallback for dev
function getApiUrl() {
  // Mobile app mode - use configured server
  if (isMobileApp() && currentServerUrl) {
    return `${currentServerUrl}/api`
  }

  // Check if we're running on a Vite dev server
  const isDevServer = ['5173', '5174', '5175', '5210'].includes(window.location.port)

  // Tauri app - use Vite proxy in dev, direct URL in production
  if (isTauriApp()) {
    return isDevServer ? '/api' : 'http://127.0.0.1:8790/api'
  }

  if (isDevServer) {
    // Dev mode - Vite proxy forwards /api to backend (same-origin)
    return '/api'
  }

  // Production - frontend served from backend, use relative URL
  // This works for both localhost AND network access
  return '/api'
}

// Update the server configuration (call when server changes)
export async function updateServerConfig() {
  if (!isMobileApp()) return

  const server = await getActiveServer()
  if (server) {
    currentServerUrl = server.url
    currentCertFingerprint = server.certFingerprint || null
    certValidated = false  // Reset validation on server change
    if (server.username && server.password) {
      currentServerAuth = 'Basic ' + btoa(`${server.username}:${server.password}`)
    } else {
      currentServerAuth = null
    }
    // Update axios base URL
    api.defaults.baseURL = `${server.url}/api`

    // Validate certificate on first connection to HTTPS server
    if (isHttps(server.url) && currentCertFingerprint) {
      console.log('[API] Server uses HTTPS with certificate pinning')
    }
  } else {
    currentServerUrl = null
    currentServerAuth = null
    currentCertFingerprint = null
    certValidated = false
    api.defaults.baseURL = null
  }
}

// Check if connected to a server (for mobile app)
export function isServerConfigured() {
  if (!isMobileApp()) return true
  return currentServerUrl !== null
}

// Initialize API with base URL
const api = axios.create({
  baseURL: getApiUrl(),
  timeout: 60000  // 60s timeout for busy servers
})

// Add request interceptor for auth on mobile and certificate validation
api.interceptors.request.use(async (config) => {
  if (isMobileApp()) {
    // Add auth header if available
    if (currentServerAuth) {
      config.headers['Authorization'] = currentServerAuth
    }

    // Validate certificate on first request to HTTPS server with stored fingerprint
    if (isHttps(currentServerUrl) && currentCertFingerprint && !certValidated) {
      const result = await validateServerCertificate(currentServerUrl, currentCertFingerprint)
      if (!result.valid) {
        // Certificate validation failed - reject the request
        console.error('[API] Certificate validation failed:', result.error)
        throw new axios.Cancel(`Certificate validation failed: ${result.error}`)
      }
      certValidated = true
      console.log('[API] Certificate validated successfully')
    }
  }
  return config
})

// Track startup time to suppress errors during initialization
const startupTime = Date.now()
const STARTUP_GRACE_PERIOD = 10000 // 10 seconds

// Add response interceptor to show errors as popups (only for real errors)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const url = error.config?.url || 'unknown'
    const method = error.config?.method?.toUpperCase() || 'UNKNOWN'
    const status = error.response?.status
    const data = error.response?.data
    const isNetworkError = !error.response

    // Skip showing popup for transient/expected errors:
    // 1. During startup grace period (backend might still be initializing)
    const duringStartup = Date.now() - startupTime < STARTUP_GRACE_PERIOD
    // 2. Network errors (connection refused, backend not ready)
    // 3. 503 Service Unavailable (backend busy)
    // 4. Timeout errors
    const isTransient = isNetworkError || status === 503 || error.code === 'ECONNABORTED'

    // Only show popup for real errors after startup
    if (!duringStartup && !isTransient) {
      let message = `API Error: ${method} ${url}\n\nStatus: ${status || 'Network Error'}`

      if (data) {
        if (typeof data === 'string') {
          message += `\n\nResponse: ${data.substring(0, 500)}`
        } else if (data.detail) {
          message += `\n\nDetail: ${JSON.stringify(data.detail, null, 2)}`
        } else if (data.message) {
          message += `\n\nMessage: ${data.message}`
        } else {
          message += `\n\nResponse: ${JSON.stringify(data, null, 2).substring(0, 500)}`
        }
      }

      if (error.message && isNetworkError) {
        message += `\n\nError: ${error.message}`
      }

      // Show popup with error details
      alert(message)
    } else {
      // Log transient errors to console instead
      console.warn(`[API] Transient error (${duringStartup ? 'startup' : 'network'}): ${method} ${url}`, error.message)
    }

    // Still reject so calling code can handle it
    return Promise.reject(error)
  }
)

// Images API
export async function fetchImages({
  tags,
  exclude_tags,
  rating,
  favorites_only,
  directory_id,
  min_age,
  max_age,
  has_faces,
  timeframe,
  filename,
  min_width,
  min_height,
  orientation,
  min_duration,
  max_duration,
  import_source,
  sort = 'newest',
  page = 1,
  per_page = 50
}) {
  const params = new URLSearchParams()
  if (tags) params.append('tags', tags)
  if (exclude_tags) params.append('exclude_tags', exclude_tags)
  if (rating) params.append('rating', rating)
  if (favorites_only) params.append('favorites_only', 'true')
  if (directory_id) params.append('directory_id', directory_id)
  if (min_age !== undefined && min_age !== null) params.append('min_age', min_age)
  if (max_age !== undefined && max_age !== null) params.append('max_age', max_age)
  if (has_faces !== undefined && has_faces !== null) params.append('has_faces', has_faces)
  if (timeframe) params.append('timeframe', timeframe)
  if (filename) params.append('filename', filename)
  if (min_width !== undefined && min_width !== null) params.append('min_width', min_width)
  if (min_height !== undefined && min_height !== null) params.append('min_height', min_height)
  if (orientation) params.append('orientation', orientation)
  if (min_duration !== undefined && min_duration !== null) params.append('min_duration', min_duration)
  if (max_duration !== undefined && max_duration !== null) params.append('max_duration', max_duration)
  if (import_source) params.append('import_source', import_source)
  params.append('sort', sort)
  params.append('page', page)
  params.append('per_page', per_page)

  const response = await api.get(`/images?${params}`)
  return response.data
}

export async function fetchFolders({ directory_id, rating, favorites_only, tags } = {}) {
  const params = new URLSearchParams()
  if (directory_id) params.append('directory_id', directory_id)
  if (rating) params.append('rating', rating)
  if (favorites_only) params.append('favorites_only', 'true')
  if (tags) params.append('tags', tags)
  const response = await api.get(`/images/folders?${params}`)
  return response.data
}

export async function fetchImage(id) {
  const response = await api.get(`/images/${id}`)
  return response.data
}

export async function toggleFavorite(imageId) {
  const response = await api.post(`/images/${imageId}/favorite`)
  return response.data
}

export async function updateRating(imageId, rating) {
  const response = await api.patch(`/images/${imageId}/rating?rating=${rating}`)
  return response.data
}

// Alias for Lightbox compatibility
export const changeRating = updateRating

export async function deleteImage(imageId, deleteFile = false, directoryId = null) {
  let url = `/images/${imageId}?delete_file=${deleteFile}`
  if (directoryId) {
    url += `&directory_id=${directoryId}`
  }
  const response = await api.delete(url)
  return response.data
}

// Batch operations
export async function batchDeleteImages(imageIds, deleteFiles = false) {
  const response = await api.post('/images/batch/delete', {
    image_ids: imageIds,
    delete_files: deleteFiles
  })
  return response.data
}

export async function batchRetag(imageIds) {
  const response = await api.post('/images/batch/retag', {
    image_ids: imageIds
  })
  return response.data
}

export async function batchAgeDetect(imageIds) {
  const response = await api.post('/images/batch/age-detect', {
    image_ids: imageIds
  })
  return response.data
}

export async function batchMoveImages(imageIds, targetDirectoryId) {
  const response = await api.post('/images/batch/move', {
    image_ids: imageIds,
    target_directory_id: targetDirectoryId
  })
  return response.data
}

export async function applyImageAdjustments(imageId, { brightness, contrast, gamma }) {
  const response = await api.post(`/images/${imageId}/adjust`, {
    brightness,
    contrast,
    gamma
  })
  return response.data
}

export async function previewImageAdjustments(imageId, { brightness, contrast, gamma }) {
  const response = await api.post(`/images/${imageId}/preview-adjust`, {
    brightness,
    contrast,
    gamma
  })
  return response.data
}

export async function discardImagePreview(imageId) {
  const response = await api.delete(`/images/${imageId}/preview`)
  return response.data
}

// Video preview frames API with rate limiting
// Limit concurrent requests to prevent exhausting DB connection pool
const previewFrameQueue = {
  maxConcurrent: 3,
  running: 0,
  queue: [],

  async enqueue(fn) {
    return new Promise((resolve, reject) => {
      this.queue.push({ fn, resolve, reject })
      this.process()
    })
  },

  async process() {
    if (this.running >= this.maxConcurrent || this.queue.length === 0) return

    const { fn, resolve, reject } = this.queue.shift()
    this.running++

    try {
      const result = await fn()
      resolve(result)
    } catch (err) {
      reject(err)
    } finally {
      this.running--
      this.process()
    }
  }
}

export async function fetchPreviewFrames(imageId, directoryId = null) {
  return previewFrameQueue.enqueue(async () => {
    const params = directoryId ? `?directory_id=${directoryId}` : ''
    const response = await api.get(`/images/${imageId}/preview-frames${params}`)
    return response.data
  })
}

// Tags API
export async function fetchTags({ q, category, page = 1, per_page = 50, sort = 'count' } = {}) {
  const params = new URLSearchParams()
  if (q) params.append('q', q)
  if (category) params.append('category', category)
  params.append('page', page)
  params.append('per_page', per_page)
  params.append('sort', sort)

  const response = await api.get(`/tags?${params}`)
  return response.data
}

export async function searchTags(query, limit = 10) {
  const response = await api.get(`/tags/autocomplete?q=${encodeURIComponent(query)}&limit=${limit}`)
  return response.data
}

export async function getTagStats() {
  const response = await api.get('/tags/stats/overview')
  return response.data
}

// Directories API (with caching to avoid redundant calls)
let directoriesCache = null
let directoriesCacheTime = 0
const DIRECTORIES_CACHE_TTL = 5000 // 5 seconds

export async function fetchDirectories(forceRefresh = false) {
  const now = Date.now()
  if (!forceRefresh && directoriesCache && (now - directoriesCacheTime) < DIRECTORIES_CACHE_TTL) {
    return directoriesCache
  }
  const response = await api.get('/directories')
  directoriesCache = response.data
  directoriesCacheTime = now
  return response.data
}

export function invalidateDirectoriesCache() {
  directoriesCache = null
  directoriesCacheTime = 0
}

export async function addDirectory(path, options = {}) {
  const response = await api.post('/directories', {
    path,
    name: options.name,
    recursive: options.recursive ?? true,
    auto_tag: options.auto_tag ?? true
  })
  invalidateDirectoriesCache()
  return response.data
}

export async function addParentDirectory(path, options = {}) {
  const response = await api.post('/directories/add-parent', {
    path,
    recursive: options.recursive ?? true,
    auto_tag: options.auto_tag ?? true
  })
  invalidateDirectoriesCache()
  return response.data
}

export async function updateDirectory(id, updates) {
  const response = await api.patch(`/directories/${id}`, updates)
  invalidateDirectoriesCache()
  return response.data
}

export async function updateDirectoryPath(id, newPath) {
  const response = await api.patch(`/directories/${id}/path`, { new_path: newPath })
  invalidateDirectoriesCache()
  return response.data
}

export async function removeDirectory(id, keepImages = false) {
  const response = await api.delete(`/directories/${id}?keep_images=${keepImages}`)
  invalidateDirectoriesCache()
  return response.data
}

export async function bulkDeleteDirectories(directoryIds, keepImages = false) {
  // Long timeout for bulk operations with many images
  const response = await api.post('/directories/bulk-delete',
    { directory_ids: directoryIds, keep_images: keepImages },
    { timeout: 600000 }  // 10 minute timeout for large deletions
  )
  invalidateDirectoriesCache()
  return response.data
}

export async function scanDirectory(id) {
  const response = await api.post(`/directories/${id}/scan`)
  return response.data
}

export async function pruneDirectory(id, dumpsterPath = null) {
  const response = await api.post(`/directories/${id}/prune`, {
    dumpster_path: dumpsterPath
  })
  return response.data
}

// Library API
export async function getLibraryStats() {
  const response = await api.get('/library/stats')
  return response.data
}

export async function getQueueStatus() {
  const response = await api.get('/library/queue')
  return response.data
}

export async function retryFailedTasks() {
  const response = await api.post('/library/queue/retry-failed')
  return response.data
}

export async function verifyFiles() {
  const response = await api.post('/library/verify-files')
  return response.data
}

export async function tagUntagged(directoryId = null) {
  const params = directoryId ? { directory_id: directoryId } : {}
  const response = await api.post('/library/tag-untagged', null, { params })
  return response.data
}

export async function clearDirectoryTagQueue(directoryId) {
  const response = await api.delete(`/library/queue/pending/directory/${directoryId}`)
  return response.data
}

export async function clearPendingTasks() {
  const response = await api.delete('/library/queue/pending')
  return response.data
}

export async function getQueuePaused() {
  const response = await api.get('/library/queue/paused')
  return response.data
}

export async function pauseQueue() {
  const response = await api.post('/library/queue/pause')
  return response.data
}

export async function resumeQueue() {
  const response = await api.post('/library/queue/resume')
  return response.data
}

export async function cleanMissingFiles() {
  const response = await api.post('/library/clean-missing')
  return response.data
}

export async function verifyDirectoryFiles(directoryId) {
  const response = await api.post(`/directories/${directoryId}/verify`)
  return response.data
}

export async function repairDirectoryPaths(directoryId) {
  const response = await api.post(`/directories/${directoryId}/repair`)
  return response.data
}

export async function bulkVerifyDirectories(directoryIds) {
  const response = await api.post('/directories/bulk-verify', { directory_ids: directoryIds })
  return response.data
}

export async function bulkRepairDirectories(directoryIds) {
  const response = await api.post('/directories/bulk-repair', { directory_ids: directoryIds })
  return response.data
}

export async function detectAgesRetrospective() {
  const response = await api.post('/library/detect-ages')
  return response.data
}

// Settings API
export async function getSettings() {
  const response = await api.get('/settings')
  return response.data
}

export async function getAgeDetectionStatus() {
  const response = await api.get('/settings/age-detection/status')
  return response.data
}

export async function toggleAgeDetection(enabled) {
  const response = await api.post('/settings/age-detection/toggle', { enabled })
  return response.data
}

export async function installAgeDetection() {
  const response = await api.post('/settings/age-detection/install')
  return response.data
}

// Family Mode API
export async function getFamilyModeStatus() {
  const response = await api.get('/settings/family-mode')
  return response.data
}

export async function configureFamilyMode(config) {
  const response = await api.post('/settings/family-mode', config)
  return response.data
}

export async function unlockFamilyMode(pin) {
  const response = await api.post('/settings/family-mode/unlock', { pin })
  invalidateDirectoriesCache()
  return response.data
}

export async function lockFamilyMode() {
  const response = await api.post('/settings/family-mode/lock')
  invalidateDirectoriesCache()
  return response.data
}

// Network API
export async function getNetworkConfig() {
  const response = await api.get('/network')
  return response.data
}

export async function getQRData() {
  const response = await api.get('/network/qr-data')
  return response.data
}

export async function updateNetworkConfig(config) {
  const response = await api.post('/network', config)
  return response.data
}

export async function testPort(port) {
  const response = await api.post('/network/test-port', { port })
  return response.data
}

export async function discoverUPnP() {
  const response = await api.post('/network/upnp/discover')
  return response.data
}

export async function openUPnPPort(externalPort, internalPort = null, description = 'LocalBooru') {
  const response = await api.post('/network/upnp/open-port', {
    external_port: externalPort,
    internal_port: internalPort || externalPort,
    description
  })
  return response.data
}

export async function closeUPnPPort(externalPort) {
  const response = await api.delete(`/network/upnp/close-port/${externalPort}`)
  return response.data
}

export async function getUPnPMappings() {
  const response = await api.get('/network/upnp/mappings')
  return response.data
}

// Users API
export async function listUsers() {
  const response = await api.get('/users')
  return response.data
}

export async function createUser(user) {
  const response = await api.post('/users', user)
  return response.data
}

export async function updateUser(id, updates) {
  const response = await api.patch(`/users/${id}`, updates)
  return response.data
}

export async function deleteUser(id) {
  const response = await api.delete(`/users/${id}`)
  return response.data
}

// Utility functions
export function getMediaUrl(path) {
  if (!path) return ''
  if (path.startsWith('http')) return path

  // On mobile, prepend server URL for relative paths
  if (isMobileApp() && currentServerUrl) {
    const cleanPath = path.startsWith('/') ? path : `/${path}`
    return `${currentServerUrl}${cleanPath}`
  }

  // Dev mode - serve media directly from backend (proper range request support)
  // Only VTT files in <track> elements need the Vite proxy (same-origin requirement)
  const isDevServer = ['5173', '5174', '5175', '5210'].includes(window.location.port)
  if (isDevServer) {
    const cleanPath = path.startsWith('/') ? path : `/${path}`
    return `http://127.0.0.1:8790${cleanPath}`
  }

  // On web, relative URLs work fine
  return path
}

export function isVideo(filename) {
  if (!filename) return false
  const ext = filename.split('.').pop()?.toLowerCase()
  return ['mp4', 'webm', 'mov', 'avi', 'mkv'].includes(ext)
}

export function isAnimated(filename) {
  if (!filename) return false
  const ext = filename.split('.').pop()?.toLowerCase()
  return ['gif', 'apng', 'webp'].includes(ext)
}

// Subscribe to library events via Server-Sent Events
export function subscribeToLibraryEvents(onEvent, onError) {
  const apiUrl = getApiUrl()
  // apiUrl already includes /api, so just append the path
  const eventSource = new EventSource(`${apiUrl}/library/events`)

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onEvent(data)
    } catch (e) {
      console.error('[SSE] Failed to parse event:', e)
    }
  }

  eventSource.onerror = (error) => {
    console.error('[SSE] Connection error:', error)
    if (onError) onError(error)
  }

  // Return cleanup function
  return () => {
    eventSource.close()
  }
}

// Migration API
export async function getMigrationInfo() {
  const response = await api.get('/settings/migration')
  return response.data
}

export async function getMigrationDirectories(mode) {
  const response = await api.get('/settings/migration/directories', { params: { mode } })
  return response.data
}

export async function validateMigration(mode, directoryIds = null) {
  const payload = { mode }
  if (directoryIds && directoryIds.length > 0) {
    payload.directory_ids = directoryIds
  }
  const response = await api.post('/settings/migration/validate', payload)
  return response.data
}

export async function startMigration(mode, directoryIds = null) {
  const payload = { mode }
  if (directoryIds && directoryIds.length > 0) {
    payload.directory_ids = directoryIds
  }
  const response = await api.post('/settings/migration/start', payload)
  return response.data
}

export async function getMigrationStatus() {
  const response = await api.get('/settings/migration/status')
  return response.data
}

export async function cleanupMigration(mode) {
  const response = await api.post('/settings/migration/cleanup', { mode })
  return response.data
}

export async function deleteSourceData(mode) {
  const response = await api.post('/settings/migration/delete-source', { mode })
  return response.data
}

export async function verifyMigration(mode) {
  const response = await api.post('/settings/migration/verify', { mode })
  return response.data
}

// Import API (add directories to existing database)
export async function validateImport(mode, directoryIds) {
  const response = await api.post('/settings/migration/import/validate', {
    mode,
    directory_ids: directoryIds
  })
  return response.data
}

export async function startImport(mode, directoryIds) {
  const response = await api.post('/settings/migration/import/start', {
    mode,
    directory_ids: directoryIds
  })
  return response.data
}

// Subscribe to migration events via Server-Sent Events
export function subscribeToMigrationEvents(onEvent) {
  const apiUrl = getApiUrl()
  const eventSource = new EventSource(`${apiUrl}/settings/migration/events`)

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onEvent(data)
    } catch (e) {
      console.error('[Migration SSE] Failed to parse event:', e)
    }
  }

  eventSource.onerror = (error) => {
    console.error('[Migration SSE] Connection error:', error)
  }

  return () => {
    eventSource.close()
  }
}

// Collections API
export async function fetchCollections() {
  const response = await api.get('/collections')
  return response.data
}

export async function createCollection(name, description = null) {
  const response = await api.post('/collections', { name, description })
  return response.data
}

export async function fetchCollection(id, page = 1, perPage = 50) {
  const response = await api.get(`/collections/${id}?page=${page}&per_page=${perPage}`)
  return response.data
}

export async function updateCollection(id, updates) {
  const response = await api.patch(`/collections/${id}`, updates)
  return response.data
}

export async function deleteCollection(id) {
  const response = await api.delete(`/collections/${id}`)
  return response.data
}

export async function addToCollection(collectionId, imageIds) {
  const response = await api.post(`/collections/${collectionId}/items`, { image_ids: imageIds })
  return response.data
}

export async function removeFromCollection(collectionId, imageIds) {
  const response = await api.delete(`/collections/${collectionId}/items`, { data: { image_ids: imageIds } })
  return response.data
}

export async function reorderCollection(collectionId, imageIds) {
  const response = await api.patch(`/collections/${collectionId}/items/reorder`, { image_ids: imageIds })
  return response.data
}

// Saved Searches API
export async function getSavedSearches() {
  const response = await api.get('/settings/saved-searches')
  return response.data
}

export async function createSavedSearch(name, filters) {
  const response = await api.post('/settings/saved-searches', { name, filters })
  return response.data
}

export async function deleteSavedSearch(searchId) {
  const response = await api.delete(`/settings/saved-searches/${searchId}`)
  return response.data
}

// Watch History API
export async function savePlaybackPosition(imageId, position, duration, directoryId = null) {
  const body = { position, duration }
  if (directoryId) body.directory_id = directoryId
  const response = await api.post(`/watch-history/${imageId}`, body)
  return response.data
}

export async function getContinueWatching() {
  const response = await api.get('/watch-history/continue-watching')
  return response.data
}

export async function getPlaybackPosition(imageId) {
  const response = await api.get(`/watch-history/${imageId}`)
  return response.data
}

export async function clearWatchHistory(imageId = null) {
  if (imageId) {
    const response = await api.delete(`/watch-history/${imageId}`)
    return response.data
  }
  const response = await api.delete('/watch-history')
  return response.data
}

// Video Playback Config API (auto-advance, etc.)
export async function getVideoPlaybackConfig() {
  const response = await api.get('/settings/video-playback')
  return response.data
}

export async function updateVideoPlaybackConfig(config) {
  const response = await api.post('/settings/video-playback', config)
  return response.data
}

// Optical Flow Interpolation API
export async function getOpticalFlowConfig() {
  const response = await api.get('/settings/optical-flow')
  return response.data
}

export async function updateOpticalFlowConfig(config) {
  const response = await api.post('/settings/optical-flow', config)
  return response.data
}

export async function playVideoInterpolated(filePath, startPosition = 0, qualityPreset = null) {
  // Longer timeout since buffering can take time
  const response = await api.post('/settings/optical-flow/play', {
    file_path: filePath,
    start_position: startPosition,
    quality_preset: qualityPreset
  }, {
    timeout: 60000  // 60 second timeout for initial buffering
  })
  return response.data
}

export async function stopInterpolatedStream() {
  const response = await api.post('/settings/optical-flow/stop')
  return response.data
}

// SVP (SmoothVideo Project) Interpolation API
export async function getSVPConfig() {
  const response = await api.get('/settings/svp')
  return response.data
}

export async function updateSVPConfig(config) {
  const response = await api.post('/settings/svp', config)
  return response.data
}

export async function playVideoSVP(filePath, startPosition = 0, qualityPreset = null) {
  // Longer timeout since SVP processing can take time
  const response = await api.post('/settings/svp/play', {
    file_path: filePath,
    start_position: startPosition,
    quality_preset: qualityPreset
  }, {
    timeout: 60000  // 60 second timeout for initial buffering
  })
  return response.data
}

export async function stopSVPStream() {
  const response = await api.post('/settings/svp/stop')
  return response.data
}

// Simple FFmpeg-based transcoding (fallback when SVP/OpticalFlow not available)
export async function playVideoTranscode(filePath, startPosition = 0, qualityPreset = null) {
  const response = await api.post('/settings/transcode/play', {
    file_path: filePath,
    start_position: startPosition,
    quality_preset: qualityPreset
  }, {
    timeout: 60000  // 60 second timeout for buffering
  })
  return response.data
}

export async function stopTranscodeStream() {
  const response = await api.post('/settings/transcode/stop')
  return response.data
}

// Get video info including VFR detection
export async function getVideoInfo(filePath) {
  const response = await api.post('/settings/video-info', {
    file_path: filePath
  })
  return response.data
}

// Whisper Subtitle API
export async function getWhisperConfig() {
  const response = await api.get('/settings/whisper')
  return response.data
}

export async function updateWhisperConfig(config) {
  const response = await api.post('/settings/whisper', config)
  return response.data
}

export async function installWhisper() {
  const response = await api.post('/settings/whisper/install')
  return response.data
}

export async function generateSubtitles(filePath, language = null, task = null, startPosition = 0) {
  const response = await api.post('/settings/whisper/generate', {
    file_path: filePath,
    language,
    task,
    start_position: startPosition
  })
  return response.data
}

export async function stopSubtitles() {
  const response = await api.post('/settings/whisper/stop')
  return response.data
}

export function subscribeToSubtitleEvents(streamId, onEvent) {
  const apiUrl = getApiUrl()
  const eventSource = new EventSource(`${apiUrl}/settings/whisper/events/${streamId}`)

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onEvent(data)
    } catch (e) {
      console.error('[Subtitle SSE] Failed to parse event:', e)
    }
  }

  eventSource.onerror = (error) => {
    console.error('[Subtitle SSE] Connection error:', error)
  }

  return () => {
    eventSource.close()
  }
}

// Share Stream API
export async function createShareSession(imageId, directoryId = null) {
  const response = await api.post('/share/create', { image_id: imageId, directory_id: directoryId })
  return response.data
}

export async function stopShareSession(token) {
  const response = await api.delete(`/share/${token}`)
  return response.data
}

export async function syncShareState(token, state) {
  const response = await api.post(`/share/${token}/sync`, state)
  return response.data
}

export async function getShareInfo(token) {
  const response = await api.get(`/share/${token}/info`)
  return response.data
}

export async function getShareNetworkInfo() {
  const response = await api.get('/share/network-info')
  return response.data
}

export function subscribeToShareEvents(token, onEvent) {
  // Use page origin for absolute URL (viewer may be on different machine)
  const eventSource = new EventSource(`${window.location.origin}/api/share/${token}/events`)

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onEvent(data)
    } catch (e) {
      console.error('[Share SSE] Failed to parse event:', e)
    }
  }

  eventSource.onerror = (error) => {
    console.error('[Share SSE] Connection error:', error)
  }

  return () => {
    eventSource.close()
  }
}

export function getShareHlsUrl(token) {
  // Use page origin for absolute URL (viewer may be on different machine)
  return `${window.location.origin}/api/share/${token}/hls/playlist.m3u8`
}

// Cast API (Chromecast & DLNA)
export async function getCastConfig() {
  const response = await api.get('/settings/cast')
  return response.data
}

export async function updateCastConfig(config) {
  const response = await api.post('/settings/cast', config)
  return response.data
}

export async function installCastDeps() {
  const response = await api.post('/settings/cast/install')
  return response.data
}

export async function getCastDevices() {
  const response = await api.get('/cast/devices')
  return response.data
}

export async function refreshCastDevices() {
  const response = await api.post('/cast/devices/refresh')
  return response.data
}

export async function castPlay(deviceId, filePath, imageId = null, directoryId = null) {
  const response = await api.post('/cast/play', {
    device_id: deviceId,
    file_path: filePath,
    image_id: imageId,
    directory_id: directoryId,
  })
  return response.data
}

export async function castControl(action, value = null) {
  const response = await api.post('/cast/control', { action, value })
  return response.data
}

export async function castStop() {
  const response = await api.post('/cast/stop')
  return response.data
}

export function subscribeToCastEvents(onEvent) {
  const apiUrl = getApiUrl()
  const eventSource = new EventSource(`${apiUrl}/cast/status`)

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onEvent(data)
    } catch (e) {
      console.error('[Cast SSE] Failed to parse event:', e)
    }
  }

  eventSource.onerror = (error) => {
    console.error('[Cast SSE] Connection error:', error)
  }

  return () => {
    eventSource.close()
  }
}

// Addons API
export async function getAddons() {
  const response = await api.get('/addons')
  return response.data
}

export async function getAddon(id) {
  const response = await api.get(`/addons/${id}`)
  return response.data
}

export async function installAddon(id) {
  const response = await api.post(`/addons/${id}/install`)
  return response.data
}

export async function uninstallAddon(id) {
  const response = await api.post(`/addons/${id}/uninstall`)
  return response.data
}

export async function startAddon(id) {
  const response = await api.post(`/addons/${id}/start`)
  return response.data
}

export async function stopAddon(id) {
  const response = await api.post(`/addons/${id}/stop`)
  return response.data
}

// Health check (used for Tauri startup readiness polling)
export async function healthCheck() {
  const baseUrl = isTauriApp() ? 'http://127.0.0.1:8790' : ''
  const response = await axios.get(`${baseUrl}/health`, { timeout: 2000 })
  return response.data
}

// Utility endpoints
export async function getFileDimensions(filePath) {
  const response = await api.get('/settings/util/dimensions', {
    params: { file_path: filePath }
  })
  return response.data
}

export async function getFileInfo(filePath) {
  const response = await api.get('/images/media/file-info', {
    params: { path: filePath }
  })
  return response.data
}
