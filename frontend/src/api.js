/**
 * LocalBooru API client - supports both local and multi-server mode
 */
import axios from 'axios'
import { isMobileApp, getActiveServer } from './serverManager'

// Current server config (cached for synchronous access)
let currentServerUrl = null
let currentServerAuth = null

// Get API URL - same origin when served from backend, fallback for dev
function getApiUrl() {
  // Mobile app mode - use configured server
  if (isMobileApp() && currentServerUrl) {
    return `${currentServerUrl}/api`
  }

  // Check if we're running on localhost with Vite dev server (port 5173/5174)
  const isDevServer = window.location.port === '5173' || window.location.port === '5174'

  if (isDevServer) {
    // Dev mode - Vite dev server, need to point to backend
    return 'http://127.0.0.1:8790/api'
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
    if (server.username && server.password) {
      currentServerAuth = 'Basic ' + btoa(`${server.username}:${server.password}`)
    } else {
      currentServerAuth = null
    }
    // Update axios base URL
    api.defaults.baseURL = `${server.url}/api`
  } else {
    currentServerUrl = null
    currentServerAuth = null
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

// Add request interceptor for auth on mobile
api.interceptors.request.use((config) => {
  if (isMobileApp() && currentServerAuth) {
    config.headers['Authorization'] = currentServerAuth
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
  params.append('sort', sort)
  params.append('page', page)
  params.append('per_page', per_page)

  const response = await api.get(`/images?${params}`)
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
    // Remove leading slash if present to avoid double slashes
    const cleanPath = path.startsWith('/') ? path : `/${path}`
    return `${currentServerUrl}${cleanPath}`
  }

  // Dev mode - Vite dev server needs full URL to backend
  const isDevServer = window.location.port === '5173' || window.location.port === '5174'
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
  return ['mp4', 'webm', 'mov', 'avi'].includes(ext)
}

export function isAnimated(filename) {
  if (!filename) return false
  const ext = filename.split('.').pop()?.toLowerCase()
  return ['gif', 'apng', 'webp'].includes(ext)
}

// Subscribe to library events via Server-Sent Events
export function subscribeToLibraryEvents(onEvent) {
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

// Utility endpoints
export async function getFileDimensions(filePath) {
  const response = await api.get('/settings/util/dimensions', {
    params: { file_path: filePath }
  })
  return response.data
}
