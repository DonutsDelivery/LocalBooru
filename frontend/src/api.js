/**
 * LocalBooru API client - simplified for single-user local use
 */
import axios from 'axios'

// Get API URL - same origin when served from backend, fallback for dev
function getApiUrl() {
  // In dev mode with vite, use explicit URL
  if (import.meta.env.DEV) {
    return import.meta.env.VITE_API_URL || 'http://127.0.0.1:8790'
  }
  // In production, frontend is served from backend - use same origin
  return ''
}

// Initialize API with base URL
const api = axios.create({
  baseURL: getApiUrl(),
  timeout: 30000
})

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

export async function deleteImage(imageId, deleteFile = false) {
  const response = await api.delete(`/images/${imageId}?delete_file=${deleteFile}`)
  return response.data
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

// Directories API
export async function fetchDirectories() {
  const response = await api.get('/directories')
  return response.data
}

export async function addDirectory(path, options = {}) {
  const response = await api.post('/directories', {
    path,
    name: options.name,
    recursive: options.recursive ?? true,
    auto_tag: options.auto_tag ?? true
  })
  return response.data
}

export async function updateDirectory(id, updates) {
  const response = await api.patch(`/directories/${id}`, updates)
  return response.data
}

export async function removeDirectory(id, removeImages = false) {
  const response = await api.delete(`/directories/${id}?remove_images=${removeImages}`)
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

export async function tagUntagged() {
  const response = await api.post('/library/tag-untagged')
  return response.data
}

export async function clearPendingTasks() {
  const response = await api.delete('/library/queue/pending')
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

// Utility functions
export function getMediaUrl(path) {
  if (!path) return ''
  if (path.startsWith('http')) return path
  // API URL will be resolved at runtime
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
