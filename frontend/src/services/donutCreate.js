import { apiClient, getApiUrl, invalidateDirectoriesCache } from '../api'

const CREATE_API = '/addons/donut-create/api/create'

function checkRequest(signal) {
  if (signal?.aborted) throw new DOMException('The server connection changed.', 'AbortError')
}

export async function getCreateStatus(signal) {
  const response = await apiClient.get(`${CREATE_API}/status`, { signal })
  checkRequest(signal)
  return response.data
}

export async function saveCreateConfig(config, signal) {
  const response = await apiClient.post(`${CREATE_API}/config`, config, { signal })
  checkRequest(signal)
  window.dispatchEvent(new CustomEvent('donut-create-backend-changed'))
  return response.data
}

export async function setupCreateBackend(options, signal) {
  const response = await apiClient.post(`${CREATE_API}/setup`, options, { signal })
  checkRequest(signal)
  return response.data
}

export async function cancelCreateSetup(signal) {
  const response = await apiClient.post(`${CREATE_API}/cancel-setup`, undefined, { signal })
  checkRequest(signal)
  return response.data
}

export async function startCreateBackend(signal) {
  const response = await apiClient.post(`${CREATE_API}/backend/start`, undefined, { signal })
  checkRequest(signal)
  return response.data
}

export async function stopCreateBackend(signal) {
  const response = await apiClient.post(`${CREATE_API}/backend/stop`, undefined, { signal })
  checkRequest(signal)
  return response.data
}

export async function createStudioSession(signal) {
  const response = await apiClient.post(`${CREATE_API}/sessions`, { workspace: true }, { signal })
  checkRequest(signal)
  return response.data
}

export async function getStudioSession(sessionId, signal) {
  const response = await apiClient.get(`${CREATE_API}/sessions/${encodeURIComponent(sessionId)}`, { signal })
  checkRequest(signal)
  return response.data
}

export async function cancelStudioJobs(sessionId, signal) {
  const response = await apiClient.post(`${CREATE_API}/sessions/${encodeURIComponent(sessionId)}/cancel`, undefined, { signal })
  checkRequest(signal)
  return response.data
}

// The session ID is an expiring capability. Keep normal account credentials out
// of iframe and preview URLs and retain the active server's API routing.
export function studioUrl(sessionId) {
  return `${getApiUrl()}/create/studio/${encodeURIComponent(sessionId)}/`
}

export function outputUrl(sessionId, outputId) {
  return `${getApiUrl()}/create/output/${encodeURIComponent(sessionId)}/${encodeURIComponent(outputId)}`
}

export async function importStudioOutput(sessionId, outputId, destination, signal) {
  const response = await apiClient.post('/create/import', {
    session_id: sessionId,
    output_id: outputId,
    library_id: destination.libraryId,
    directory_id: destination.directoryId,
  }, { signal })
  checkRequest(signal)
  window.dispatchEvent(new CustomEvent('donut-create-imported', { detail: response.data }))
  return response.data
}

export async function createOutputDirectory(libraryId, signal) {
  const response = await apiClient.post('/create/output-directory', libraryId ? { library_id: libraryId } : {}, { signal })
  checkRequest(signal)
  invalidateDirectoriesCache()
  window.dispatchEvent(new CustomEvent('donut-create-directory-created', { detail: response.data }))
  return response.data
}

export async function getImageWorkflow(locator, summary = false, signal) {
  const response = await apiClient.get('/create/workflow', { signal, params: {
    library_id: locator.libraryId,
    directory_id: locator.directoryId,
    image_id: locator.imageId,
    ...(locator.fileHash ? { file_hash: locator.fileHash } : {}),
    ...(summary ? { summary: true } : {}),
  } })
  checkRequest(signal)
  return response.data
}

export function openImageWorkflow(workflow, locator, label) {
  if (!workflow || typeof workflow !== 'object' || Array.isArray(workflow)) throw new Error('This image does not contain a saved workflow.')
  window.dispatchEvent(new CustomEvent('donut-create-load-workflow', { detail: { workflow, source: { ...locator, label } } }))
}

export function createErrorMessage(error) {
  const detail = error?.response?.data?.error
    || error?.response?.data?.detail
    || error?.response?.data?.message
  return typeof detail === 'string' ? detail : error?.message || 'The creator could not complete this action.'
}

export function formatCreateBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return 'Unknown'
  if (bytes < 1_000_000_000) return `${Math.round(bytes / 1_000_000)} MB`
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`
}

export function createLibraryOptions(libraries) {
  return libraries
    .filter(library => library.mounted && library.accessible !== false && library.read_only !== true && library.remote !== true)
    .map(library => ({ id: library.uuid, label: library.name, primary: library.is_primary === true }))
    .sort((left, right) => left.label.localeCompare(right.label, undefined, { numeric: true }))
}

export function createDirectoryOptions(directories, libraries) {
  const available = new Map(createLibraryOptions(libraries).map(library => [library.id, library.label]))
  return directories
    .filter(directory => available.has(directory.library_id)
      && directory.show_images === true
      && directory.enabled !== false
      && directory.path_exists !== false
      && directory.read_only !== true)
    .map(directory => ({
      key: `${directory.library_id}:${directory.id}`,
      libraryId: directory.library_id,
      directoryId: directory.id,
      label: `${available.get(directory.library_id)} · ${directory.name}`,
    }))
    .sort((left, right) => left.label.localeCompare(right.label, undefined, { numeric: true }))
}
