import { imageIdentityKey } from './imageAdjustments.js'

const MIN_RETRY_DELAY_MS = 250
const MAX_RETRY_DELAY_MS = 30_000

export function createViewRequestOwner() {
  let activeView = null
  let generation = 0

  return {
    activate(view) {
      if (view !== activeView) {
        activeView = view
        generation += 1
      }
    },
    begin(view) {
      if (view !== activeView) return { view, generation: null }
      generation += 1
      return { view, generation }
    },
    owns(request) {
      return request.view === activeView && request.generation === generation
    },
  }
}

export function mergeFirstPage(existing, incoming) {
  const incomingKeys = new Set(incoming.map(imageIdentityKey))
  return [...incoming, ...existing.filter(image => !incomingKeys.has(imageIdentityKey(image)))]
}

export function nextLoadRetryDelay(currentDelay, succeeded) {
  if (succeeded) return MIN_RETRY_DELAY_MS
  return Math.min(MAX_RETRY_DELAY_MS, Math.max(MIN_RETRY_DELAY_MS, currentDelay) * 2)
}

export function isUnexpectedEmptyPage({ append, pageLength, total, loaded }) {
  return append && pageLength === 0 && total > loaded
}

export function shouldRefreshForLibraryEvent(event) {
  return libraryRefreshMode(event) !== null
}

export function libraryRefreshMode(event) {
  if (event?.type === 'image_added') return 'merge'
  if (event?.type === 'task_completed' && event?.data?.task_type === 'scan_directory') {
    return 'replace'
  }
  return null
}

export function mergeAuthoritativePages(pageResults) {
  const seen = new Set()
  return pageResults.flatMap(result => result.images || []).filter(image => {
    const key = imageIdentityKey(image)
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

export function completeAuthoritativeRefresh({ completed, generation, latest, refreshed }) {
  const nextCompleted = refreshed ? Math.max(completed, generation) : completed
  return {
    completed: nextCompleted,
    pending: latest > nextCompleted,
  }
}

export function reconcileAuthoritativeGallery(images, currentLocator) {
  if (!currentLocator) return { images, currentLocator: null }
  const currentKey = imageIdentityKey(currentLocator)
  return {
    images,
    currentLocator: images.some(image => imageIdentityKey(image) === currentKey)
      ? currentLocator
      : null,
  }
}

function normalizedScopeValue(value) {
  return value == null || value === '' ? null : String(value)
}

export function galleryScopePreservesFolder(current, next) {
  return normalizedScopeValue(current.directoryId) === normalizedScopeValue(next.directoryId)
    && normalizedScopeValue(current.libraryId) === normalizedScopeValue(next.libraryId)
}

export async function refreshGroupedFolderCatalog({ groupByFolders, currentFolder, loadFolders }) {
  if (!groupByFolders || currentFolder) return false
  await loadFolders()
  return true
}
