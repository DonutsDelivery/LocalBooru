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
  return event?.type === 'image_added' || (
    event?.type === 'task_completed' && event?.data?.task_type === 'scan_directory'
  )
}

export async function refreshGroupedFolderCatalog({ groupByFolders, currentFolder, loadFolders }) {
  if (!groupByFolders || currentFolder) return false
  await loadFolders()
  return true
}
