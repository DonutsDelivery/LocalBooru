export const CURATION_BATCH_SIZE = 50

export function getCurationRecoveryMode({ active, complete, current, loading, refillError }) {
  if (!active || complete || current) return null
  if (refillError) return 'error'
  if (loading) return 'loading'
  return null
}

export function createCurationActionLock() {
  let locked = false
  return {
    tryAcquire() {
      if (locked) return false
      locked = true
      return true
    },
    release() {
      locked = false
    },
  }
}

export function commitCurationAction(state, kind, item, countedDate) {
  return {
    ...state,
    queue: state.queue.slice(1),
    lastAction: { kind, item, countedDate },
    processed: state.processed + 1,
    busy: true,
    loading: true,
    complete: false,
    refillError: null,
  }
}

export function markCurationRefillFailure(state, error) {
  return {
    ...state,
    busy: false,
    loading: false,
    refillError: error?.message || String(error),
  }
}

export const CURATION_REFILL_AT = 10

export function imageLocatorKey(image) {
  return `${image.library_id || 'primary'}:${image.directory_id}:${image.id}`
}

export function buildCurationQuery(filters) {
  return {
    ...filters,
    favorites_only: false,
    exclude_favorites: true,
    page: 1,
    per_page: CURATION_BATCH_SIZE,
  }
}

export function seedCandidates(images) {
  return (images || []).filter(image => !image._isFolder && !image.is_favorite && image.directory_id)
}

export function mergeCandidates(queue, incoming) {
  const seen = new Set(queue.map(imageLocatorKey))
  return [...queue, ...(incoming || []).filter(item => {
    const key = imageLocatorKey(item)
    if (seen.has(key) || item.is_favorite || item._isFolder) return false
    seen.add(key)
    return true
  })]
}
