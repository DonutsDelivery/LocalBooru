export const CURATION_BATCH_SIZE = 50
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
