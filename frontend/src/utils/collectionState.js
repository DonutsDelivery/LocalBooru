export function savedCollectionPage(storage, key) {
  const value = Number(storage.getItem(`${key}_page`))
  return Number.isSafeInteger(value) && value > 0 ? value : 1
}

export async function loadCollectionPages(fetchPage, lastSavedPage, isActive = () => true) {
  let images = []
  let collection = null
  let page = 0
  for (let nextPage = 1; nextPage <= lastSavedPage; nextPage++) {
    const result = await fetchPage(nextPage)
    if (!isActive()) return null
    images = [...images, ...(result.images || [])]
    collection = result
    page = nextPage
    if (!result.has_more) break
  }
  return { collection, images, page, hasMore: Boolean(collection?.has_more) }
}

export function restoreCollectionScroll(storage, key, container) {
  if (!container) return
  const value = Number(storage.getItem(`${key}_scroll`))
  container.scrollTop = Number.isFinite(value) ? Math.max(0, value) : 0
}
