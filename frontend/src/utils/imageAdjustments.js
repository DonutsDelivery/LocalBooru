export function adjustmentLocator(image) {
  if (!image?.library_id || image?.directory_id == null || image?.id == null) {
    throw new Error('Image adjustments require library_id, directory_id, and image id')
  }

  return {
    libraryId: String(image.library_id),
    directoryId: Number(image.directory_id),
    imageId: Number(image.id),
  }
}

export function adjustmentQuery(locator, adjustmentHash = null) {
  const params = new URLSearchParams()
  params.set('library_id', locator.libraryId)
  params.set('directory_id', String(locator.directoryId))
  if (adjustmentHash) params.set('adjustment_hash', adjustmentHash)
  return params.toString()
}

export function appendCacheBuster(url, value = Date.now()) {
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}t=${value}`
}

export function imageMatchesLocator(image, locator) {
  return String(image?.library_id) === locator.libraryId
    && Number(image?.directory_id) === locator.directoryId
    && Number(image?.id) === locator.imageId
}

export function createAdjustmentRequestOwner() {
  let generation = 0

  return {
    begin(locator, adjustments) {
      generation += 1
      return { generation, locator, adjustments: { ...adjustments } }
    },
    invalidate() {
      generation += 1
    },
    owns(request) {
      return request.generation === generation
    },
  }
}
