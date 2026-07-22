function queryValue(image, name) {
  for (const value of [image?.url, image?.thumbnail_url]) {
    if (!value) continue
    try {
      const parsed = new URL(value, 'http://localbooru.invalid')
      const found = parsed.searchParams.get(name)
      if (found != null) return found
    } catch {
      // Ignore malformed compatibility URLs and report the missing locator below.
    }
  }
  return null
}

export function imageFileHash(image) {
  return image?.file_hash ?? queryValue(image, 'file_hash') ?? null
}

export function adjustmentLocator(image) {
  const libraryId = image?.library_id ?? queryValue(image, 'library_id') ?? 'primary'
  const directoryId = image?.directory_id ?? queryValue(image, 'directory_id')
  if (directoryId == null || image?.id == null) {
    throw new Error('Image adjustments require library_id, directory_id, and image id')
  }

  return {
    libraryId: String(libraryId),
    directoryId: Number(directoryId),
    imageId: Number(image.id),
  }
}

export function adjustmentQuery(locator, preview = null) {
  const params = new URLSearchParams()
  params.set('library_id', locator.libraryId)
  params.set('directory_id', String(locator.directoryId))
  if (typeof preview === 'string') {
    params.set('adjustment_hash', preview)
  } else if (preview) {
    if (preview.adjustment_hash) params.set('adjustment_hash', preview.adjustment_hash)
    if (preview.preview_key) params.set('preview_key', preview.preview_key)
    if (preview.source_file_hash) params.set('source_file_hash', preview.source_file_hash)
  }
  return params.toString()
}

export function adjustmentControlState({ applying, generatingPreview, adjustments }) {
  const hasAdjustments = adjustments.brightness !== 0
    || adjustments.contrast !== 0
    || adjustments.gamma !== 0

  return {
    inputsDisabled: applying,
    resetDisabled: applying,
    previewDisabled: applying || generatingPreview || !hasAdjustments,
    applyDisabled: applying || !hasAdjustments,
  }
}

export function appendCacheBuster(url, value = Date.now()) {
  if (!url) return url
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}t=${value}`
}

export function imageMatchesLocator(image, locator) {
  try {
    const candidate = adjustmentLocator(image)
    return candidate.libraryId === locator.libraryId
      && candidate.directoryId === locator.directoryId
      && candidate.imageId === locator.imageId
  } catch {
    return false
  }
}

export function imageIdentityKey(imageOrLocator) {
  const locator = 'imageId' in (imageOrLocator || {})
    ? imageOrLocator
    : adjustmentLocator(imageOrLocator)
  return `${encodeURIComponent(locator.libraryId)}:${locator.directoryId}:${locator.imageId}`
}

export function updateImagesByLocator(images, locator, updates) {
  return images.map(image => imageMatchesLocator(image, locator)
    ? { ...image, ...updates }
    : image)
}

export function reorderImagesForSort(images, sort) {
  const direction = ['oldest', 'filename_asc', 'filesize_smallest', 'resolution_low', 'duration_shortest', 'folder_asc'].includes(sort) ? 1 : -1
  const value = image => {
    switch (sort) {
      case 'newest':
      case 'oldest': return Date.parse(image.file_modified_at || image.created_at || 0) || 0
      case 'filename_asc':
      case 'filename_desc': return String(image.original_filename || image.filename || '').toLowerCase()
      case 'filesize_largest':
      case 'filesize_smallest': return Number(image.file_size || 0)
      case 'resolution_high':
      case 'resolution_low': return Number(image.width || 0) * Number(image.height || 0)
      case 'duration_longest':
      case 'duration_shortest': return Number(image.duration || 0)
      case 'folder_asc':
      case 'folder_desc': return String(image.import_source || '').toLowerCase()
      default: return null
    }
  }
  if (value(images[0] || {}) == null) return images
  return [...images].sort((left, right) => {
    const a = value(left)
    const b = value(right)
    const compared = typeof a === 'string' ? a.localeCompare(b) : a - b
    return compared * direction
  })
}

export function createImageSourceOwner() {
  let activeSource = null

  return {
    activate(source) {
      activeSource = { source }
      return activeSource
    },
    owns(source) {
      return source === activeSource
    },
  }
}

export function commitAdjustmentSourceTransition({
  operationOwner,
  sourceOwner,
  committedSource,
  clearPreview,
  publishCommittedSource,
  cleanupPreview,
}) {
  operationOwner.invalidatePreview()
  sourceOwner.activate(committedSource)
  clearPreview()
  publishCommittedSource()
  if (cleanupPreview) {
    Promise.resolve().then(cleanupPreview).catch(() => {})
  }
}

export function createAdjustmentOperationOwner() {
  let previewGeneration = 0
  let applyOperation = null

  return {
    beginPreview(locator, adjustments) {
      previewGeneration += 1
      return { generation: previewGeneration, locator, adjustments: { ...adjustments } }
    },
    invalidatePreview() {
      previewGeneration += 1
    },
    ownsPreview(request) {
      return request.generation === previewGeneration
    },
    beginApply(locator, adjustments, expectedFileHash) {
      if (applyOperation) return null
      applyOperation = { locator, adjustments: { ...adjustments }, expectedFileHash }
      return applyOperation
    },
    finishApply(operation) {
      if (applyOperation === operation) applyOperation = null
    },
    isApplyInFlight() {
      return applyOperation !== null
    },
  }
}
