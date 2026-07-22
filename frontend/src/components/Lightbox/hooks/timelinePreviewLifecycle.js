export function timelinePreviewLocator(image) {
  const filename = image?.original_filename || image?.filename
  const extension = filename?.split('.').pop()?.toLowerCase()
  if (!['mp4', 'webm', 'mov', 'avi', 'mkv'].includes(extension)) return null
  if (!image?.library_id || image?.directory_id == null || image?.id == null || !image?.file_hash) {
    return null
  }
  return {
    libraryId: image.library_id,
    directoryId: image.directory_id,
    imageId: image.id,
    fileHash: image.file_hash,
  }
}

export function timelinePreviewIdentityKey(locator) {
  if (!locator) return ''
  return `${locator.libraryId}:${locator.directoryId}:${locator.imageId}:${locator.fileHash}`
}

export function shouldRetryTimelinePreview(generating, retries, maxRetries = 10) {
  return generating === true && retries < maxRetries
}

export function createTimelinePreviewOwner(timers = globalThis) {
  let generation = 0
  let controller = null
  let timer = null

  const cancel = () => {
    generation += 1
    controller?.abort()
    controller = null
    if (timer != null) timers.clearTimeout(timer)
    timer = null
  }

  const begin = (key) => {
    cancel()
    controller = new AbortController()
    return { generation, key, signal: controller.signal }
  }

  const isCurrent = (request) => (
    request != null
    && request.generation === generation
    && !request.signal.aborted
  )

  const schedule = (request, callback, delay) => {
    if (!isCurrent(request)) return
    if (timer != null) timers.clearTimeout(timer)
    timer = timers.setTimeout(() => {
      timer = null
      if (isCurrent(request)) callback()
    }, delay)
  }

  return { begin, cancel, isCurrent, schedule }
}
