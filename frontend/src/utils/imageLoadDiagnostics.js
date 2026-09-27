const MEDIA_ERRORS = new Map([
  ['Media target is ambiguous because multiple existing file paths share this image', 'Several copies share this image ID; the server cannot choose a source file.'],
  ['No file matches the current image version', 'The source files no longer match the imported image.'],
  ['Media version is no longer current', 'This image changed since the gallery loaded. Refresh the library.'],
  ['File not found on disk', 'The original image file is missing from disk.'],
  ['File not found', 'The original image file is missing from disk.'],
  ['Image file not found', 'No available source file is recorded for this image.'],
  ['Drive is offline', 'The drive containing the original image is offline.'],
  ['Directory not found', 'The image directory is no longer available.'],
  ['Image not found', 'The image is no longer in this directory.'],
  ['directory_id is required', 'The media request is missing its directory identity.'],
  ['library_id is required', 'The media request is missing its library identity.'],
])

function httpReason(status, detail) {
  if (MEDIA_ERRORS.has(detail)) return MEDIA_ERRORS.get(detail)
  if (status === 401 || status === 403) return 'The media request was denied. Reconnect to the server and retry.'
  if (status === 404) return 'The media server could not find the original image.'
  if (status === 429) return 'The media server is busy. Retry shortly.'
  if (status >= 500) return 'The media server failed to serve this image. Check its log for the matching image ID.'
  return 'The media server rejected this image request. Check its log for the matching image ID.'
}

// <img> error events have no HTTP status. Probe only after failure, and never
// consume a successful full-resolution response just to diagnose its display.
export async function diagnoseImageLoad(source, fetchMedia = fetch) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 8000)
  try {
    const response = await fetchMedia(source, { cache: 'no-store', signal: controller.signal })
    if (response.ok) {
      await response.body?.cancel().catch(() => {})
      const contentType = response.headers.get('content-type') || ''
      const mediaType = contentType.split(';')[0].trim().toLowerCase()
      const contentLength = response.headers.get('content-length') || ''
      return {
        status: response.status,
        mediaType: /^[a-z0-9.+-]+\/[a-z0-9.+-]+$/.test(mediaType) ? mediaType : null,
        contentLength: /^\d{1,20}$/.test(contentLength) ? contentLength : null,
        reason: mediaType && !mediaType.startsWith('image/') && mediaType !== 'application/octet-stream'
          ? 'A repeat request returned non-image content. Check the media route and server log.'
          : 'A repeat request returned the image. The original failure may have been transient, or the viewer may not be able to decode or render it.',
      }
    }
    let detail
    if (response.headers.get('content-type')?.includes('application/json')) {
      try {
        detail = (await response.json())?.detail
      } catch {
        // A broken error body must not obscure the HTTP status.
      }
    }
    return { status: response.status, reason: httpReason(response.status, detail) }
  } catch {
    return {
      status: null,
      reason: controller.signal.aborted
        ? 'The diagnostic request timed out. Check the media server and retry.'
        : 'The diagnostic request failed (connection or browser policy). Check the media server log.',
    }
  } finally {
    clearTimeout(timeout)
  }
}
