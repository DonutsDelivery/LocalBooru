const VIDEO_EXTENSIONS = new Set(['webm', 'mp4', 'mov', 'avi', 'mkv'])

function hasVideoExtension(filename) {
  if (!filename) return false
  const extension = filename.toLowerCase().split('.').pop()
  return VIDEO_EXTENSIONS.has(extension)
}

export function isGridVideo(image) {
  return hasVideoExtension(image?.original_filename) || hasVideoExtension(image?.filename)
}

export function gridMediaPath(image, useFullImage = false) {
  if (useFullImage && !isGridVideo(image) && image?.url) {
    return image.url
  }
  return image?.thumbnail_url || ''
}
