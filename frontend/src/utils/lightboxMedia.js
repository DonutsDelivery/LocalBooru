export function isVideoMediaElement(media) {
  return Boolean(
    media
      && String(media.tagName).toUpperCase() === 'VIDEO'
      && typeof media.pause === 'function'
      && typeof media.play === 'function'
      && typeof media.removeAttribute === 'function'
      && typeof media.load === 'function'
  )
}

export function releaseVideoMedia(media) {
  if (!isVideoMediaElement(media)) return false
  media.pause()
  media.removeAttribute('src')
  media.load()
  return true
}
