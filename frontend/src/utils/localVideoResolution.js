// Backend selection is authoritative: a loopback remote proxy is still remote.
export function localRawResizeAvailable(linuxDesktop, embeddedBackend) {
  return Boolean(linuxDesktop && embeddedBackend)
}

export function localResolutionBounds(quality) {
  const bounds = {
    original: null,
    '1440p': [2560, 1440],
    '1080p': [1920, 1080],
    '1080p_enhanced': [1920, 1080],
    '720p': [1280, 720],
    '480p': [854, 480],
  }
  if (!Object.hasOwn(bounds, quality)) throw new Error(`Unknown video resolution: ${quality}`)
  return bounds[quality] ? { maxWidth: bounds[quality][0], maxHeight: bounds[quality][1] } : null
}

export function videoQualityOptions(rawResize = false) {
  const options = [
    { id: 'original', label: 'Original', description: 'No transcoding', maxHeight: Infinity },
    { id: '1440p', label: '1440p (QHD)', description: '30 Mbps', maxHeight: 1440 },
    { id: '1080p_enhanced', label: '1080p Enhanced', description: '20 Mbps', maxHeight: 1080 },
    { id: '1080p', label: '1080p', description: '12 Mbps', maxHeight: 1080 },
    { id: '720p', label: '720p', description: '8 Mbps', maxHeight: 720 },
    { id: '480p', label: '480p', description: '4 Mbps', maxHeight: 480 },
  ]
  return rawResize
    ? options.filter(option => option.id !== '1080p_enhanced').map(option => ({
      ...option, description: option.id === 'original' ? 'Source resolution' : 'Resize decoded frames',
    }))
    : options
}

export async function prepareLocalResolution(send, owner, quality, isCurrent) {
  if (!isCurrent()) return false
  const accepted = await send({ ...owner, bounds: localResolutionBounds(quality) })
  return Boolean(accepted && isCurrent())
}
