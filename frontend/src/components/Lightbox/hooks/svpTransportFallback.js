export function shouldFallbackToHlsFromMse(error) {
  const detail = error?.response?.data?.detail || error?.response?.data?.error || ''
  return error?.response?.status === 503
    && detail === 'No trusted SVP Manager graph is available'
}

export function shouldUseMseSvp(mseSupported, sourceWidth) {
  return Boolean(mseSupported)
    && (!Number.isFinite(sourceWidth) || sourceWidth <= 4096)
}

export function getSvpStreamQuality(requestedQuality, sourceWidth) {
  if (Number.isFinite(sourceWidth) && sourceWidth > 4096
      && (!requestedQuality || requestedQuality === 'original')) {
    return '2160p'
  }
  return requestedQuality
}
