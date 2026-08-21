export function getCodecFallbackStartPosition(currentTime) {
  return Number.isFinite(currentTime) && currentTime > 0 ? currentTime : 0
}

export function getCodecFallbackQuality(sourceWidth) {
  return Number.isFinite(sourceWidth) && sourceWidth > 3840 ? '2160p' : null
}

export function getCompatibilityRestartQuality(currentQuality, compatibilityRemuxActive) {
  return compatibilityRemuxActive ? 'apple_remux' : currentQuality
}

export function shouldStartCodecFallback({
  streamActive,
  alreadyStarted,
  video,
  force = false,
}) {
  if (streamActive || alreadyStarted || !video) return false
  if (force) return true
  return video.videoWidth === 0 && video.videoHeight === 0 && video.readyState >= 2
}