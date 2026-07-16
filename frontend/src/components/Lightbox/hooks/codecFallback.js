export function getCodecFallbackStartPosition(currentTime) {
  return Number.isFinite(currentTime) && currentTime > 0 ? currentTime : 0
}
