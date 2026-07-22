export function classifyHorizontalSwipe({
  deltaX,
  deltaY,
  zoomScale,
  touchCount = 1,
  blocked = false,
  handled = false,
}) {
  if (touchCount !== 1 || blocked || handled || zoomScale > 1) return null
  if (Math.abs(deltaX) <= 50 || Math.abs(deltaX) <= Math.abs(deltaY)) return null
  return deltaX > 0 ? 'right' : 'left'
}

export function isGestureCandidateCurrent(startToken, currentToken) {
  return startToken !== null && startToken === currentToken
}

export function curationActionForSwipe() {
  return null
}
