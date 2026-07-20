export const SIDEBAR_EDGE_SWIPE_WIDTH = 72

export function classifySidebarSwipe({ startX, deltaX, deltaY, isOpen }) {
  const horizontalDistance = Math.abs(deltaX)
  if (horizontalDistance <= 50 || horizontalDistance <= Math.abs(deltaY) * 1.25) {
    return null
  }

  if (isOpen) return deltaX < 0 ? 'close' : null
  return deltaX > 0 && startX <= SIDEBAR_EDGE_SWIPE_WIDTH ? 'open' : null
}
