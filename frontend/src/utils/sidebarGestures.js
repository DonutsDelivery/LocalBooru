export function classifySidebarSwipe({ deltaX, deltaY, isOpen }) {
  const horizontalDistance = Math.abs(deltaX)
  // Require a firm horizontal swipe: at least 50px and clearly more horizontal than vertical
  if (horizontalDistance <= 50 || horizontalDistance <= Math.abs(deltaY) * 1.25) {
    return null
  }

  if (isOpen) return deltaX < 0 ? 'close' : null
  return deltaX > 0 ? 'open' : null
}
