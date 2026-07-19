export const WINDOW_RESIZE_HANDLES = [
  { direction: 'North', edge: 'north' },
  { direction: 'South', edge: 'south' },
  { direction: 'East', edge: 'east' },
  { direction: 'West', edge: 'west' },
  { direction: 'NorthEast', edge: 'north-east' },
  { direction: 'NorthWest', edge: 'north-west' },
  { direction: 'SouthEast', edge: 'south-east' },
  { direction: 'SouthWest', edge: 'south-west' },
]

export function startWindowResize({ event, direction, isDesktop, isMaximized, startResizeDragging }) {
  if (event.button !== 0 || !isDesktop || isMaximized !== false) return false
  event.preventDefault()
  event.stopPropagation()
  startResizeDragging(direction)
  return true
}
