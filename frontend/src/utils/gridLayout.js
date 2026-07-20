// Shared grid layout utilities used by App.jsx and MasonryGrid.jsx

// Base column counts for tileSize = 3 (medium, the default)
// Extended for high-res screens (3840, 3200)
export const baseColumnCounts = {
  3840: 10, 3200: 9, 2400: 8, 1800: 7, 1400: 6, 1200: 5, 900: 4, 600: 3, 0: 2
}

// Column adjustments for each tile size level
export const tileSizeAdjustments = {
  1: 3,   // +3 columns (smallest tiles)
  2: 1,   // +1 columns
  3: 0,   // base (medium)
  4: -2,  // -2 columns
  5: -4   // -4 columns (largest tiles)
}

// Tile widths per size level
export const tileWidths = { 1: 200, 2: 250, 3: 300, 4: 450, 5: 600 }

// Calculate column count based on window width and tile size
export function getColumnCount(width, tileSize) {
  const adjustment = tileSizeAdjustments[tileSize] || 0
  const breakpoints = Object.keys(baseColumnCounts).map(Number).sort((a, b) => b - a)
  for (const bp of breakpoints) {
    if (width >= bp) {
      return Math.max(1, baseColumnCounts[bp] + adjustment)
    }
  }
  return Math.max(1, 2 + adjustment)
}
