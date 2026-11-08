import { imageIdentityKey } from './imageAdjustments.js'

// Identity of a tile for layout purposes. Must match the identity the grid
// uses for React keys so a tile keeps its column (and its DOM node) when new
// media is prepended above it.
export function tileLayoutKey(image) {
  return image._isFolder ? `folder-${image.path}` : imageIdentityKey(image)
}

function tileHeight(image) {
  const hasDimensions = image.width && image.height && image.width > 0 && image.height > 0
  // Normalized height (column width of 1); 4:3 landscape when unknown.
  return hasDimensions ? image.height / image.width : 3 / 4
}

function shortestColumn(heights) {
  let shortest = 0
  for (let i = 1; i < heights.length; i++) {
    if (heights[i] < heights[shortest]) shortest = i
  }
  return shortest
}

// Distribute tiles into columns by estimated height. When a previous layout
// with the same column count is supplied, tiles that were already placed keep
// their column; only new tiles are placed. New tiles that precede every known
// tile (live imports prepended to a newest-first gallery) are balanced among
// themselves, because the known tiles below them are pinned. Re-running the
// shortest-column pass over the whole list instead would shift almost every
// tile into another column, remounting the grid and moving the user's view.
export function distributeToColumns(images, columnCount, previousLayout = null) {
  const columns = Array.from({ length: columnCount }, () => ({ items: [], height: 0 }))
  const pinned = previousLayout?.columnCount === columnCount ? previousLayout.columns : null
  const leadingHeights = new Array(columnCount).fill(0)
  const placed = new Map()
  let reachedKnownTile = false

  for (const image of images) {
    const key = tileLayoutKey(image)
    const height = tileHeight(image)
    let column = pinned?.get(key)
    if (column === undefined) {
      if (pinned && !reachedKnownTile) {
        column = shortestColumn(leadingHeights)
        leadingHeights[column] += height
      } else {
        column = shortestColumn(columns.map(entry => entry.height))
      }
    } else {
      reachedKnownTile = true
    }
    columns[column].items.push(image)
    columns[column].height += height
    placed.set(key, column)
  }

  return { columns, layout: { columnCount, columns: placed } }
}
