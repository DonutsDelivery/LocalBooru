import assert from 'node:assert/strict'
import test from 'node:test'

import { loadCollectionPages, restoreCollectionScroll, savedCollectionPage } from './collectionState.js'

function storage(values = {}) {
  return { getItem: key => values[key] ?? null }
}

test('restores the saved page count and ordered collection items', async () => {
  const key = 'collection_video_7'
  const page = savedCollectionPage(storage({ [`${key}_page`]: '3' }), key)
  const requested = []
  const result = await loadCollectionPages(async nextPage => {
    requested.push(nextPage)
    return { images: [{ id: nextPage }], has_more: nextPage < 4 }
  }, page)

  assert.deepEqual(requested, [1, 2, 3])
  assert.deepEqual(result.images.map(image => image.id), [1, 2, 3])
  assert.equal(result.page, 3)
  assert.equal(result.hasMore, true)
})

test('stops at the last available page and ignores invalid saved pages', async () => {
  assert.equal(savedCollectionPage(storage({ 'collection_page': 'Infinity' }), 'collection'), 1)
  const requested = []
  const result = await loadCollectionPages(async nextPage => {
    requested.push(nextPage)
    return { images: [{ id: nextPage }], has_more: nextPage < 2 }
  }, 5)

  assert.deepEqual(requested, [1, 2])
  assert.equal(result.page, 2)
  assert.equal(result.hasMore, false)
})

test('restores collection scroll only after the container is available', () => {
  const key = 'collection_image_4'
  const saved = storage({ [`${key}_scroll`]: '482' })
  restoreCollectionScroll(saved, key, null)
  const container = { scrollTop: 0 }
  restoreCollectionScroll(saved, key, container)
  assert.equal(container.scrollTop, 482)
})
