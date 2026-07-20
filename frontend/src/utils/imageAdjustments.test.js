import assert from 'node:assert/strict'
import test from 'node:test'

import {
  adjustmentLocator,
  adjustmentQuery,
  appendCacheBuster,
  createAdjustmentOperationOwner,
  imageMatchesLocator,
  reorderImagesForSort,
  updateImagesByLocator,
} from './imageAdjustments.js'

const imageA = { id: 7, directory_id: 2, library_id: 'library-a' }
const imageB = { id: 7, directory_id: 3, library_id: 'library-a' }

// AC: @identity-safe-image-adjustments ac-3
test('adjustment request ownership rejects slider and navigation responses that became stale', () => {
  const owner = createAdjustmentOperationOwner()
  const first = owner.beginPreview(adjustmentLocator(imageA), { brightness: 10, contrast: 0, gamma: 0 })

  owner.invalidatePreview()
  assert.equal(owner.ownsPreview(first), false)

  const second = owner.beginPreview(adjustmentLocator(imageA), { brightness: 20, contrast: 0, gamma: 0 })
  const navigated = owner.beginPreview(adjustmentLocator(imageB), { brightness: 20, contrast: 0, gamma: 0 })
  assert.equal(owner.ownsPreview(second), false)
  assert.equal(owner.ownsPreview(navigated), true)
})

// AC: @identity-safe-image-adjustments ac-3
// AC: @identity-safe-image-adjustments ac-4
test('cache busting and image updates preserve and match the full locator', () => {
  assert.equal(
    adjustmentQuery(adjustmentLocator(imageA), 'abc123'),
    'library_id=library-a&directory_id=2&adjustment_hash=abc123'
  )
  assert.equal(
    appendCacheBuster('/api/images/7/file?directory_id=2&library_id=library-a', 123),
    '/api/images/7/file?directory_id=2&library_id=library-a&t=123'
  )
  assert.equal(imageMatchesLocator(imageA, adjustmentLocator(imageA)), true)
  assert.equal(imageMatchesLocator(imageB, adjustmentLocator(imageA)), false)
})

// AC: @identity-safe-image-adjustments ac-3
// AC: @identity-safe-image-adjustments ac-4
test('apply lock survives preview invalidation and exact updates include duplicate-id queues', () => {
  const owner = createAdjustmentOperationOwner()
  const apply = owner.beginApply(adjustmentLocator(imageA), { brightness: 10 }, 'old-hash')
  assert.ok(apply)

  owner.invalidatePreview()
  assert.equal(owner.beginApply(adjustmentLocator(imageB), { brightness: 20 }, 'other-hash'), null)
  assert.equal(owner.isApplyInFlight(), true)
  owner.finishApply(apply)
  assert.equal(owner.isApplyInFlight(), false)

  const updated = updateImagesByLocator([imageA, imageB], adjustmentLocator(imageA), { file_hash: 'new' })
  assert.equal(updated[0].file_hash, 'new')
  assert.equal(updated[1].file_hash, undefined)

  const reordered = reorderImagesForSort(
    [{ ...imageA, file_size: 5 }, { ...imageB, file_size: 10 }],
    'filesize_largest'
  )
  assert.equal(reordered[0].directory_id, 3)
})

// AC: @identity-safe-image-adjustments ac-3
// AC: @identity-safe-image-adjustments ac-4
test('completed apply updates its captured locator after navigation while active UI ownership stays stale', async () => {
  const owner = createAdjustmentOperationOwner()
  const locator = adjustmentLocator(imageA)
  const operation = owner.beginApply(locator, { brightness: 10 }, 'old-hash')
  const ui = owner.beginPreview(locator, { brightness: 10 })
  const completed = Promise.resolve({ file_hash: 'committed-hash' })

  owner.beginPreview(adjustmentLocator(imageB), { brightness: 0 })
  const result = await completed
  const images = updateImagesByLocator([imageA, imageB], operation.locator, result)

  assert.equal(images[0].file_hash, 'committed-hash')
  assert.equal(images[1].file_hash, undefined)
  assert.equal(owner.ownsPreview(ui), false)
  assert.equal(owner.isApplyInFlight(), true)
  owner.finishApply(operation)
})

// AC: @identity-safe-image-adjustments ac-3
test('legacy image payload derives an exact primary locator from media URL parameters', () => {
  const legacy = {
    id: 11,
    url: '/api/images/11/file?directory_id=4',
    thumbnail_url: '/api/images/11/thumbnail?directory_id=4',
  }
  assert.deepEqual(adjustmentLocator(legacy), {
    libraryId: 'primary',
    directoryId: 4,
    imageId: 11,
  })
})
