import assert from 'node:assert/strict'
import test from 'node:test'

import {
  adjustmentLocator,
  adjustmentQuery,
  appendCacheBuster,
  createAdjustmentRequestOwner,
  imageMatchesLocator,
} from './imageAdjustments.js'

const imageA = { id: 7, directory_id: 2, library_id: 'library-a' }
const imageB = { id: 7, directory_id: 3, library_id: 'library-a' }

// AC: @identity-safe-image-adjustments ac-3
test('adjustment request ownership rejects slider and navigation responses that became stale', () => {
  const owner = createAdjustmentRequestOwner()
  const first = owner.begin(adjustmentLocator(imageA), { brightness: 10, contrast: 0, gamma: 0 })

  owner.invalidate()
  assert.equal(owner.owns(first), false)

  const second = owner.begin(adjustmentLocator(imageA), { brightness: 20, contrast: 0, gamma: 0 })
  const navigated = owner.begin(adjustmentLocator(imageB), { brightness: 20, contrast: 0, gamma: 0 })
  assert.equal(owner.owns(second), false)
  assert.equal(owner.owns(navigated), true)
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
