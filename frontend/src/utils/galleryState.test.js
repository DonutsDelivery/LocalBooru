import test from 'node:test'
import assert from 'node:assert/strict'

import { isUnexpectedEmptyPage, mergeFirstPage, nextLoadRetryDelay } from './galleryState.js'

// AC: @identity-safe-image-adjustments ac-3
test('live page merge preserves loaded records while inserting and updating page-one records', () => {
  const existing = [
    { id: 2, directory_id: 1, library_id: 'primary', filename: 'two' },
    { id: 1, directory_id: 1, library_id: 'primary', filename: 'one' },
  ]
  const incoming = [
    { id: 3, directory_id: 1, library_id: 'primary', filename: 'three' },
    { id: 2, directory_id: 1, library_id: 'primary', filename: 'duplicate' },
    { id: 2, directory_id: 2, library_id: 'primary', filename: 'same-id-other-directory' },
  ]

  assert.deepEqual(mergeFirstPage(existing, incoming), [
    ...incoming,
    existing[1]
  ])
})

test('pagination retry delay backs off after failures and resets after success', () => {
  assert.equal(nextLoadRetryDelay(250, false), 500)
  assert.equal(nextLoadRetryDelay(20_000, false), 30_000)
  assert.equal(nextLoadRetryDelay(8_000, true), 250)
})

test('an empty middle page is retried instead of advancing the pagination cursor', () => {
  assert.equal(isUnexpectedEmptyPage({ append: true, pageLength: 0, total: 120, loaded: 50 }), true)
  assert.equal(isUnexpectedEmptyPage({ append: true, pageLength: 0, total: 50, loaded: 50 }), false)
  assert.equal(isUnexpectedEmptyPage({ append: false, pageLength: 0, total: 120, loaded: 50 }), false)
})
