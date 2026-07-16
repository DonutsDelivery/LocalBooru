import test from 'node:test'
import assert from 'node:assert/strict'

import { isUnexpectedEmptyPage, mergeFirstPage, nextLoadRetryDelay } from './galleryState.js'

test('live page merge preserves loaded records while inserting and updating page-one records', () => {
  const existing = [{ id: 2, filename: 'two' }, { id: 1, filename: 'one' }]
  const incoming = [{ id: 3, filename: 'three' }, { id: 2, filename: 'duplicate' }]

  assert.deepEqual(mergeFirstPage(existing, incoming), [
    { id: 3, filename: 'three' },
    { id: 2, filename: 'duplicate' },
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
