import test from 'node:test'
import assert from 'node:assert/strict'

import { isDoubleClickZone } from './doubleClickZone.js'

const image = { width: 1200, isVideo: false, zoomed: false, curation: false }

test('only accepts the middle of an image, where a click does not navigate', () => {
  assert.deepEqual([0.05, 0.2, 0.39, 0.61, 0.8, 0.95].map(ratio => isDoubleClickZone({ ...image, ratio })), Array(6).fill(false))
  assert.deepEqual([0.4, 0.5, 0.6].map(ratio => isDoubleClickZone({ ...image, ratio })), [true, true, true])
})

test('only accepts the playback-toggle third of a video', () => {
  const video = { ...image, isVideo: true }
  assert.deepEqual([0.1, 0.32, 0.68, 0.9].map(ratio => isDoubleClickZone({ ...video, ratio })), [false, false, false, false])
  assert.deepEqual([0.34, 0.5, 0.66].map(ratio => isDoubleClickZone({ ...video, ratio })), [true, true, true])
})

test('lets a zoomed desktop image reset anywhere but keeps zones on narrow screens', () => {
  assert.equal(isDoubleClickZone({ ...image, ratio: 0.1, zoomed: true }), true)
  assert.equal(isDoubleClickZone({ ...image, ratio: 0.1, zoomed: true, width: 600 }), false)
})

test('never fires in curation, where every point keeps or discards', () => {
  assert.equal(isDoubleClickZone({ ...image, ratio: 0.5, curation: true }), false)
})
