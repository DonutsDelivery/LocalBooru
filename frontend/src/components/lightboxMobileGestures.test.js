import test from 'node:test'
import assert from 'node:assert/strict'
import { calculateDragSeek, classifyTapZone } from './Lightbox/hooks/useVideoGestures.js'
import { shouldRevealOnly } from './Lightbox/hooks/useUIVisibility.js'

// AC: @android-phone-lightbox ac-2
test('a hidden interface consumes the first tap to reveal controls', () => {
  assert.equal(shouldRevealOnly(false), true)
  assert.equal(shouldRevealOnly(true), false)
})

// AC: @android-phone-lightbox ac-2
test('visible tap zones classify backward, toggle, and forward actions', () => {
  assert.equal(classifyTapZone(0.1), 'backward')
  assert.equal(classifyTapZone(0.5), 'toggle')
  assert.equal(classifyTapZone(0.9), 'forward')
})

// AC: @android-phone-lightbox ac-3
test('horizontal drag seeking uses thresholds and ten-second steps', () => {
  assert.equal(calculateDragSeek(20, 0), null)
  assert.equal(calculateDragSeek(21, 0), 10)
  assert.equal(calculateDragSeek(71, 0), 20)
  assert.equal(calculateDragSeek(-71, 0), -20)
})

// AC: @android-phone-lightbox ac-3
test('vertical movement does not claim a seek gesture', () => {
  assert.equal(calculateDragSeek(60, 50), null)
  assert.equal(calculateDragSeek(10, 80), null)
})
