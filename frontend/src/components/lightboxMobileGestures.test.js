import test from 'node:test'
import assert from 'node:assert/strict'
import { calculateDragSeek, classifyTapZone } from './Lightbox/hooks/useVideoGestures.js'
import {
  classifyHorizontalSwipe,
  curationActionForSwipe,
  isGestureCandidateCurrent,
} from '../utils/lightboxGestures.js'
import { shouldRevealOnly } from './Lightbox/hooks/useUIVisibility.js'

// AC: @curation-gesture-decisions-recovery ac-1
// AC: @curation-gesture-decisions-recovery ac-6
test('horizontal lightbox swipes classify both curation decision directions', () => {
  assert.equal(classifyHorizontalSwipe({ deltaX: 51, deltaY: 10, zoomScale: 1 }), 'right')
  assert.equal(classifyHorizontalSwipe({ deltaX: -51, deltaY: 10, zoomScale: 1 }), 'left')
  assert.equal(curationActionForSwipe('right'), 'keep')
  assert.equal(curationActionForSwipe('left'), 'discard')
  assert.equal(curationActionForSwipe(null), null)
})

// AC: @curation-gesture-decisions-recovery ac-2
test('pinch, control, zoomed, vertical, and short gestures do not classify as decisions', () => {
  assert.equal(classifyHorizontalSwipe({ deltaX: 100, deltaY: 0, zoomScale: 1, touchCount: 2 }), null)
  assert.equal(classifyHorizontalSwipe({ deltaX: 100, deltaY: 0, zoomScale: 1, blocked: true }), null)
  assert.equal(classifyHorizontalSwipe({ deltaX: 100, deltaY: 0, zoomScale: 2 }), null)
  assert.equal(classifyHorizontalSwipe({ deltaX: 60, deltaY: 61, zoomScale: 1 }), null)
  assert.equal(classifyHorizontalSwipe({ deltaX: 50, deltaY: 0, zoomScale: 1 }), null)
  assert.equal(classifyHorizontalSwipe({ deltaX: 100, deltaY: 0, zoomScale: 1, handled: true }), null)
})

// AC: @curation-gesture-decisions-recovery ac-3
test('a gesture cannot decide a replacement or revisited curation candidate', () => {
  assert.equal(isGestureCandidateCurrent('directory-a:1:7', 'directory-a:1:7'), true)
  assert.equal(isGestureCandidateCurrent('directory-a:1:7', 'directory-a:2:8'), false)
  assert.equal(isGestureCandidateCurrent('directory-a:1:7', 'directory-a:1:9'), false)
  assert.equal(isGestureCandidateCurrent(null, 'directory-a:1:7'), false)
})

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
