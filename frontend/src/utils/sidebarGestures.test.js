import test from 'node:test'
import assert from 'node:assert/strict'
import { classifySidebarSwipe } from './sidebarGestures.js'

test('opens the sidebar from the phone edge and menu-button region', () => {
  assert.equal(classifySidebarSwipe({ startX: 12, deltaX: 70, deltaY: 4, isOpen: false }), 'open')
  assert.equal(classifySidebarSwipe({ startX: 52, deltaX: 70, deltaY: 4, isOpen: false }), 'open')
})

test('rejects opening swipes outside the activation region or without horizontal intent', () => {
  assert.equal(classifySidebarSwipe({ startX: 100, deltaX: 70, deltaY: 4, isOpen: false }), null)
  assert.equal(classifySidebarSwipe({ startX: 20, deltaX: 55, deltaY: 50, isOpen: false }), null)
  assert.equal(classifySidebarSwipe({ startX: 20, deltaX: 50, deltaY: 0, isOpen: false }), null)
})

test('closes an open sidebar with a leftward horizontal swipe', () => {
  assert.equal(classifySidebarSwipe({ startX: 240, deltaX: -70, deltaY: 4, isOpen: true }), 'close')
  assert.equal(classifySidebarSwipe({ startX: 20, deltaX: 70, deltaY: 4, isOpen: true }), null)
})
