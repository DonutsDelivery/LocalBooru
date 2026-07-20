import test from 'node:test'
import assert from 'node:assert/strict'
import { WINDOW_RESIZE_HANDLES, startWindowResize } from './windowResize.js'

function resizeEvent(button = 0) {
  const calls = []
  return {
    button,
    calls,
    preventDefault() { calls.push('preventDefault') },
    stopPropagation() { calls.push('stopPropagation') },
  }
}

// AC: @frameless-window-resizing ac-1
test('every frameless edge and corner dispatches its matching resize direction', () => {
  const dispatched = []
  for (const { direction } of WINDOW_RESIZE_HANDLES) {
    const event = resizeEvent()
    assert.equal(startWindowResize({
      event,
      direction,
      isDesktop: true,
      isMaximized: false,
      startResizeDragging: value => dispatched.push(value),
    }), true)
    assert.deepEqual(event.calls, ['preventDefault', 'stopPropagation'])
  }

  assert.deepEqual(
    dispatched.sort(),
    ['East', 'North', 'NorthEast', 'NorthWest', 'South', 'SouthEast', 'SouthWest', 'West']
  )
})

// AC: @frameless-window-resizing ac-2
// AC: @frameless-window-resizing ac-3
test('resize dispatch ignores non-primary, non-desktop, maximized, and unresolved states', () => {
  const dispatched = []
  const scenarios = [
    { button: 1, isDesktop: true, isMaximized: false },
    { button: 0, isDesktop: false, isMaximized: false },
    { button: 0, isDesktop: true, isMaximized: true },
    { button: 0, isDesktop: true, isMaximized: null },
  ]

  for (const scenario of scenarios) {
    const event = resizeEvent(scenario.button)
    assert.equal(startWindowResize({
      event,
      direction: 'East',
      isDesktop: scenario.isDesktop,
      isMaximized: scenario.isMaximized,
      startResizeDragging: value => dispatched.push(value),
    }), false)
    assert.deepEqual(event.calls, [])
  }
  assert.deepEqual(dispatched, [])
})
