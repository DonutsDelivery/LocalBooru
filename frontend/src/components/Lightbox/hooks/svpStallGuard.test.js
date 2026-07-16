import test from 'node:test'
import assert from 'node:assert/strict'

import { shouldRestartStalledSVP } from './svpStallGuard.js'

test('does not restart an SVP stream before its first fragment is buffered', () => {
  assert.equal(shouldRestartStalledSVP({ hasBufferedFragment: false, readyState: 0, isBuffered: false }), false)
})

test('does not force-restart an established SVP stream on browser buffering events', () => {
  assert.equal(shouldRestartStalledSVP({ hasBufferedFragment: true, readyState: 2, isBuffered: false }), false)
  assert.equal(shouldRestartStalledSVP({ hasBufferedFragment: true, readyState: 3, isBuffered: true }), false)
})
