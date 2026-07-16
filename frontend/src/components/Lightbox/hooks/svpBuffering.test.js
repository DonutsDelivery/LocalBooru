import test from 'node:test'
import assert from 'node:assert/strict'

import { shouldStartSVPPlayback } from './svpBuffering.js'

test('waits for two HLS segments before starting SVP playback', () => {
  assert.equal(shouldStartSVPPlayback(4), false)
  assert.equal(shouldStartSVPPlayback(8), true)
})
