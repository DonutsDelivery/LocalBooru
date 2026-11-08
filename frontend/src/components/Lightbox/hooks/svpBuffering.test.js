import test from 'node:test'
import assert from 'node:assert/strict'

import { shouldStartSVPPlayback, svpBufferedAhead } from './svpBuffering.js'

const ranges = (...spans) => ({
  length: spans.length,
  start: i => spans[i][0],
  end: i => spans[i][1],
})

test('starts SVP playback after three 2-second HLS segments despite MSE rounding', () => {
  assert.equal(shouldStartSVPPlayback(4), false)
  assert.equal(shouldStartSVPPlayback(5.4), false)
  assert.equal(shouldStartSVPPlayback(5.98), true)
  assert.equal(shouldStartSVPPlayback(8), true)
})

test('measures buffered seconds ahead of the playhead', () => {
  assert.equal(svpBufferedAhead(ranges([0, 6]), 1), 5)
  assert.equal(svpBufferedAhead(ranges([0, 3], [10, 20]), 12), 8)
  assert.equal(svpBufferedAhead(ranges(), 0), 0)
  assert.equal(svpBufferedAhead(ranges([0, 6]), 6), 0)
})

test('a small audio-led start gap still counts as buffered', () => {
  assert.ok(Math.abs(svpBufferedAhead(ranges([0.021, 6]), 0) - 5.979) < 1e-9)
  assert.equal(shouldStartSVPPlayback(svpBufferedAhead(ranges([0.021, 6]), 0)), true)
})

test('a real hole before the buffered range is not bridged', () => {
  assert.equal(svpBufferedAhead(ranges([2, 8]), 0), 0)
})
