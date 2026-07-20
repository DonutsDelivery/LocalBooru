import test from 'node:test'
import assert from 'node:assert/strict'

import { getCodecFallbackStartPosition } from './codecFallback.js'

test('preserves the current playback position when codec fallback begins', () => {
  assert.equal(getCodecFallbackStartPosition(2.75), 2.75)
  assert.equal(getCodecFallbackStartPosition(Number.NaN), 0)
})
