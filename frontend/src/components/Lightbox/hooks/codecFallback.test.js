import test from 'node:test'
import assert from 'node:assert/strict'

import { getCodecFallbackStartPosition, getCodecFallbackQuality, getCompatibilityRestartQuality } from './codecFallback.js'

test('preserves the current playback position when codec fallback begins', () => {
  assert.equal(getCodecFallbackStartPosition(2.75), 2.75)
  assert.equal(getCodecFallbackStartPosition(Number.NaN), 0)
})

test('getCodecFallbackQuality returns 2160p for >3840 width', () => {
  assert.equal(getCodecFallbackQuality(4096), '2160p')
  assert.equal(getCodecFallbackQuality(3840), null)
  assert.equal(getCodecFallbackQuality(1920), null)
})

test('getCompatibilityRestartQuality returns apple_remux when active', () => {
  assert.equal(getCompatibilityRestartQuality('original', true), 'apple_remux')
  assert.equal(getCompatibilityRestartQuality('1440p', false), '1440p')
})