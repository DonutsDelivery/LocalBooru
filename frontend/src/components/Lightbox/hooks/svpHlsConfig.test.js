import test from 'node:test'
import assert from 'node:assert/strict'

import { SVP_HLS_CONFIG } from './svpHlsConfig.js'

test('SVP buffer window stays bounded for a phone WebView', () => {
  assert.ok(SVP_HLS_CONFIG.maxBufferLength <= 30)
  assert.ok(SVP_HLS_CONFIG.maxMaxBufferLength <= 60)
  assert.ok(SVP_HLS_CONFIG.backBufferLength <= 60)
})

test('SVP stream starts at the first segment, not the live edge', () => {
  assert.equal(SVP_HLS_CONFIG.startPosition, 0)
  assert.equal(SVP_HLS_CONFIG.lowLatencyMode, false)
})

test('SVP manifest keeps retrying while the first segment is produced', () => {
  assert.ok(SVP_HLS_CONFIG.manifestLoadingMaxRetry >= 30)
})
