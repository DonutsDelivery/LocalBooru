import test from 'node:test'
import assert from 'node:assert/strict'
import {
  getSvpStreamQuality,
  shouldFallbackToHlsFromMse,
  shouldUseMseSvp,
} from './svpTransportFallback.js'

test('falls back to server HLS when no trusted MSE graph is available', () => {
  assert.equal(shouldFallbackToHlsFromMse({
    response: {
      status: 503,
      data: { detail: 'No trusted SVP Manager graph is available' },
    },
  }), true)
})

test('does not hide unrelated MSE transport or processing failures', () => {
  assert.equal(shouldFallbackToHlsFromMse({
    response: { status: 503, data: { detail: 'SVP sidecar is unavailable' } },
  }), false)
  assert.equal(shouldFallbackToHlsFromMse({
    response: { status: 500, data: { detail: 'No trusted SVP Manager graph is available' } },
  }), false)
  assert.equal(shouldFallbackToHlsFromMse(new Error('MSE codec is not supported')), false)
})

test('routes oversized SVP sources through capped server HLS', () => {
  assert.equal(shouldUseMseSvp(true, 3840), true)
  assert.equal(shouldUseMseSvp(true, 7680), false)
  assert.equal(getSvpStreamQuality('original', 7680), '2160p')
  assert.equal(getSvpStreamQuality(null, 5800), '2160p')
  assert.equal(getSvpStreamQuality('1080p', 7680), '1080p')
  assert.equal(getSvpStreamQuality('original', 3840), 'original')
})
