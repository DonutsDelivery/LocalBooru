import test from 'node:test'
import assert from 'node:assert/strict'

import { nativeViewportForLightbox, selectPlaybackTarget } from './nativePlaybackPolicy.js'

const base = {
  desktopTauri: true,
  nativeRendererAvailable: true,
  safeMode: false,
  desktopPlayerMode: 'native',
  isVideo: true,
  isCasting: false,
  castProtocol: null,
  mobileWebView: false,
  remoteBrowser: false,
  localFileAvailable: true,
}

test('uses native playback only for supported local desktop videos', () => {
  assert.equal(selectPlaybackTarget(base), 'native-local')
  assert.equal(selectPlaybackTarget({ ...base, desktopPlayerMode: 'native_svp' }), 'native-local')
  assert.equal(selectPlaybackTarget({ ...base, desktopPlayerMode: 'react' }), 'web-hls')
  assert.equal(selectPlaybackTarget({ ...base, isVideo: false }), 'react-image')
  assert.equal(selectPlaybackTarget({ ...base, desktopTauri: false }), 'web-hls')
  assert.equal(selectPlaybackTarget({ ...base, nativeRendererAvailable: false }), 'web-hls')
  assert.equal(selectPlaybackTarget({ ...base, safeMode: true }), 'web-hls')
  assert.equal(selectPlaybackTarget({ ...base, localFileAvailable: false }), 'web-hls')
})

test('casting always retains the receiver HLS path', () => {
  assert.equal(selectPlaybackTarget({ ...base, isCasting: true }), 'cast-receiver')
  assert.equal(selectPlaybackTarget({ ...base, isCasting: true, castProtocol: 'dlna' }), 'dlna-receiver')
})

test('mobile and remote browser clients use their existing streamed paths', () => {
  assert.equal(selectPlaybackTarget({ ...base, mobileWebView: true }), 'mobile-webview')
  assert.equal(selectPlaybackTarget({ ...base, remoteBrowser: true, desktopTauri: false }), 'web-hls')
  assert.equal(selectPlaybackTarget({ ...base, directRemote: true }), 'direct-remote')
})

test('native viewport exactly matches the measured React player rectangle', () => {
  assert.deepEqual(nativeViewportForLightbox({
    x: 98,
    y: 78,
    width: 2462,
    height: 1362,
    right: 2560,
    bottom: 1440,
  }, true), {
    x: 98,
    y: 78,
    width: 2462,
    height: 1362,
    visible: true,
  })
})
