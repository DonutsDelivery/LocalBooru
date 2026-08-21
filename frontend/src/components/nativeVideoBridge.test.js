import test from 'node:test'
import assert from 'node:assert/strict'
import {
  createNativeVideoAdapter,
  nativeReductionHeight,
  openNativeVideo,
  setNativeVideoReduction,
  setNativeVideoViewport,
} from './Lightbox/utils/nativeVideoBridge.js'
import { getQualityOptions } from './Lightbox/utils/videoQualityOptions.js'

function installWindow(bridge = {}) {
  const target = new EventTarget()
  globalThis.window = Object.assign(target, {
    AndroidNativeVideo: bridge,
    devicePixelRatio: 1,
  })
}

test('maps quality labels to playback-time output heights', () => {
  assert.equal(nativeReductionHeight('original'), 0)
  assert.equal(nativeReductionHeight('1080p_enhanced'), 1080)
  assert.equal(nativeReductionHeight('1080p'), 1080)
  assert.equal(nativeReductionHeight('720p'), 720)
})

test('opens the original loopback URL and changes reduction without transcoding', () => {
  const calls = []
  installWindow({
    open: (...args) => calls.push(['open', ...args]),
    setReduction: (...args) => calls.push(['setReduction', ...args]),
  })

  openNativeVideo(4, 'http://127.0.0.1:8790/api/images/7/file', {
    startPosition: 12.5,
    autoplay: true,
    quality: '1080p',
    viewport: {
      getBoundingClientRect: () => ({ left: 10, top: 20, width: 640, height: 360 }),
    },
  })
  setNativeVideoReduction(4, '720p')

  assert.deepEqual(calls, [
    ['open', 4, 'http://127.0.0.1:8790/api/images/7/file', 12.5, true, 1080, 10, 20, 640, 360, 1],
    ['setReduction', 4, 720],
  ])
})

test('requires a real viewport before native playback can open', () => {
  const calls = []
  installWindow({
    open: (...args) => calls.push(['open', ...args]),
    setViewport: (...args) => calls.push(['setViewport', ...args]),
  })

  assert.equal(openNativeVideo(5, 'http://127.0.0.1:8790/media', {}), false)
  assert.equal(openNativeVideo(5, 'http://127.0.0.1:8790/media', {
    viewport: { getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 360 }) },
  }), false)
  assert.equal(calls.length, 0)

  const viewport = {
    getBoundingClientRect: () => ({ left: 4, top: 8, width: 320, height: 180 }),
  }
  assert.equal(setNativeVideoViewport(5, viewport), true)
  assert.deepEqual(calls, [['setViewport', 5, 4, 8, 320, 180, 1]])
})

test('Android quality choices are direct playback effects, not renditions', () => {
  assert.deepEqual(
    getQualityOptions(true).map(({ id, description }) => ({ id, description })),
    [
      { id: 'original', description: 'Direct stream, native output' },
      { id: '1080p', description: 'Direct stream, playback downscale' },
      { id: '720p', description: 'Direct stream, playback downscale' },
    ],
  )
})

test('adapts generation-tagged native events to the existing media controls', async () => {
  const calls = []
  installWindow({
    play: (...args) => calls.push(['play', ...args]),
    pause: (...args) => calls.push(['pause', ...args]),
    seek: (...args) => calls.push(['seek', ...args]),
    setVolume: (...args) => calls.push(['setVolume', ...args]),
    setMuted: (...args) => calls.push(['setMuted', ...args]),
    setSpeed: (...args) => calls.push(['setSpeed', ...args]),
  })

  const adapter = createNativeVideoAdapter(9)
  let timeUpdates = 0
  let firstFrames = 0
  adapter.addEventListener('timeupdate', () => { timeUpdates += 1 })
  adapter.addEventListener('firstframe', () => { firstFrames += 1 })

  window.dispatchEvent(new CustomEvent('localbooru-native-video', {
    detail: {
      generation: 8,
      type: 'position',
      value: { position: 99, duration: 100, buffered: 100 },
    },
  }))
  window.dispatchEvent(new CustomEvent('localbooru-native-video', {
    detail: {
      generation: 9,
      type: 'position',
      value: { position: 3.5, duration: 20, buffered: 12 },
    },
  }))

  assert.equal(adapter.currentTime, 3.5)
  assert.equal(adapter.duration, 20)
  assert.equal(adapter.buffered.end(0), 12)
  assert.equal(timeUpdates, 1)

  const diagnostics = {
    sourceWidth: 3840,
    sourceHeight: 2160,
    requestedEffectHeight: 720,
    effectOutputWidth: 1280,
    effectOutputHeight: 720,
    decoder: 'c2.qti.hevc.decoder',
    decoderHardwareAccelerated: true,
  }
  window.dispatchEvent(new CustomEvent('localbooru-native-video', {
    detail: { generation: 9, type: 'video-size', value: diagnostics },
  }))
  assert.equal(adapter.videoWidth, 3840)
  assert.equal(adapter.videoHeight, 2160)
  assert.deepEqual(adapter.nativeVideoDiagnostics, diagnostics)

  window.dispatchEvent(new CustomEvent('localbooru-native-video', {
    detail: { generation: 9, type: 'first-frame', value: null },
  }))
  assert.equal(firstFrames, 1)

  adapter.currentTime = 7
  adapter.volume = 0.4
  adapter.muted = true
  adapter.playbackRate = 1.5
  await adapter.play()
  adapter.pause()

  assert.deepEqual(calls, [
    ['seek', 9, 7],
    ['setVolume', 9, 0.4],
    ['setMuted', 9, true],
    ['setSpeed', 9, 1.5],
    ['play', 9],
    ['pause', 9],
  ])
  adapter.destroy()
})
