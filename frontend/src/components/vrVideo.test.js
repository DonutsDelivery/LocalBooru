import test from 'node:test'
import assert from 'node:assert/strict'
import {
  DEFAULT_VR_CAMERA,
  DEFAULT_VR_CONFIG,
  VR_SETTINGS_STORAGE_KEY,
  detectVRInputProjection,
  detectVRProjection,
  detectVRStereo,
  fitVRTextureSize,
  loadVRConfig,
  normalizeYaw,
  saveVRConfig,
  shouldStageVRTexture,
  uploadVRVideoFrame,
  updateVRCamera,
  updateVRFov,
  vrPointerDelta,
} from './Lightbox/utils/vrVideo.js'

test('uses 180 SBS fisheye defaults and persists the last VR settings', () => {
  const values = new Map()
  const storage = {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  }

  assert.deepEqual(loadVRConfig(storage), DEFAULT_VR_CONFIG)

  const selected = {
    inputProjection: 'equirectangular',
    projection: '360',
    stereo: 'tb',
    eye: 'right',
    fov: 75,
  }
  saveVRConfig(selected, storage)
  assert.equal(values.has(VR_SETTINGS_STORAGE_KEY), true)
  assert.deepEqual(loadVRConfig(storage), selected)
})

test('sanitizes invalid persisted VR settings', () => {
  const storage = {
    getItem: () => JSON.stringify({
      inputProjection: 'cube', projection: '720', stereo: 'red-cyan', eye: 'both', fov: 999,
    }),
  }
  assert.deepEqual(loadVRConfig(storage), {
    ...DEFAULT_VR_CONFIG,
    fov: 120,
  })
})

test('detects explicit 180 and 360 VR filename markers', () => {
  assert.equal(detectVRProjection('concert_VR180_SBS.mp4'), '180')
  assert.equal(detectVRProjection('clip-hequirect.mov'), '180')
  assert.equal(detectVRProjection('travel 360 degree.webm'), '360')
  assert.equal(detectVRProjection('demo_equirect.mp4'), '360')
})

test('does not infer VR from ordinary video dimensions', () => {
  assert.equal(detectVRProjection('camera_3840x1920.mp4'), null)
  assert.equal(detectVRProjection('holiday_1080p.mp4'), null)
  assert.equal(detectVRProjection(null), null)
})

test('detects common stereo layouts while leaving ordinary files mono', () => {
  assert.equal(detectVRStereo('concert_VR180_SBS.mp4'), 'sbs')
  assert.equal(detectVRStereo('concert_VR180_3dh.mp4'), 'sbs')
  assert.equal(detectVRStereo('concert-vr180-top-bottom.mp4'), 'tb')
  assert.equal(detectVRStereo('concert-vr180-3dv.mp4'), 'tb')
  assert.equal(detectVRStereo('concert-vr360.mp4'), 'mono')
})

test('detects explicit fisheye input while defaulting rectangular panoramas to equirectangular', () => {
  assert.equal(detectVRInputProjection('concert_VR180_fisheye_SBS.mp4'), 'fisheye')
  assert.equal(detectVRInputProjection('concert_VR180_dual-fisheye_SBS.mp4'), 'fisheye')
  assert.equal(detectVRInputProjection('concert_VR180_fish-eye.mp4'), 'fisheye')
  assert.equal(detectVRInputProjection('concert_VR180_SBS.mp4'), 'equirectangular')
  assert.equal(detectVRInputProjection(null), 'equirectangular')
})

test('fits oversized VR frames inside the runtime texture limit', () => {
  assert.deepEqual(fitVRTextureSize(8192, 4096, 4096), {
    width: 4096,
    height: 2048,
    scaled: true,
  })
  assert.deepEqual(fitVRTextureSize(3840, 2160, 4096), {
    width: 3840,
    height: 2160,
    scaled: false,
  })
})

test('stages a 2D canvas only when the decoded frame exceeds MAX_TEXTURE_SIZE', () => {
  assert.equal(shouldStageVRTexture('MacIntel', false), false)
  assert.equal(shouldStageVRTexture('Linux x86_64', false), false)
  assert.equal(shouldStageVRTexture('Linux x86_64', true), true)
})

test('360 wins when a filename describes a 360 by 180 panorama', () => {
  assert.equal(detectVRProjection('panorama_360x180.mp4'), '360')
})

test('camera dragging wraps 360 yaw and clamps 180 yaw and pitch', () => {
  assert.deepEqual(updateVRCamera(DEFAULT_VR_CAMERA, -1200, -1200, '360'), {
    yaw: -144,
    pitch: -89,
    roll: 0,
  })
  assert.deepEqual(updateVRCamera(DEFAULT_VR_CAMERA, -1200, 1200, '180'), {
    yaw: 90,
    pitch: 89,
    roll: 0,
  })
  assert.equal(normalizeYaw(540), 180)
  assert.equal(normalizeYaw(-540), -180)
})

test('field of view remains inside the interactive viewer limits', () => {
  assert.equal(updateVRFov(100, -500), 30)
  assert.equal(updateVRFov(100, 500), 120)
  assert.equal(updateVRFov(100, -5), 95)
})

test('mouse dragging uses bounded cursor position deltas', () => {
  assert.deepEqual(vrPointerDelta({ x: 240, y: 160 }, 228, 173), {
    deltaX: -12,
    deltaY: 13,
  })
})

test('uploads video with the full texImage2D overload required by WebKitGTK', () => {
  const calls = []
  const gl = {
    TEXTURE_2D: 1,
    RGBA: 2,
    UNSIGNED_BYTE: 3,
    texImage2D: (...args) => calls.push(args),
  }
  const video = { videoWidth: 3840, videoHeight: 1920 }

  uploadVRVideoFrame(gl, video)

  assert.deepEqual(calls, [[1, 0, 2, 2, 3, video]])
})
