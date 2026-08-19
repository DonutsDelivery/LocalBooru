import test from 'node:test'
import assert from 'node:assert/strict'
import {
  DEFAULT_VR_CAMERA,
  detectVRProjection,
  detectVRStereo,
  normalizeYaw,
  updateVRCamera,
  updateVRFov,
} from './Lightbox/utils/vrVideo.js'

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
  assert.equal(detectVRStereo('concert-vr180-top-bottom.mp4'), 'tb')
  assert.equal(detectVRStereo('concert-vr360.mp4'), 'mono')
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
