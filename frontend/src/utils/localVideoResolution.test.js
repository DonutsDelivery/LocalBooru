import assert from 'node:assert/strict'
import test from 'node:test'
import { localRawResizeAvailable, localResolutionBounds, videoQualityOptions, prepareLocalResolution } from './localVideoResolution.js'

// AC: @local-decoded-video-resolution ac-local-raw
// AC: @local-decoded-video-resolution ac-remote
test('only Linux desktop playback from its embedded backend uses raw resize', () => {
  assert.equal(localRawResizeAvailable(true, true), true)
  for (const flags of [[false, true], [true, false], [false, false]]) {
    assert.equal(localRawResizeAvailable(...flags), false)
  }
  assert.deepEqual(localResolutionBounds('720p'), { maxWidth: 1280, maxHeight: 720 })
  assert.equal(localResolutionBounds('original'), null)
  assert.deepEqual(localResolutionBounds('1080p_enhanced'), localResolutionBounds('1080p'))
  assert.throws(() => localResolutionBounds('bogus'), /Unknown/)
})

// AC: @local-decoded-video-resolution ac-labels
test('local resolution options have no bitrate or duplicate enhanced quality', () => {
  const options = videoQualityOptions(true)
  assert.deepEqual(options.map(x => x.id), ['original', '1440p', '1080p', '720p', '480p'])
  assert.ok(options.every(x => !/Mbps|Enhanced/.test(`${x.label} ${x.description}`)))
  assert.ok(videoQualityOptions(false).some(x => x.id === '1080p_enhanced' && x.description === '20 Mbps'))
})

// AC: @local-decoded-video-resolution ac-ownership
test('source readiness belongs only to the prepared physical video generation', async () => {
  let current = true, release
  const owner = { hostId: 'localbooru-svp-host-synthetic', hostEpoch: 2, hostRevision: 4 }
  const promise = prepareLocalResolution(update => {
    assert.deepEqual(update, { ...owner, bounds: { maxWidth: 854, maxHeight: 480 } })
    return new Promise(resolve => { release = resolve })
  }, owner, '480p', () => current)
  current = false
  release(true)
  assert.equal(await promise, false)
  assert.equal(await prepareLocalResolution(async () => false, owner, 'original', () => true), false)
  assert.equal(await prepareLocalResolution(async () => true, owner, 'original', () => true), true)
})
