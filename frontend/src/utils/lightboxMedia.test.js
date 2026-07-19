import test from 'node:test'
import assert from 'node:assert/strict'
import { isVideoMediaElement, releaseVideoMedia } from './lightboxMedia.js'

// AC: @ordinary-lightbox-media-rendering ac-1
// AC: @ordinary-lightbox-media-rendering ac-3
test('video cleanup never invokes video methods on an image element', () => {
  let paused = false
  const image = {
    tagName: 'IMG',
    pause() { paused = true },
    play() {},
    removeAttribute() {},
    load() {},
  }

  assert.equal(isVideoMediaElement(image), false)
  assert.equal(releaseVideoMedia(image), false)
  assert.equal(paused, false)
})

// AC: @ordinary-lightbox-media-rendering ac-1
test('video cleanup releases an actual video element', () => {
  const calls = []
  const video = {
    tagName: 'VIDEO',
    pause() { calls.push('pause') },
    play() {},
    removeAttribute(name) { calls.push(`remove:${name}`) },
    load() { calls.push('load') },
  }

  assert.equal(isVideoMediaElement(video), true)
  assert.equal(releaseVideoMedia(video), true)
  assert.deepEqual(calls, ['pause', 'remove:src', 'load'])
})
