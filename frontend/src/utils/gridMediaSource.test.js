import assert from 'node:assert/strict'
import test from 'node:test'

import { gridMediaPath, isGridVideo } from './gridMediaSource.js'

const still = {
  filename: 'photo.png',
  original_filename: 'photo.png',
  thumbnail_url: '/thumb/photo.webp',
  url: '/file/photo.png',
}

const video = {
  filename: 'clip.mp4',
  original_filename: 'clip.mp4',
  thumbnail_url: '/thumb/clip.webp',
  url: '/file/clip.mp4',
}

test('largest masonry tiles use full still images', () => {
  assert.equal(gridMediaPath(still, false), still.thumbnail_url)
  assert.equal(gridMediaPath(still, true), still.url)
})

test('videos keep thumbnails even at the largest tile size', () => {
  assert.equal(isGridVideo(video), true)
  assert.equal(gridMediaPath(video, true), video.thumbnail_url)
})

test('full-image mode falls back to the thumbnail when no file URL exists', () => {
  assert.equal(gridMediaPath({ ...still, url: null }, true), still.thumbnail_url)
})

test('video detection accepts either canonical filename field', () => {
  assert.equal(isGridVideo({ filename: 'fallback.MKV' }), true)
  assert.equal(isGridVideo({ original_filename: 'still.jpeg', filename: 'still.jpeg' }), false)
})
