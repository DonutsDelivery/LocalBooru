import test from 'node:test'
import assert from 'node:assert/strict'

import { isVideo } from '../components/Lightbox/utils/helpers.js'

test('lightbox recognizes every video format imported into the library', () => {
  for (const filename of ['clip.webm', 'clip.mp4', 'clip.mov', 'clip.avi', 'clip.mkv']) {
    assert.equal(isVideo(filename), true, filename)
  }
  assert.equal(isVideo('still.png'), false)
})
