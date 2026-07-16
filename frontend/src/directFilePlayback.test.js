import test from 'node:test'
import assert from 'node:assert/strict'
import { createDirectFileItem } from './directFilePlayback.js'

test('maps a validated desktop file into a local-only Lightbox item', () => {
  const item = createDirectFileItem({
    id: -123,
    filename: 'movie.mp4',
    original_filename: 'movie.mp4',
    file_path: '/videos/movie.mp4',
    url: '/api/direct-files/test-token',
    direct_file_token: 'test-token',
    direct_file: true,
    muted: true,
  })
  assert.equal(item.id, -123)
  assert.equal(item.file_path, '/videos/movie.mp4')
  assert.equal(item.direct_file, true)
  assert.equal(item.muted, true)
  assert.equal(item.is_local_direct_file, true)
  assert.equal(item.directory_id, null)
})
