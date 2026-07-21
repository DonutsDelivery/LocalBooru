import test from 'node:test'
import assert from 'node:assert/strict'

import {
  canonicalWatchItem,
  watchHistoryIdentityKey,
  watchHistoryLocator,
} from './continueWatching.js'

// AC: @identity-safe-image-adjustments ac-canonical-entry
test('continue watching keeps canonical image identity and overlays playback only', () => {
  const history = {
    image_id: 12,
    library_id: 'library-a',
    directory_id: 1,
    playback_position: 8,
    duration: 20,
    progress: 0.4,
    url: '/stale',
    file_hash: 'stale',
  }
  const image = {
    id: 12,
    library_id: 'library-a',
    directory_id: 1,
    file_hash: 'current',
    url: '/current',
    thumbnail_url: '/current-thumb',
    filename: 'video.mp4',
  }

  const hydrated = canonicalWatchItem(history, image)

  assert.equal(hydrated.url, '/current')
  assert.equal(hydrated.file_hash, 'current')
  assert.equal(hydrated.playback_position, 8)
  assert.equal(hydrated.progress, 0.4)
  assert.deepEqual(watchHistoryLocator(hydrated), {
    imageId: 12,
    libraryId: 'library-a',
    directoryId: 1,
  })
  assert.equal(watchHistoryIdentityKey(hydrated), 'library-a:1:12')
})

// AC: @identity-safe-image-adjustments ac-canonical-entry
test('continue watching rejects incomplete or mismatched canonical hydration', () => {
  const history = { image_id: 12, library_id: 'library-a', directory_id: 1 }
  assert.throws(() => canonicalWatchItem(history, {
    id: 12,
    library_id: 'library-b',
    directory_id: 1,
  }), /does not match/)
  assert.throws(() => watchHistoryLocator({ image_id: 12 }), /full image locator/)
})
