import assert from 'node:assert/strict'
import test from 'node:test'

import {
  createTimelinePreviewOwner,
  shouldRetryTimelinePreview,
  timelinePreviewIdentityKey,
  timelinePreviewLocator,
} from './timelinePreviewLifecycle.js'

// AC: @identity-safe-timeline-previews ac-video-only
test('still images never produce a timeline preview locator', () => {
  assert.equal(timelinePreviewLocator({
    id: 15,
    directory_id: 1,
    library_id: 'library-a',
    file_hash: 'hash-a',
    filename: 'image.png',
  }), null)
})

// AC: @identity-safe-timeline-previews ac-exact-preview-identity
test('video preview identity includes library, directory, image, and content hash', () => {
  const locator = timelinePreviewLocator({
    id: 15,
    directory_id: 1,
    library_id: 'library-a',
    file_hash: 'hash-a',
    original_filename: 'clip.mp4',
  })
  assert.deepEqual(locator, {
    imageId: 15,
    directoryId: 1,
    libraryId: 'library-a',
    fileHash: 'hash-a',
  })
  assert.equal(timelinePreviewIdentityKey(locator), 'library-a:1:15:hash-a')
})

// AC: @identity-safe-timeline-previews ac-optional-failure
test('timeline preview retries are bounded and stop for terminal responses', () => {
  assert.equal(shouldRetryTimelinePreview(true, 0), true)
  assert.equal(shouldRetryTimelinePreview(true, 9), true)
  assert.equal(shouldRetryTimelinePreview(true, 10), false)
  assert.equal(shouldRetryTimelinePreview(false, 0), false)
})

// AC: @identity-safe-timeline-previews ac-request-ownership
test('new timeline ownership aborts old work and clears its retry', () => {
  const scheduled = new Map()
  let nextTimer = 0
  const owner = createTimelinePreviewOwner({
    setTimeout(callback) {
      const id = ++nextTimer
      scheduled.set(id, callback)
      return id
    },
    clearTimeout(id) {
      scheduled.delete(id)
    },
  })

  const first = owner.begin('first')
  let retried = false
  owner.schedule(first, () => { retried = true }, 3000)
  const second = owner.begin('second')

  assert.equal(first.signal.aborted, true)
  assert.equal(owner.isCurrent(first), false)
  assert.equal(owner.isCurrent(second), true)
  assert.equal(scheduled.size, 0)
  assert.equal(retried, false)

  owner.cancel()
  assert.equal(second.signal.aborted, true)
})
