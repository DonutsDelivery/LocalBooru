import test from 'node:test'
import assert from 'node:assert/strict'

import {
  createViewRequestOwner,
  isUnexpectedEmptyPage,
  mergeFirstPage,
  nextLoadRetryDelay,
  refreshGroupedFolderCatalog,
  shouldRefreshForLibraryEvent,
} from './galleryState.js'

// AC: @identity-safe-image-adjustments ac-3
test('live page merge preserves loaded records while inserting and updating page-one records', () => {
  const existing = [
    { id: 2, directory_id: 1, library_id: 'primary', filename: 'two' },
    { id: 1, directory_id: 1, library_id: 'primary', filename: 'one' },
  ]
  const incoming = [
    { id: 3, directory_id: 1, library_id: 'primary', filename: 'three' },
    { id: 2, directory_id: 1, library_id: 'primary', filename: 'duplicate' },
    { id: 2, directory_id: 2, library_id: 'primary', filename: 'same-id-other-directory' },
  ]

  assert.deepEqual(mergeFirstPage(existing, incoming), [
    ...incoming,
    existing[1]
  ])
})

test('pagination retry delay backs off after failures and resets after success', () => {
  assert.equal(nextLoadRetryDelay(250, false), 500)
  assert.equal(nextLoadRetryDelay(20_000, false), 30_000)
  assert.equal(nextLoadRetryDelay(8_000, true), 250)
})

test('an empty middle page is retried instead of advancing the pagination cursor', () => {
  assert.equal(isUnexpectedEmptyPage({ append: true, pageLength: 0, total: 120, loaded: 50 }), true)
  assert.equal(isUnexpectedEmptyPage({ append: true, pageLength: 0, total: 50, loaded: 50 }), false)
  assert.equal(isUnexpectedEmptyPage({ append: false, pageLength: 0, total: 120, loaded: 50 }), false)
})

// AC: @folder-thumbnail-route-identity ac-rescan-refresh
test('background folder results cannot overwrite a newer gallery view', async () => {
  const owner = createViewRequestOwner()
  owner.activate('group=folders&library=old')
  const oldRequest = owner.begin('group=folders&library=old')
  let visibleGallery = 'new gallery'

  const oldResult = Promise.resolve('old folders').then(result => {
    if (owner.owns(oldRequest)) visibleGallery = result
  })

  owner.activate('library=new')
  await oldResult

  assert.equal(visibleGallery, 'new gallery')
  const newRequest = owner.begin('library=new')
  assert.equal(owner.owns(oldRequest), false)
  assert.equal(owner.owns(newRequest), true)
})

// AC: @folder-thumbnail-route-identity ac-rescan-refresh
test('grouped root refresh replaces stale folder previews after image and scan completion', async () => {
  assert.equal(shouldRefreshForLibraryEvent({ type: 'image_added' }), true)
  assert.equal(shouldRefreshForLibraryEvent({
    type: 'task_completed',
    data: { task_type: 'scan_directory' },
  }), true)
  assert.equal(shouldRefreshForLibraryEvent({
    type: 'task_completed',
    data: { task_type: 'tag_image' },
  }), false)

  let visibleFolders = [{ path: '/set', thumbnail_url: '/thumbnail?file_hash=stale' }]
  const refreshed = await refreshGroupedFolderCatalog({
    groupByFolders: true,
    currentFolder: null,
    loadFolders: async () => {
      visibleFolders = [{ path: '/set', thumbnail_url: '/thumbnail?file_hash=current' }]
    },
  })

  assert.equal(refreshed, true)
  assert.equal(visibleFolders[0].thumbnail_url, '/thumbnail?file_hash=current')
})
