import test from 'node:test'
import assert from 'node:assert/strict'

import {
  completeAuthoritativeRefresh,
  createViewRequestOwner,
  galleryScopePreservesFolder,
  isUnexpectedEmptyPage,
  libraryRefreshMode,
  mergeAuthoritativePages,
  mergeFirstPage,
  nextLoadRetryDelay,
  reconcileAuthoritativeGallery,
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
test('new gallery publications supersede older work in the same complete view', () => {
  const owner = createViewRequestOwner()
  owner.activate('library=old|tile=3')
  const oldLoad = owner.begin('library=old|tile=3')
  const jump = owner.begin('library=old|tile=3')

  assert.equal(owner.owns(oldLoad), false)
  assert.equal(owner.owns(jump), true)

  owner.activate('library=old|tile=4')
  const resizedLoad = owner.begin('library=old|tile=4')

  assert.equal(owner.owns(jump), false)
  assert.equal(owner.owns(resizedLoad), true)
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

// AC: @identity-safe-image-adjustments ac-scan-reconcile
test('scan completion authoritatively removes stale image identities', () => {
  const stale = { id: 12, directory_id: 1, library_id: 'library-a' }
  const current = { id: 13, directory_id: 1, library_id: 'library-a' }

  assert.equal(libraryRefreshMode({ type: 'image_added' }), 'merge')
  assert.equal(libraryRefreshMode({
    type: 'task_completed',
    data: { task_type: 'scan_directory' },
  }), 'replace')
  assert.equal(libraryRefreshMode({
    type: 'task_completed',
    data: { task_type: 'tag_image' },
  }), null)

  assert.deepEqual(reconcileAuthoritativeGallery([current], stale), {
    images: [current],
    currentLocator: null,
  })
  assert.deepEqual(reconcileAuthoritativeGallery([current], current), {
    images: [current],
    currentLocator: current,
  })
})

test('authoritative scan refresh retains loaded pages and does not clear newer scan work', () => {
  const pageOne = { images: [
    { id: 1, directory_id: 1, library_id: 'primary' },
    { id: 2, directory_id: 1, library_id: 'primary' },
  ] }
  const pageTwo = { images: [
    { id: 2, directory_id: 1, library_id: 'primary' },
    { id: 3, directory_id: 1, library_id: 'primary' },
  ] }
  assert.deepEqual(mergeAuthoritativePages([pageOne, pageTwo]), [
    pageOne.images[0],
    pageOne.images[1],
    pageTwo.images[1],
  ])
  assert.deepEqual(completeAuthoritativeRefresh({
    completed: 0,
    generation: 1,
    latest: 2,
    refreshed: true,
  }), { completed: 1, pending: true })
})

// AC: @grouped-folder-scope-navigation ac-scope-change
// AC: @grouped-folder-scope-navigation ac-same-scope
test('grouped folder path survives filters but not directory or library scope changes', () => {
  const current = { directoryId: 1, libraryId: 'library-a' }

  assert.equal(galleryScopePreservesFolder(current, { directoryId: 1, libraryId: 'library-a' }), true)
  assert.equal(galleryScopePreservesFolder(current, { directoryId: 2, libraryId: 'library-a' }), false)
  assert.equal(galleryScopePreservesFolder(current, { directoryId: 1, libraryId: 'library-b' }), false)
  assert.equal(galleryScopePreservesFolder(current, { directoryId: null, libraryId: 'library-a' }), false)
  assert.equal(galleryScopePreservesFolder(
    { directoryId: null, libraryId: null },
    { directoryId: null, libraryId: null }
  ), true)
})
