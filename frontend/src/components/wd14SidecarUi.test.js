import assert from 'node:assert/strict'
import test from 'node:test'

import {
  buildWd14DirectoryOptions,
  buildWd14RequestDirectories,
  getWd14Failures,
  getWd14SummaryItems,
  wd14ConfirmationMessage,
  wd14DirectoryKey,
  wd14StatusLabel,
} from './wd14SidecarUi.js'

// AC: @wd14-addon-settings ac-managed-selection
test('uses compound identities for colliding directory IDs and emits no paths', () => {
  const libraries = [
    { uuid: 'library-a', name: 'Primary', mounted: true },
    { uuid: 'library-b', name: 'Archive', mounted: true },
    { uuid: 'library-c', name: 'Offline', mounted: false },
  ]
  const directories = [
    { id: 7, library_id: 'library-a', name: 'Art', path: '/media/art', image_count: 3 },
    { id: 7, library_id: 'library-b', name: 'Art', path: '/archive/art', image_count: 5 },
    { id: 8, library_id: 'library-c', name: 'Hidden', path: '/offline', image_count: 1 },
  ]

  const options = buildWd14DirectoryOptions(directories, libraries)
  assert.equal(options.length, 2)
  assert.notEqual(options[0].key, options[1].key)
  assert.equal(wd14DirectoryKey('library-a', 7), 'library-a:7')

  const request = buildWd14RequestDirectories(
    options,
    new Set(options.map(option => option.key)),
  )
  assert.deepEqual(
    request.sort((left, right) => left.library_id.localeCompare(right.library_id)),
    [
      { library_id: 'library-a', directory_id: 7 },
      { library_id: 'library-b', directory_id: 7 },
    ],
  )
  assert.equal(JSON.stringify(request).includes('/media'), false)
  assert.equal(JSON.stringify(request).includes('path'), false)
})

// AC: @wd14-addon-settings ac-operation-controls
test('requires confirmation only for destructive operation modes', () => {
  assert.equal(wd14ConfirmationMessage('import', false, 2), null)
  assert.equal(wd14ConfirmationMessage('export', false, 2), null)
  assert.match(wd14ConfirmationMessage('absorb', false, 2), /permanently deleted/)
  assert.match(wd14ConfirmationMessage('export', true, 1), /atomically overwritten/)
})

// AC: @wd14-addon-settings ac-visible-results
test('formats aggregate values and isolates actionable sidecar failures', () => {
  const results = [
    { sidecar_path: '/media/a.txt', status: 'imported', error: null },
    { sidecar_path: '/media/b.txt', status: 'skipped_missing', error: null },
    { sidecar_path: '/media/c.txt', status: 'failed_read', error: 'Permission denied' },
    { sidecar_path: '/media/d.txt', status: 'imported_not_removed', error: null },
  ]

  assert.deepEqual(
    getWd14Failures(results).map(result => result.sidecar_path),
    ['/media/c.txt', '/media/d.txt'],
  )
  assert.equal(wd14StatusLabel('imported_not_removed'), 'Imported Not Removed')
  assert.deepEqual(
    Object.fromEntries(getWd14SummaryItems({
      directories: 2,
      sidecars_succeeded: 1,
      sidecars_skipped: 1,
      sidecars_failed: 2,
      tags_added: 4,
    })),
    {
      Directories: 2,
      'Media candidates': 0,
      'Sidecars found': 0,
      Succeeded: 1,
      Skipped: 1,
      Failed: 2,
      'Tags parsed': 0,
      'Tags added': 4,
      'Files written': 0,
      'Files removed': 0,
    },
  )
})
