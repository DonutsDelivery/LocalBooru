import assert from 'node:assert/strict'
import test from 'node:test'

import {
  directoryLibraryTargetValue,
  resolveDirectoryLibraryTarget,
} from './directoryLibraryTarget.js'

const libraries = [
  {
    uuid: 'primary-uuid',
    name: 'Primary',
    is_primary: true,
    mounted: true,
  },
  {
    uuid: 'archive-uuid',
    name: 'Archive',
    is_primary: false,
    mounted: true,
  },
  {
    uuid: 'offline-uuid',
    name: 'Offline',
    is_primary: false,
    mounted: false,
  },
]

test('directory additions preserve the explicitly selected auxiliary library UUID', () => {
  assert.equal(resolveDirectoryLibraryTarget(libraries, 'archive-uuid'), 'archive-uuid')
})

test('primary directory additions use the explicit primary library alias', () => {
  assert.equal(directoryLibraryTargetValue(libraries[0]), 'primary')
  assert.equal(resolveDirectoryLibraryTarget(libraries, 'primary'), 'primary')
})

test('directory additions reject missing and unmounted destination libraries', () => {
  assert.throws(
    () => resolveDirectoryLibraryTarget(libraries, 'offline-uuid'),
    /mounted destination library/
  )
  assert.throws(
    () => resolveDirectoryLibraryTarget(libraries, 'missing-uuid'),
    /mounted destination library/
  )
})
