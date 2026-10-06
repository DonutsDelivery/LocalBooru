// SPDX-License-Identifier: GPL-3.0-only
import assert from 'node:assert/strict'
import test from 'node:test'
import { collectionRequestFilters, readCollectionFilters, sidebarFilterParams, toggleCollectionTag } from './galleryCollectionFilters.js'

test('remote filters use native whitespace tag terms and exclude local or private age fields', () => {
  const filters = readCollectionFilters(new URLSearchParams('q=cat,-dog&rating=pg&sort=oldest&timeframe=week&favorites=true&directory=7&filename=secret&watched=watched&min_age=18&max_age=30&resolution=1920x1080&orientation=portrait&duration=0-60'), true)
  assert.deepEqual(collectionRequestFilters(filters, true), { q: 'cat -dog', rating: 'pg', sort: 'oldest', favorites_only: true, timeframe: 'week', min_width: 1920, min_height: 1080, orientation: 'portrait', min_duration: 0, max_duration: 60 })
})
test('collection default ordering is omitted while explicit gallery sorting is preserved', () => {
  assert.deepEqual(collectionRequestFilters(readCollectionFilters(new URLSearchParams())), {})
  assert.deepEqual(collectionRequestFilters(readCollectionFilters(new URLSearchParams('tags=cat,-dog&sort=newest&min_age=0&directory=7&library=library-a&watched=in_progress'))), { sort: 'newest', min_age: 0, directory_id: 7, library_id: 'library-a', watched_status: 'in_progress', tags: 'cat', exclude_tags: 'dog' })
})
test('sidebar changes preserve media category and watched status across unrelated filters', () => {
  const params = new URLSearchParams('media_type=video&watched=watched')
  const next = sidebarFilterParams(params, false, 'cat', 'pg', 'oldest', true, null, null, null, 'week', '', null, null, { min: 0, max: 60 }, null)
  assert.equal(next.get('media_type'), 'video')
  assert.equal(next.get('watched'), 'watched')
  assert.equal(next.get('sort'), 'oldest')
  assert.equal(next.get('duration'), '0-60')
})
test('adding a sidebar tag preserves existing filter scope and toggles that tag only', () => {
  const params = new URLSearchParams('media_type=image&q=cat%20-dog&rating=pg&timeframe=week')
  const next = toggleCollectionTag(params, 'bird', true)
  assert.equal(next.get('q'), 'cat,-dog,bird')
  assert.equal(next.get('rating'), 'pg')
  assert.equal(next.get('timeframe'), 'week')
  assert.equal(toggleCollectionTag(next, 'bird', true).get('q'), 'cat,-dog')
})

test('remote sidebar serialization cannot retain inferred subject ages', () => {
  const next = sidebarFilterParams(new URLSearchParams('min_age=18&max_age=30'), true, '', 'pg,pg13', 'newest', false, null, 18, 30)
  assert.equal(next.has('min_age'), false)
  assert.equal(next.has('max_age'), false)
})

test('creator and explicit offline scope survive tag and sidebar changes only for remote galleries', () => {
  const params = new URLSearchParams('media_type=video&creator_id=account-7&offline_only=true&q=cat&rating=pg&creator_view=offline')
  assert.deepEqual(collectionRequestFilters(readCollectionFilters(params, true), true), { q: 'cat', creator_id: 'account-7', offline_only: true, rating: 'pg' })
  const changed = sidebarFilterParams(params, true, 'cat,-dog', 'pg13', 'oldest', false, null, null, null, 'week')
  assert.equal(changed.get('creator_id'), 'account-7')
  assert.equal(changed.get('offline_only'), 'true')
  assert.equal(changed.get('creator_view'), 'offline')
  const local = collectionRequestFilters(readCollectionFilters(params))
  assert.equal(local.creator_id, undefined)
  assert.equal(local.offline_only, undefined)
})
