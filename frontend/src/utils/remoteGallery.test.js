// SPDX-License-Identifier: GPL-3.0-only
import assert from 'node:assert/strict'
import test from 'node:test'
import { appendRemoteGalleryItems, remoteGalleryItem, remotePageHasMore } from './remoteGallery.js'
const fixture = overrides => ({ source_id: 'source/a', remote_post_id: 'post:4', tags: ['one'], title: 'Caption', media: [{ kind: 'original', asset_url: 'https://untrusted.invalid/file.jpg', mime_type: 'image/jpeg', width: 800, height: 1200 }], ...overrides })
// AC: @creator-community-replication ac-4
test('personal offline viewer URLs explicitly select retained media without changing online reads', () => {
  const item = fixture({ media: [
    { kind: 'original', mime_type: 'image/jpeg' },
    { kind: 'thumbnail', mime_type: 'image/webp' },
  ] })
  const offline = remoteGalleryItem(item, { offlineOnly: true })
  assert.match(offline.url, /\/asset\/original\?offline_only=true$/)
  assert.match(offline.thumbnail_url, /\/asset\/thumbnail\?offline_only=true$/)
  assert.doesNotMatch(remoteGalleryItem(item).url, /offline_only/)
})
test('preserves source identity, dimensions and protected proxy paths without local locators', () => {
  const image = remoteGalleryItem(fixture())
  assert.equal(image.id, 'remote:source%2Fa:post%3A4')
  assert.equal(image.width, 800)
  assert.equal(image.height, 1200)
  assert.equal(image.url, '/api/online/sources/source%2Fa/items/post%3A4/asset/original')
  assert.equal(image.thumbnail_url, image.url)
  assert.equal(image.title, 'Caption')
  assert.equal(image.prompt, undefined)
  assert.deepEqual(image.tags, [{ name: 'one' }])
  assert.equal(image.directory_id, undefined)
  assert.equal(image.library_id, undefined)
  assert.equal(image._remote, true)
  assert.throws(() => remoteGalleryItem(fixture(), { sourceId: 'different' }), /another source/)
})
test('video originals are never used as image previews', () => {
  const image = remoteGalleryItem(fixture({ media: [{ kind: 'original', mime_type: 'video/webm', width: 1280, height: 720 }] }))
  assert.match(image.filename, /\.webm$/)
  assert.equal(image.thumbnail_url, '')
  assert.match(image.file_url, /\/asset\/original$/)
})
test('overlapping pages append once while preserving post IDs from different sources', () => {
  const first = remoteGalleryItem(fixture())
  const other = remoteGalleryItem(fixture({ source_id: 'another' }))
  assert.deepEqual(appendRemoteGalleryItems([first], [first, other, other]), [first, other])
})
test('empty filtered pages continue when the catalog has more pages', () => {
  assert.equal(remotePageHasMore({ page: 1, per_page: 60, items: [], total: 121 }), true)
  assert.equal(remotePageHasMore({ page: 3, per_page: 60, items: [], total: 121 }), false)
  assert.equal(remotePageHasMore({ page: 1, per_page: 60, items: [], total: null, next_cursor: '2' }), true)
})
