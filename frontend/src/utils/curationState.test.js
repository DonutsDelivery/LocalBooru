import test from 'node:test'
import assert from 'node:assert/strict'
import { buildCurationQuery, imageLocatorKey, mergeCandidates, seedCandidates } from './curationState.js'

test('query preserves filters and forces non-favorite first page', () => {
  const query = buildCurationQuery({ library_id: 'lib', directory_id: null, tags: 'cat', sort: 'oldest' })
  assert.equal(query.library_id, 'lib')
  assert.equal(query.directory_id, null)
  assert.equal(query.tags, 'cat')
  assert.equal(query.sort, 'oldest')
  assert.equal(query.exclude_favorites, true)
  assert.equal(query.page, 1)
})

test('queue skips favorites and dedupes composite locators', () => {
  const a = { id: 1, directory_id: 2, library_id: 'a', is_favorite: false }
  const other = { ...a, library_id: 'b' }
  assert.equal(seedCandidates([a, { ...a, id: 2, is_favorite: true }]).length, 1)
  assert.notEqual(imageLocatorKey(a), imageLocatorKey(other))
  assert.deepEqual(mergeCandidates([a], [a, other]), [a, other])
})

test('queue keeps video candidates in the current gallery order', () => {
  const image = { id: 1, directory_id: 2, library_id: 'a', original_filename: 'still.png', is_favorite: false }
  const video = { id: 2, directory_id: 2, library_id: 'a', original_filename: 'clip.mp4', duration: 12, is_favorite: false }
  assert.deepEqual(seedCandidates([video, image]), [video, image])
})

test('repeated first-page refills reach items beyond fifty without skips', () => {
  const items = Array.from({ length: 61 }, (_, index) => ({
    id: index + 1,
    directory_id: 2,
    library_id: 'library',
    is_favorite: false,
  }))
  let eligible = [...items]
  let queue = eligible.slice(0, 50)
  const visited = []
  while (queue.length) {
    const current = queue.shift()
    visited.push(current.id)
    eligible = eligible.filter(item => item.id !== current.id)
    queue = mergeCandidates(queue, eligible.slice(0, 50))
  }
  assert.deepEqual(visited, items.map(item => item.id))
})
