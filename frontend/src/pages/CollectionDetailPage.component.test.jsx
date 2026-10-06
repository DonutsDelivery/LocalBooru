// SPDX-License-Identifier: GPL-3.0-only
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
const api = vi.hoisted(() => ({ fetchCollection: vi.fn(), updateCollection: vi.fn(), removeFromCollection: vi.fn(), getConnectedHubs: vi.fn(), fetchCollections: vi.fn(), fetchDirectories: vi.fn(), getFileDimensions: vi.fn(), getSavedSearches: vi.fn(), createSavedSearch: vi.fn(), deleteSavedSearch: vi.fn(), searchTags: vi.fn() }))
vi.mock('../api', () => api)
vi.mock('../components/FamilyModeLock', () => ({ default: () => null }))
vi.mock('../components/Music/PersistentMusicPlayer', () => ({ default: () => null }))
vi.mock('../components/MasonryGrid', () => ({ default: ({ images, onLoadMore, loading }) => <div className="masonry-container">{images.map(image => <span key={image.id}>Image {image.id}</span>)}<button disabled={loading} onClick={onLoadMore}>Load next page</button></div> }))
vi.mock('../components/Lightbox', () => ({ default: () => null }))
import CollectionDetailPage from './CollectionDetailPage'
const response = (ids, hasMore = false) => ({ id: 7, name: 'Example collection', item_count: ids.length, images: ids.map(id => ({ id, directory_id: 1, library_id: 'primary', filename: 'example.jpg' })), has_more: hasMore })
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done }); return { promise, resolve } }
const mount = (url = '/collections/7?media_type=image') => render(<MemoryRouter initialEntries={[url]}><Routes><Route path="/collections/:id" element={<CollectionDetailPage />} /></Routes></MemoryRouter>)
beforeEach(() => { vi.resetAllMocks(); localStorage.clear(); sessionStorage.clear(); api.getConnectedHubs.mockResolvedValue({ hubs: [] }); api.fetchCollections.mockResolvedValue({ collections: [] }); api.fetchDirectories.mockResolvedValue({ directories: [] }); api.getSavedSearches.mockResolvedValue({ searches: [] }); api.searchTags.mockResolvedValue([]) })
afterEach(cleanup)

test('local collection uses sidebar filters and preserves them on the next server page', async () => {
  api.fetchCollection.mockResolvedValueOnce(response([1], true)).mockResolvedValueOnce(response([2]))
  mount('/collections/7?media_type=image&tags=cat,-dog&rating=pg&sort=oldest&timeframe=week&favorites=true&min_age=18')
  await screen.findByText('Image 1')
  expect(document.querySelector('.collection-detail-header')).toBeNull()
  expect(screen.getByRole('combobox', { name: 'Sort by' }).closest('aside')).not.toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Load next page' }))
  await screen.findByText('Image 2')
  expect(api.fetchCollection).toHaveBeenLastCalledWith('7', 2, 50, 'image', { tags: 'cat', exclude_tags: 'dog', rating: 'pg', sort: 'oldest', favorites_only: true, min_age: 18, timeframe: 'week' })
})

test('new collection filters discard a late previous result instead of mixing loaded pages', async () => {
  const old = deferred()
  api.fetchCollection.mockReturnValueOnce(old.promise).mockResolvedValueOnce(response([2]))
  mount()
  expect(screen.getByRole('combobox', { name: 'Sort by' }).value).toBe('collection')
  fireEvent.change(screen.getByRole('combobox', { name: 'Sort by' }), { target: { value: 'oldest' } })
  await screen.findByText('Image 2')
  await act(async () => old.resolve(response([1])))
  expect(screen.queryByText('Image 1')).toBeNull()
  expect(api.fetchCollection).toHaveBeenLastCalledWith('7', 1, 50, 'image', { sort: 'oldest' })
})


test('a late rename cannot overwrite the next collection sidebar', async () => {
  const rename = deferred()
  api.updateCollection.mockReturnValue(rename.promise)
  api.fetchCollections.mockResolvedValue({ collections: [{ id: 8, name: 'Other collection', item_count: 1 }] })
  api.fetchCollection.mockResolvedValueOnce(response([1])).mockResolvedValueOnce({ ...response([2]), id: 8, name: 'Other collection' })
  mount()
  await screen.findByText('Image 1')
  fireEvent.click(screen.getByRole('button', { name: 'Rename Collection' }))
  fireEvent.change(screen.getByRole('textbox', { name: 'Collection name' }), { target: { value: 'Changed old collection' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  fireEvent.click(screen.getByRole('link', { name: /Other collection/ }))
  await screen.findByText('Image 2')
  await act(async () => rename.resolve({}))
  expect(screen.queryByText('Changed old collection')).toBeNull()
  expect(screen.getByRole('heading', { name: 'Other collection' })).not.toBeNull()
  expect(api.updateCollection).toHaveBeenCalledWith('7', { name: 'Changed old collection' })
})
