// SPDX-License-Identifier: GPL-3.0-only
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
const api = vi.hoisted(() => ({ getConnectedHubs: vi.fn(), fetchCollections: vi.fn(), fetchDirectories: vi.fn(), getFileDimensions: vi.fn(), getSavedSearches: vi.fn(), createSavedSearch: vi.fn(), deleteSavedSearch: vi.fn(), searchTags: vi.fn() }))
vi.mock('../../api', () => api)
vi.mock('../FamilyModeLock', () => ({ default: () => null }))
vi.mock('../Music/PersistentMusicPlayer', () => ({ default: () => null }))
import Sidebar from './Sidebar'
import { REMOTE_SORT_VALUES } from '../../utils/galleryCollectionFilters'
import GalleryTileSizeControl from '../GalleryTileSizeControl'
import { useGalleryTileSize } from '../../hooks/useGalleryTileSize'
const search = vi.fn()
beforeEach(() => {
  vi.resetAllMocks(); localStorage.clear(); sessionStorage.clear(); localStorage.setItem('filtersExpanded', 'true')
  api.getConnectedHubs.mockResolvedValue({ hubs: [] }); api.fetchCollections.mockResolvedValue({ collections: [] }); api.fetchDirectories.mockResolvedValue({ directories: [] }); api.getSavedSearches.mockResolvedValue({ searches: [] }); api.searchTags.mockResolvedValue([])
})
afterEach(() => { cleanup(); vi.useRealTimers() })

test('remote collection uses source tags and realm ratings without local or inferred-age controls', async () => {
  const remoteTags = vi.fn().mockResolvedValue([{ name: 'cat_sky', category: 'general', count: 2 }])
  render(<MemoryRouter initialEntries={['/shared/main']}><Sidebar filterScope="remote" availableRatings={['pg', 'pg13']} supportedSortValues={REMOTE_SORT_VALUES} onSearch={search} onTagClick={vi.fn()} searchSuggestions={remoteTags} tags={[{ name: 'popular_tag', category: 'general', count: 3 }]} /></MemoryRouter>)
  expect(screen.getByRole('button', { name: 'PG', exact: true })).toBeTruthy()
  expect(screen.getByRole('button', { name: 'PG13', exact: true })).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'XXX', exact: true })).toBeNull()
  expect(screen.queryByText('All Directories')).toBeNull()
  expect(screen.queryByPlaceholderText('Search by filename...')).toBeNull()
  expect(screen.queryByText('Group by Folders')).toBeNull()
  expect(screen.queryByText('Age Range:')).toBeNull()
  expect(screen.queryByText('Watched Status')).toBeNull()
  expect(screen.queryByRole('option', { name: 'Folder (A-Z)' })).toBeNull()
  expect(screen.queryByTitle('Save current filters')).toBeNull()
  expect(screen.getByRole('button', { name: /popular tag\s*\(3\)/ })).toBeTruthy()
  fireEvent.change(screen.getByPlaceholderText('Search tag or -tag...'), { target: { value: 'cat' } })
  await waitFor(() => expect(remoteTags).toHaveBeenCalledWith('cat', 10))
  await screen.findByRole('button', { name: /cat sky\s*\(2\)/ })
  expect(api.searchTags).not.toHaveBeenCalled()
  expect(api.fetchDirectories).not.toHaveBeenCalled()
})

test('local collection retains its saved order default and exposes normal local filters', () => {
  render(<MemoryRouter initialEntries={['/collections/7']}><Sidebar defaultSort="collection" onSearch={search} onTagClick={vi.fn()} tags={[]} /></MemoryRouter>)
  expect(screen.getByRole('combobox', { name: 'Sort by' }).value).toBe('collection')
  expect(screen.getByRole('option', { name: 'Collection Order' })).toBeTruthy()
  expect(screen.getByText('All Directories')).toBeTruthy()
  expect(screen.getByPlaceholderText('Search by filename...')).toBeTruthy()
  expect(screen.getByText('Age Range:')).toBeTruthy()
  expect(screen.queryByText('Group by Folders')).toBeNull()
  fireEvent.change(screen.getByRole('combobox', { name: 'Sort by' }), { target: { value: 'oldest' } })
  expect(search).toHaveBeenLastCalledWith('', 'pg,pg13,r,x,xxx', 'oldest', false, null, null, null, null, '', null, null, null, null)
})

test('local saved searches exclude filters belonging to remote communities', async () => {
  api.getSavedSearches.mockResolvedValue({ searches: [{ id: 1, name: 'Remote source filters', filters: { remote_source_id: 'main', media_type: 'image' } }, { id: 2, name: 'Local filters', filters: { media_type: 'image' } }] })
  render(<MemoryRouter initialEntries={['/collections/7']}><Sidebar defaultSort="collection" onSearch={search} onTagClick={vi.fn()} tags={[]} /></MemoryRouter>)
  await screen.findByRole('button', { name: 'Local filters' })
  expect(screen.queryByRole('button', { name: 'Remote source filters' })).toBeNull()
})

function Tiles() { const [tileSize, setTileSize] = useGalleryTileSize(); return <GalleryTileSizeControl tileSize={tileSize} onChange={setTileSize} /> }
test('the shared regular tile control uses XS/S/M/L/XL and the gallery setting', () => {
  localStorage.setItem('localbooru_tileSize', '2')
  render(<Tiles />)
  expect(screen.getByRole('slider', { name: 'Tile size' }).value).toBe('2')
  expect(screen.getByRole('slider').min).toBe('1')
  expect(screen.getByRole('slider').max).toBe('5')
  fireEvent.change(screen.getByRole('slider', { name: 'Tile size' }), { target: { value: '5' } })
  expect(localStorage.getItem('localbooru_tileSize')).toBe('5')
  expect(screen.getByRole('slider').value).toBe('5')
})

test('remote tag search never falls back to the local library when source suggestions are unavailable', async () => {
  vi.useFakeTimers()
  render(<MemoryRouter initialEntries={['/shared/main']}><Sidebar filterScope="remote" availableRatings={['pg', 'pg13']} onSearch={search} onTagClick={vi.fn()} tags={[]} /></MemoryRouter>)
  fireEvent.change(screen.getByPlaceholderText('Search tag or -tag...'), { target: { value: 'cat' } })
  await act(async () => { await vi.advanceTimersByTimeAsync(200) })
  expect(api.searchTags).not.toHaveBeenCalled()
})

test('creator collections follow regular media filters and remain before tag browsing', () => {
  render(<MemoryRouter initialEntries={['/shared/main']}><Sidebar filterScope="remote" availableRatings={['pg', 'pg13']} onSearch={search} onTagClick={vi.fn()} searchSuggestions={vi.fn()} tags={[]} sidebarActions={<h3>Community title</h3>} sidebarCollections={<section aria-label="Creator collection controls">Creator controls</section>} /></MemoryRouter>)
  const filterSection = screen.getByRole('combobox', { name: 'Sort by' }).closest('.search-section')
  const creatorSection = screen.getByRole('region', { name: 'Creator collection controls' })
  const tagSection = screen.getByPlaceholderText('Search tag or -tag...')
  expect(filterSection.compareDocumentPosition(creatorSection) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  expect(creatorSection.compareDocumentPosition(tagSection) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
})

function CollectionRoute() {
  const location = useLocation()
  return <output data-testid="collection-route">{JSON.stringify({ url: location.pathname + location.search, state: location.state })}</output>
}

// AC: @collections ac-collection-sidebar
// AC: @collections ac-gallery-back
test('shared collections appear beside local collection navigation and preserve the owning gallery on selection', async () => {
  const close = vi.fn()
  api.getConnectedHubs.mockResolvedValue({ hubs: [
    { source_id: 'main', display_name: 'DonutBooru', enabled: true, transport_mode: 'relayed' },
    { source_id: 'r18', display_name: 'DonutBooru R18', enabled: true, transport_mode: 'relayed' },
  ] })
  render(<MemoryRouter initialEntries={[{ pathname: '/collections/7', search: '?media_type=video', state: { galleryReturnTo: '/videos?directory_id=7' } }]}><Sidebar mediaType="video" onClose={close} collectionItems={[{ id: 7, name: 'Local videos', item_count: 3 }]} /><CollectionRoute /></MemoryRouter>)
  const sidebar = document.querySelector('aside')
  const community = await within(sidebar).findByRole('link', { name: 'DonutBooru', exact: true })
  expect(within(sidebar).getByRole('link', { name: /Local videos/ }).getAttribute('href')).toBe('/collections/7?media_type=video')
  expect(community.getAttribute('href')).toBe('/shared/main?media_type=video')
  fireEvent.click(community)
  await waitFor(() => expect(community.getAttribute('aria-current')).toBe('page'))
  expect(close).toHaveBeenCalledOnce()
  const route = JSON.parse(screen.getByTestId('collection-route').textContent)
  expect(route.state.galleryReturnTo).toBe('/videos?directory_id=7')
  expect(within(sidebar).getByRole('link', { name: '← Back to Videos' }).getAttribute('href')).toBe('/videos?directory_id=7')
  const adult = within(sidebar).getByRole('link', { name: 'DonutBooru R18', exact: true })
  fireEvent.click(adult)
  expect(adult.getAttribute('aria-current')).toBe('page')
  expect(community.getAttribute('aria-current')).toBeNull()
})
