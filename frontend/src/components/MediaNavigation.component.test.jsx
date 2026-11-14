import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'

const api = vi.hoisted(() => ({
  fetchCollections: vi.fn(), createCollection: vi.fn(), deleteCollection: vi.fn(),
  fetchDirectories: vi.fn(), getSavedSearches: vi.fn(),
  fetchLibraries: vi.fn(), fetchMusicFacets: vi.fn(), fetchMusicCollections: vi.fn(),
  fetchMusicAlbums: vi.fn(), fetchMusicTracks: vi.fn(),
}))
const music = vi.hoisted(() => ({
  useMusicPlayer: vi.fn(),
}))
vi.mock('../api', () => ({
  ...api,
  getFileDimensions: vi.fn(), createSavedSearch: vi.fn(), deleteSavedSearch: vi.fn(),
  getMediaUrl: path => path,
}))
vi.mock('./Sidebar/FilterControls', () => ({
  default: () => null, ALL_RATINGS: ['pg', 'pg13', 'r', 'x', 'xxx'],
  MIN_AGE_LIMIT: 0, MAX_AGE_LIMIT: 100, RESOLUTION_OPTIONS: [],
  ORIENTATION_OPTIONS: [], DURATION_OPTIONS: [], SORT_OPTIONS: [{ value: 'newest', label: 'Newest' }],
}))
vi.mock('./Sidebar/TagSearch', () => ({ default: () => null }))
vi.mock('./Sidebar/PromptSection', () => ({ default: () => null }))
vi.mock('./FamilyModeLock', () => ({ default: () => null }))
vi.mock('../tauriAPI', () => ({ getDesktopAPI: () => null, isDesktopApp: () => false }))
vi.mock('./Music/MusicPlayer', () => music)

import MediaSectionsNav from './MediaSectionsNav'
import Sidebar from './Sidebar'
import MusicPage from './Music/MusicPage'
import CollectionsPage from '../pages/CollectionsPage'

function LocationProbe() {
  const location = useLocation()
  return <output data-testid="route">{location.pathname + location.search}</output>
}

beforeEach(() => {
  const stored = new Map()
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: {
    getItem: key => stored.get(key) ?? null,
    setItem: (key, value) => stored.set(key, String(value)),
    removeItem: key => stored.delete(key),
    clear: () => stored.clear(),
  } })
  sessionStorage.clear()
  vi.clearAllMocks()
  api.fetchCollections.mockImplementation(async type => ({ collections: [{ id: type === 'video' ? 2 : 1, name: `${type} set`, item_count: 1 }] }))
  api.createCollection.mockImplementation(async (name, description, mediaType) => ({ id: 4, name, media_type: mediaType, item_count: 0 }))
  api.fetchDirectories.mockResolvedValue({ directories: [] })
  api.getSavedSearches.mockResolvedValue({ searches: [] })
  api.fetchLibraries.mockResolvedValue({ libraries: [] })
  api.fetchMusicFacets.mockResolvedValue({ artists: [], albums: [], genres: [], years: [], folders: [] })
  api.fetchMusicCollections.mockResolvedValue({ collections: [{ id: 3, name: 'Music set', item_count: 1 }] })
  api.fetchMusicAlbums.mockResolvedValue({ albums: [], total: 0 })
  api.fetchMusicTracks.mockResolvedValue({ tracks: [], total: 0 })
  music.useMusicPlayer.mockReturnValue({
    browse: { mode: 'albums', byMode: { albums: {
      query: '', artist: '', album: '', genre: '', year: '', favorites: false,
      folder: '', collection: '', library: '', scroll: 0, page: 1,
    } } },
    setBrowse: vi.fn(), updateBrowse: vi.fn(), startSong: vi.fn(), openAlbum: vi.fn(),
    queueTrack: vi.fn(), session: null,
  })
})
afterEach(() => cleanup())

test('the library switcher lives in the sidebar and restores galleries, not collection pages', async () => {
  sessionStorage.setItem('donutMediaCenter_gallery_url_image', '/?sort=oldest')
  sessionStorage.setItem('donutMediaCenter_section_url_video', '/collections?media_type=video')
  sessionStorage.setItem('donutMediaCenter_gallery_url_video', '/videos?sort=oldest')
  render(<MemoryRouter initialEntries={['/collections?media_type=video']}>
    <Sidebar mediaType="video" /><LocationProbe />
  </MemoryRouter>)
  const nav = screen.getByRole('navigation', { name: 'Navigation' })
  expect(within(nav).getAllByRole('link')).toHaveLength(5)
  expect(within(nav).queryByRole('link', { name: 'Collections' })).toBeNull()
  expect(within(nav).getByRole('link', { name: 'Videos' }).getAttribute('aria-current')).toBe('page')
  expect(within(nav).getByRole('link', { name: 'Videos' }).getAttribute('href')).toBe('/videos?sort=oldest')
  expect(screen.getByRole('link', { name: 'View all' }).getAttribute('href')).toBe('/collections?media_type=video')
  await waitFor(() => expect(screen.getByRole('link', { name: /video set/ }).getAttribute('href')).toBe('/collections/2?media_type=video'))
  fireEvent.click(within(nav).getByRole('link', { name: 'Videos' }))
  expect(screen.getByTestId('route').textContent).toBe('/videos?sort=oldest')
  fireEvent.click(within(nav).getByRole('link', { name: 'Images' }))
  expect(screen.getByTestId('route').textContent).toBe('/?sort=oldest')
  expect(within(nav).getByRole('link', { name: 'Music' }).getAttribute('href')).toBe('/music')
  expect(api.fetchCollections).toHaveBeenCalledWith('video')
})

test('image collections are only reached from the image gallery sidebar', async () => {
  render(<MemoryRouter initialEntries={['/']}><Sidebar mediaType="image" /></MemoryRouter>)
  const nav = screen.getByRole('navigation', { name: 'Navigation' })
  expect(within(nav).queryByRole('link', { name: 'Collections' })).toBeNull()
  expect(screen.getByRole('link', { name: 'View all' }).getAttribute('href')).toBe('/collections?media_type=image')
  await waitFor(() => expect(screen.getByRole('link', { name: /image set/ }).getAttribute('href')).toBe('/collections/1?media_type=image'))
  expect(api.fetchCollections).toHaveBeenCalledWith('image')
})

test('music keeps its own collections and the same switcher in its sidebar, not above the grid', async () => {
  render(<MemoryRouter initialEntries={['/music']}><MusicPage /></MemoryRouter>)
  const aside = document.querySelector('.music-sidebar')
  const main = document.querySelector('.music-main')
  const nav = within(aside).getByRole('navigation', { name: 'Media libraries' })
  expect(within(nav).getAllByRole('link')).toHaveLength(3)
  expect(within(nav).getByRole('link', { name: 'Music' }).getAttribute('aria-current')).toBe('page')
  expect(main.querySelector('nav')).toBeNull()
  await waitFor(() => expect(within(aside).getByRole('button', { name: /Music set/ })).toBeTruthy())
  expect(api.fetchMusicCollections).toHaveBeenCalledWith(undefined)
  expect(api.fetchCollections).not.toHaveBeenCalled()
  fireEvent.click(within(aside).getByRole('button', { name: /Music set/ }))
  expect(music.useMusicPlayer().updateBrowse).toHaveBeenCalledWith('albums', { collection: '3', scroll: 0, page: 1 })
})

test('video collections page only displays and creates video collections', async () => {
  render(<MemoryRouter initialEntries={['/collections?media_type=video']}><CollectionsPage /></MemoryRouter>)
  const main = document.querySelector('.collections-page')
  await waitFor(() => expect(within(main).getByText('video set')).toBeTruthy())
  expect(within(main).queryByText('image set')).toBeNull()
  expect(within(main).getByRole('heading', { name: 'Video Collections' })).toBeTruthy()
  expect(main.querySelector('nav')).toBeNull()
  fireEvent.click(within(main).getByRole('button', { name: 'New Collection' }))
  fireEvent.change(within(main).getByPlaceholderText('Collection name...'), { target: { value: 'Clips' } })
  fireEvent.click(within(main).getByRole('button', { name: 'Create' }))
  await waitFor(() => expect(api.createCollection).toHaveBeenCalledWith('Clips', null, 'video'))
  expect(within(main).getByText('Clips')).toBeTruthy()
})

test('the media switcher highlights the owning library on collection detail and rejects stale cross-section targets', () => {
  sessionStorage.setItem('donutMediaCenter_gallery_url_video', '/collections?media_type=image')
  render(<MemoryRouter initialEntries={['/collections/3?media_type=video']}>
    <nav aria-label="Media libraries"><MediaSectionsNav /></nav>
  </MemoryRouter>)
  const nav = screen.getByRole('navigation', { name: 'Media libraries' })
  expect(within(nav).getByRole('link', { name: 'Videos' }).getAttribute('aria-current')).toBe('page')
  expect(within(nav).getByRole('link', { name: 'Videos' }).getAttribute('href')).toBe('/videos')
})
