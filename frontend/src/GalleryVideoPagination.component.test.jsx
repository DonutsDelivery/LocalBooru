import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const api = vi.hoisted(() => ({
  fetchImages: vi.fn(), fetchTags: vi.fn(), getLibraryStats: vi.fn(),
  getContinueWatching: vi.fn(), subscribeToLibraryEvents: vi.fn(() => () => {}), healthCheck: vi.fn(),
  getMediaUrl: path => path,
}))
vi.mock('./api', async importOriginal => ({ ...await importOriginal(), ...api }))
vi.mock('./components/Sidebar', () => ({ default: () => <aside>Video filters</aside> }))
vi.mock('./components/TitleBar', () => ({ default: () => null }))
vi.mock('./components/LicenseNotices', () => ({ default: () => null }))
vi.mock('./components/Music/MusicExperience', () => ({
  MusicPlayerProvider: ({ children }) => children, MusicPage: () => null, PersistentMusicPlayer: () => null,
}))
vi.mock('./hooks/useAddonStatus', () => ({ useAllAddonStatuses: () => ({ isInstalled: () => false }) }))
vi.mock('./hooks/useImageWorkflow', () => ({ useImageWorkflow: () => ({ available: false }) }))

import App from './App'
import GalleryTileSizeControl from './components/GalleryTileSizeControl'

class ControlledIntersectionObserver {
  static instances = []
  constructor(callback, options) {
    this.callback = callback
    this.options = options
    this.targets = new Set()
    ControlledIntersectionObserver.instances.push(this)
  }
  observe(target) { this.targets.add(target) }
  unobserve(target) { this.targets.delete(target) }
  disconnect() { this.targets.clear() }
  intersect(target) {
    if (this.targets.has(target)) this.callback([{ target, isIntersecting: true, intersectionRatio: 1 }])
  }
}

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  window.history.replaceState({}, '', '/videos?library=synthetic-library&directory=1&folder=synthetic-videos')
  vi.clearAllMocks()
  ControlledIntersectionObserver.instances = []
  vi.stubGlobal('IntersectionObserver', ControlledIntersectionObserver)
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1280 })
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: 720 })
  api.healthCheck.mockResolvedValue({ status: 'ok' })
  api.fetchTags.mockResolvedValue({ tags: [] })
  api.getLibraryStats.mockResolvedValue({ total_images: 263 })
  api.getContinueWatching.mockResolvedValue({ items: [] })
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

// AC: @local-gallery-completeness ac-all-pages
test('local Videos pages all 263 synthetic records inside its own scroller while thumbnails are still loading', async () => {
  const videos = Array.from({ length: 263 }, (_, index) => ({
    id: index + 1, directory_id: 1, library_id: 'synthetic-library', filename: `synthetic-${index + 1}.mp4`,
    thumbnail_url: `/synthetic/${index + 1}.webp`, url: `/synthetic/${index + 1}.mp4`, width: 1920, height: 1080,
  }))
  api.fetchImages.mockImplementation(async ({ page, per_page }) => ({
    images: videos.slice((page - 1) * per_page, page * per_page), total: videos.length,
  }))
  const { container } = render(<App />)
  await waitFor(() => expect(container.querySelectorAll('.media-item').length).toBe(50))
  const gallery = container.querySelector('.content > .masonry-container')
  expect(gallery).toBeTruthy()
  const sentinel = gallery.querySelector('.load-more-trigger')
  const observer = ControlledIntersectionObserver.instances.find(instance => instance.targets.has(sentinel))
  expect(observer).toBeTruthy()
  const query = window.location.search
  for (const expected of [100, 150, 200, 250, 263]) {
    // JSDOM has no layout engine. Simulate the nested scroller reaching the
    // observed sentinel; the real Gallery, MasonryGrid and MediaItem stay mounted.
    fireEvent.scroll(gallery, { target: { scrollTop: expected * 100 } })
    act(() => observer.intersect(sentinel))
    await waitFor(() => expect(container.querySelectorAll('.media-item').length).toBe(expected))
  }
  expect(api.fetchImages.mock.calls.map(([query]) => query.page)).toEqual([1, 2, 3, 4, 5, 6])
  expect(api.fetchImages.mock.calls.every(([query]) => query.media_type === 'video'
    && query.per_page === 50 && query.library_id === 'synthetic-library'
    && query.directory_id === 1 && query.import_source === 'synthetic-videos')).toBe(true)
  expect(new Set([...container.querySelectorAll('.media-item')].map(tile => tile.dataset.imageKey)).size).toBe(263)
  expect(gallery.querySelectorAll('img[loading="lazy"]').length).toBe(263)
  // No thumbnail onLoad was fired: artwork loading cannot cap metadata pagination.
  expect(gallery.querySelectorAll('.media-item.loading').length).toBe(263)
  act(() => observer.intersect(sentinel))
  await new Promise(resolve => setTimeout(resolve, 300))
  expect(api.fetchImages).toHaveBeenCalledTimes(6)
  expect(window.location.search).toBe(query)
})

// AC: @local-gallery-completeness ac-slider
test('tile size retains the original single-range markup and numeric 1 through 5 behavior', () => {
  const change = vi.fn()
  const { container, rerender } = render(<GalleryTileSizeControl tileSize={3} onChange={change} />)
  const slider = screen.getByRole('slider', { name: 'Tile size' })
  expect(slider.parentElement.className).toBe('tile-size-control')
  expect(container.querySelector('.tile-size-range')).toBeNull()
  expect(container.querySelector('.tile-size-scale')).toBeNull()
  expect(slider.min).toBe('1')
  expect(slider.max).toBe('5')
  fireEvent.change(slider, { target: { value: '5' } })
  expect(change).toHaveBeenCalledWith(5)
  rerender(<GalleryTileSizeControl tileSize={5} onChange={change} />)
  expect(slider.value).toBe('5')
})
