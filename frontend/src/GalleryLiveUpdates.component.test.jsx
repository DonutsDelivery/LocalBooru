import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const api = vi.hoisted(() => ({
  fetchImages: vi.fn(), fetchTags: vi.fn(), getLibraryStats: vi.fn(),
  subscribeToLibraryEvents: vi.fn(() => () => {}), healthCheck: vi.fn(),
  getMediaUrl: path => path,
}))
vi.mock('./api', async importOriginal => ({ ...await importOriginal(), ...api }))
vi.mock('./components/Sidebar', () => ({ default: () => <aside>Filters</aside> }))
vi.mock('./components/TitleBar', () => ({ default: () => null }))
vi.mock('./components/LicenseNotices', () => ({ default: () => null }))
vi.mock('./components/Music/MusicExperience', () => ({
  MusicPlayerProvider: ({ children }) => children, MusicPage: () => null, PersistentMusicPlayer: () => null,
}))
vi.mock('./hooks/useAddonStatus', () => ({ useAllAddonStatuses: () => ({ isInstalled: () => false }) }))
vi.mock('./hooks/useImageWorkflow', () => ({ useImageWorkflow: () => ({ available: false }) }))
vi.mock('./components/Lightbox', () => ({
  default: ({ images, currentIndex, total }) => {
    const image = images[currentIndex]
    return <div data-testid="lightbox" data-image-id={image?.id} data-index={currentIndex} data-total={total} />
  },
}))

import App from './App'

class ControlledIntersectionObserver {
  static instances = []
  constructor(callback) {
    this.callback = callback
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

const media = id => ({
  id, directory_id: 1, library_id: 'synthetic-library', filename: `synthetic-${id}.png`,
  original_filename: `synthetic-${id}.png`, thumbnail_url: `/synthetic/${id}.webp`,
  url: `/synthetic/${id}.png`, width: 400, height: 300,
})
const TILE_HEIGHT = 100
const key = id => `synthetic-library:1:${id}`

// The in-memory gallery: newest first, so ids count down.
let library
const newestFirst = () => [...library].sort((a, b) => b.id - a.id)

let emit
let run = 0
beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  // The gallery caches its last view per query string; keep tests independent.
  window.history.replaceState({}, '', `/?run=${++run}`)
  vi.clearAllMocks()
  ControlledIntersectionObserver.instances = []
  vi.stubGlobal('IntersectionObserver', ControlledIntersectionObserver)
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1280 })
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: 720 })
  api.healthCheck.mockResolvedValue({ status: 'ok' })
  api.fetchTags.mockResolvedValue({ tags: [] })
  api.getLibraryStats.mockResolvedValue({ total_images: 10 })
  library = Array.from({ length: 10 }, (_, index) => media(index + 1))
  api.fetchImages.mockImplementation(async ({ page = 1, per_page = 50 }) => ({
    images: newestFirst().slice((page - 1) * per_page, page * per_page),
    total: library.length,
  }))
  emit = null
  api.subscribeToLibraryEvents.mockImplementation(callback => {
    emit = callback
    return () => {}
  })
  // JSDOM has no layout. Model each column as a stack of TILE_HEIGHT tiles
  // inside the gallery scroller so scroll anchoring can be observed.
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function rect() {
    if (this.dataset?.imageKey) {
      const column = this.closest('.masonry-column')
      const scroller = this.closest('.masonry-container')
      const row = Array.from(column.children).indexOf(this.parentElement)
      const top = row * TILE_HEIGHT - scroller.scrollTop
      return { top, bottom: top + TILE_HEIGHT, left: 0, right: 100, width: 100, height: TILE_HEIGHT }
    }
    return { top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 }
  })
})
afterEach(() => {
  cleanup(); vi.unstubAllGlobals()
})

async function renderGallery() {
  const view = render(<App />)
  await waitFor(() => expect(view.container.querySelectorAll('.media-item').length).toBe(10))
  expect(emit).toBeTypeOf('function')
  return view
}

async function importAndNotify(ids) {
  library.push(...ids.map(media))
  act(() => emit({ type: 'image_added', data: { image_id: ids[0], directory_id: 1 } }))
  // The refresh is throttled to one request per 750ms.
  await waitFor(() => expect(api.fetchImages.mock.calls.length).toBeGreaterThan(1), { timeout: 3000 })
}

const tile = (container, id) => container.querySelector(`[data-image-key="${key(id)}"]`)

test('a live import appears without a manual refresh even after infinite scroll loaded later pages', async () => {
  library = Array.from({ length: 60 }, (_, index) => media(index + 1))
  const view = render(<App />)
  await waitFor(() => expect(view.container.querySelectorAll('.media-item').length).toBe(50))
  const sentinel = view.container.querySelector('.load-more-trigger')
  const observer = ControlledIntersectionObserver.instances.find(instance => instance.targets.has(sentinel))
  act(() => observer.intersect(sentinel))
  await waitFor(() => expect(view.container.querySelectorAll('.media-item').length).toBe(60))
  expect(api.fetchImages.mock.calls.map(([query]) => query.page)).toEqual([1, 2])

  library.push(media(61))
  act(() => emit({ type: 'image_added', data: { image_id: 61, directory_id: 1 } }))
  await waitFor(() => expect(tile(view.container, 61)).toBeTruthy(), { timeout: 3000 })
  expect(view.container.querySelectorAll('.media-item').length).toBe(61)
  // Everything the user had loaded is still there.
  expect(tile(view.container, 1)).toBeTruthy()
})

test('prepending live imports keeps existing tiles mounted in their columns', async () => {
  const { container } = await renderGallery()
  const existing = new Map([3, 7, 10].map(id => [id, tile(container, id)]))

  await importAndNotify([11, 12, 13])
  await waitFor(() => expect(tile(container, 13)).toBeTruthy())

  for (const [id, element] of existing) {
    expect(tile(container, id)).toBe(element)
  }
  // The new tiles sit on top of their columns.
  const topRow = Array.from(container.querySelectorAll('.masonry-column'))
    .map(column => Number(column.querySelector('.media-item')?.dataset.imageId))
  expect(topRow).toEqual(expect.arrayContaining([11, 12, 13]))
})

test('prepending while scrolled down keeps the tile the user is looking at in place', async () => {
  const { container } = await renderGallery()
  const scroller = container.querySelector('.content > .masonry-container')
  scroller.scrollTop = 100
  const anchorId = (() => {
    // First tile whose bottom is below the scroller top: second row of column 0.
    const first = container.querySelector('.masonry-column')
    return Number(first.children[1].querySelector('.media-item').dataset.imageId)
  })()
  const topBefore = tile(container, anchorId).getBoundingClientRect().top

  await importAndNotify([11, 12])
  await waitFor(() => expect(tile(container, 12)).toBeTruthy())

  expect(tile(container, anchorId).getBoundingClientRect().top).toBe(topBefore)
  expect(scroller.scrollTop).toBeGreaterThan(100)
})

test('a user at the very top sees new tiles appear without being scrolled away', async () => {
  const { container } = await renderGallery()
  const scroller = container.querySelector('.content > .masonry-container')
  scroller.scrollTop = 0
  await importAndNotify([11])
  await waitFor(() => expect(tile(container, 11)).toBeTruthy())
  expect(scroller.scrollTop).toBe(0)
})

test('prepending while the lightbox is open keeps the same image selected', async () => {
  const { container } = await renderGallery()
  fireEvent.click(tile(container, 7))
  const lightbox = await screen.findByTestId('lightbox')
  expect(lightbox.dataset.imageId).toBe('7')
  const indexBefore = Number(lightbox.dataset.index)

  await importAndNotify([11, 12])
  await waitFor(() => expect(tile(container, 12)).toBeTruthy())

  const after = screen.getByTestId('lightbox')
  expect(after.dataset.imageId).toBe('7')
  expect(Number(after.dataset.index)).toBe(indexBefore + 2)
  expect(after.dataset.total).toBe('12')
})

test('prepending keeps selection mode and the selected tiles', async () => {
  const { container } = await renderGallery()
  fireEvent.click(screen.getByTitle('Enter selection mode'))
  fireEvent.click(tile(container, 4))
  fireEvent.click(tile(container, 9))
  expect(screen.getByText('2 selected')).toBeTruthy()

  await importAndNotify([11, 12])
  await waitFor(() => expect(tile(container, 12)).toBeTruthy())

  expect(screen.getByTitle('Exit selection mode')).toBeTruthy()
  expect(screen.getByText('2 selected')).toBeTruthy()
  expect(tile(container, 4).classList.contains('selected')).toBe(true)
  expect(tile(container, 9).classList.contains('selected')).toBe(true)
  expect(tile(container, 12).classList.contains('selected')).toBe(false)
})
