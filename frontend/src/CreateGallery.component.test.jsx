import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const api = vi.hoisted(() => ({
  fetchImages: vi.fn(), fetchTags: vi.fn(), getLibraryStats: vi.fn(),
  subscribeToLibraryEvents: vi.fn(() => () => {}), healthCheck: vi.fn(),
}))
vi.mock('./api', async importOriginal => ({ ...await importOriginal(), ...api }))
vi.mock('./components/Sidebar', () => ({ default: () => <aside>Image filters</aside> }))
vi.mock('./components/TitleBar', () => ({ default: () => null }))
vi.mock('./components/Music/MusicExperience', () => ({
  MusicPlayerProvider: ({ children }) => children,
  MusicPage: () => null,
  PersistentMusicPlayer: () => null,
}))
vi.mock('./hooks/useAddonStatus', () => ({ useAllAddonStatuses: () => ({ isInstalled: () => false }) }))
vi.mock('./components/MasonryGrid', () => ({ default: ({ images, onLoadMore, hasMore, loading }) => (
  <div className="masonry-container">
    {images.map(image => <div key={image.id} data-image-id={image.id}>{image.filename}</div>)}
    <button type="button" disabled={loading || !hasMore} onClick={onLoadMore}>Load next images</button>
  </div>
) }))

import App from './App'

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  window.history.replaceState({}, '', '/?tags=landscape&library=library-b&directory=1')
  vi.clearAllMocks()
  api.fetchTags.mockResolvedValue({ tags: [] })
  api.getLibraryStats.mockResolvedValue({ total_images: 3 })
  api.healthCheck.mockResolvedValue({ status: 'ok' })
})

afterEach(() => { cleanup() })

// AC: @donut-create-plugin ac-save-gallery
// AC: @donut-create-plugin ac-image-entry
test('refreshes every loaded gallery page after a creation save while retaining filters and scroll', async () => {
  const original = { id: 1, directory_id: 1, library_id: 'library-b', filename: 'original.png' }
  const second = { ...original, id: 2, filename: 'second.png' }
  const generated = { ...original, id: 42, filename: 'generated.png' }
  let imported = false
  api.fetchImages.mockImplementation(async query => ({
    images: query.page === 2 ? [second] : imported ? [generated, original] : [original],
    total: imported ? 3 : 2,
  }))
  const { container } = render(<App />)
  await screen.findByText('original.png')
  const loadNext = screen.getByRole('button', { name: 'Load next images' })
  await waitFor(() => expect(loadNext.disabled).toBe(false))
  fireEvent.click(loadNext)
  await screen.findByText('second.png')
  const gallery = container.querySelector('.masonry-container')
  gallery.scrollTop = 480
  const queryBeforeImport = window.location.search
  imported = true
  act(() => {
    window.dispatchEvent(new CustomEvent('donut-create-imported', {
      detail: { library_id: 'library-b', directory_id: 1, image_id: 42 },
    }))
  })
  await screen.findByText('generated.png')
  expect(screen.getByText('second.png')).toBeTruthy()
  expect(gallery.scrollTop).toBe(480)
  expect(window.location.search).toBe(queryBeforeImport)
  expect(api.fetchImages.mock.calls.map(([query]) => query.page)).toEqual([1, 2, 1, 2])
  expect(api.fetchImages.mock.calls.every(([query]) => query.media_type === 'image'
    && query.tags === 'landscape' && query.library_id === 'library-b' && query.directory_id === 1)).toBe(true)
})
