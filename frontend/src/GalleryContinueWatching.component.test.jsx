import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

const api = vi.hoisted(() => ({
  fetchImages: vi.fn(), fetchImage: vi.fn(), fetchTags: vi.fn(), getLibraryStats: vi.fn(),
  getContinueWatching: vi.fn(), clearWatchHistory: vi.fn(),
  subscribeToLibraryEvents: vi.fn(() => () => {}), healthCheck: vi.fn(),
}))
vi.mock('./api', async importOriginal => ({ ...await importOriginal(), ...api }))
vi.mock('./components/Sidebar', () => ({ default: ({ selectedImage }) => <aside>{selectedImage?.filename || 'Video filters'}</aside> }))
vi.mock('./components/TitleBar', () => ({ default: () => null }))
vi.mock('./components/Music/MusicExperience', () => ({
  MusicPlayerProvider: ({ children }) => children, MusicPage: () => null, PersistentMusicPlayer: () => null,
}))
vi.mock('./hooks/useAddonStatus', () => ({ useAllAddonStatuses: () => ({ isInstalled: () => false }) }))
vi.mock('./components/MasonryGrid', () => ({ default: ({ images }) => <div data-testid="gallery">{images.map(image => <span key={image.directory_id}>{image.filename}</span>)}</div> }))
vi.mock('./components/Lightbox', () => ({ default: ({ images, currentIndex, total, onNav, onClose, onImageUpdate, onDelete }) => {
  const image = images[currentIndex]
  if (!image) return null
  const locator = { imageId: image.id, directoryId: image.directory_id, libraryId: image.library_id }
  return <section aria-label="Video player">
    <div data-testid="playing-video">{image.library_id}:{image.directory_id}:{image.id}:{image.filename}</div>
    <div data-testid="player-count">{currentIndex + 1}/{total}</div>
    <div data-testid="player-favorite">{String(image.is_favorite)}</div>
    <button onClick={() => onNav(1)}>Next video</button>
    <button onClick={() => onImageUpdate(locator, { is_favorite: true })}>Favorite video</button>
    <button onClick={() => onDelete(locator)}>Delete video</button>
    <button onClick={onClose}>Close video</button>
  </section>
} }))

import App from './App'

const loaded = { id: 12, directory_id: 1, library_id: 'library-a', filename: 'loaded.mp4' }
const historyVideo = { id: 12, directory_id: 2, library_id: 'library-b', filename: 'resume.mp4', thumbnail_url: '/synthetic.jpg', is_favorite: false }

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  window.history.replaceState({}, '', '/videos?tags=landscape')
  vi.clearAllMocks()
  api.healthCheck.mockResolvedValue({ status: 'ok' })
  api.fetchTags.mockResolvedValue({ tags: [] })
  api.getLibraryStats.mockResolvedValue({ total_images: 1 })
  api.fetchImages.mockResolvedValue({ images: [loaded], total: 1 })
  api.fetchImage.mockResolvedValue(historyVideo)
  api.getContinueWatching.mockResolvedValue({ items: [{ image_id: 12, directory_id: 2, library_id: 'library-b', playback_position: 40, duration: 120, progress: 1 / 3 }] })
  api.clearWatchHistory.mockResolvedValue({})
})
afterEach(cleanup)

test('opens a history video outside the loaded gallery with exact identity and preserves gallery scope', async () => {
  render(<App />)
  fireEvent.click(await screen.findByText('resume.mp4'))
  expect(await screen.findByTestId('playing-video')).toHaveProperty('textContent', 'library-b:2:12:resume.mp4')
  expect(screen.getByTestId('player-count').textContent).toBe('1/1')
  expect(screen.getByTestId('gallery').textContent).toBe('loaded.mp4')
  expect(window.location.search).toBe('?tags=landscape')
  expect(api.fetchImage).toHaveBeenCalledWith(12, { directoryId: 2, libraryId: 'library-b', optional: true })
  fireEvent.click(screen.getByRole('button', { name: 'Next video' }))
  expect(screen.getByTestId('playing-video').textContent).toBe('library-b:2:12:resume.mp4')
  fireEvent.click(screen.getByRole('button', { name: 'Favorite video' }))
  expect(screen.getByTestId('player-favorite').textContent).toBe('true')
  fireEvent.click(screen.getByRole('button', { name: 'Close video' }))
  await waitFor(() => expect(screen.queryByRole('region', { name: 'Video player' })).toBeNull())
  expect(screen.getByTestId('gallery').textContent).toBe('loaded.mp4')
})

test('deleting a standalone history video closes it without opening a different gallery video', async () => {
  render(<App />)
  fireEvent.click(await screen.findByText('resume.mp4'))
  await screen.findByTestId('playing-video')
  fireEvent.click(screen.getByRole('button', { name: 'Delete video' }))
  await waitFor(() => expect(screen.queryByRole('region', { name: 'Video player' })).toBeNull())
  expect(screen.getByTestId('gallery').textContent).toBe('loaded.mp4')
})

test('an authoritative scan refresh updates the standalone player without adding it to the filtered gallery', async () => {
  render(<App />)
  fireEvent.click(await screen.findByText('resume.mp4'))
  await screen.findByTestId('playing-video')
  api.fetchImage.mockResolvedValue({ ...historyVideo, is_favorite: true })
  const onLibraryEvent = api.subscribeToLibraryEvents.mock.calls.at(-1)[0]
  onLibraryEvent({ type: 'task_completed', data: { task_type: 'scan_directory' } })
  await waitFor(() => expect(screen.getByTestId('player-favorite').textContent).toBe('true'), { timeout: 2500 })
  expect(screen.getByTestId('gallery').textContent).toBe('loaded.mp4')
  expect(screen.getByTestId('player-count').textContent).toBe('1/1')
  fireEvent.click(screen.getByRole('button', { name: 'Next video' }))
  expect(screen.getByTestId('playing-video').textContent).toBe('library-b:2:12:resume.mp4')
})

test('Clear All remains in the Continue Watching heading and clears history without selecting media', async () => {
  const { container } = render(<App />)
  await screen.findByText('resume.mp4')
  const clear = screen.getByRole('button', { name: 'Clear All' })
  expect(clear.closest('.continue-watching-header')).toBeTruthy()
  expect(clear.closest('.floating-controls')).toBeNull()
  fireEvent.click(clear)
  await waitFor(() => expect(container.querySelector('.continue-watching')).toBeNull())
  expect(api.clearWatchHistory).toHaveBeenCalledWith()
  expect(screen.queryByRole('region', { name: 'Video player' })).toBeNull()
})
