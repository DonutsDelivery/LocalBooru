// SPDX-License-Identifier: GPL-3.0-only
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
const api = vi.hoisted(() => ({
  getContinueWatching: vi.fn(),
  clearWatchHistory: vi.fn(),
  fetchImage: vi.fn(),
  getMediaUrl: path => path,
  getFamilyModeStatus: vi.fn(),
  subscribeToLibraryEvents: vi.fn(),
}))
vi.mock('../api', () => api)
import ContinueWatching from './ContinueWatching'

let emitLibraryEvent
function watchItem(imageId, filename) {
  return {
    image_id: imageId,
    directory_id: 3,
    library_id: 'lib-1',
    playback_position: 30,
    duration: 100,
    progress: 0.3,
    completed: false,
    last_watched: '2026-10-19T00:00:00Z',
    original_filename: filename,
  }
}

beforeEach(() => {
  vi.resetAllMocks()
  api.subscribeToLibraryEvents.mockImplementation(handler => {
    emitLibraryEvent = handler
    return () => { emitLibraryEvent = null }
  })
  api.getFamilyModeStatus.mockResolvedValue({ enabled: true, is_locked: false })
  api.getContinueWatching.mockResolvedValue({ total: 1, items: [watchItem(11, 'family-safe.mp4')] })
  api.fetchImage.mockResolvedValue({ id: 11, library_id: 'lib-1', directory_id: 3, original_filename: 'family-safe.mp4', thumbnail_url: '/thumb/11' })
})
afterEach(cleanup)

// AC: @family-mode the continue watching row refreshes when the family lock flips
test('continue watching reloads the server-filtered list when family mode locks or unlocks', async () => {
  render(<ContinueWatching onImageClick={() => {}} />)
  await waitFor(() => expect(screen.getByText(/family-safe\.mp4/)).toBeTruthy())
  expect(api.getContinueWatching).toHaveBeenCalledTimes(1)

  api.getContinueWatching.mockResolvedValue({ total: 0, items: [] })
  emitLibraryEvent({ type: 'family_mode', data: { is_locked: true } })
  await waitFor(() => expect(api.getContinueWatching).toHaveBeenCalledTimes(2))
  await waitFor(() => expect(screen.queryByText(/family-safe\.mp4/)).toBeNull())

  emitLibraryEvent({ type: 'family_mode', is_locked: false })
  await waitFor(() => expect(api.getContinueWatching).toHaveBeenCalledTimes(3))
})

// AC: @family-mode family mode starts locked so the first load already reflects it
test('initial load waits for the lock state and loads the filtered list once known', async () => {
  api.getFamilyModeStatus.mockResolvedValue({ enabled: true, is_locked: true })
  render(<ContinueWatching onImageClick={() => {}} />)
  await waitFor(() => expect(api.getFamilyModeStatus).toHaveBeenCalled())
  await waitFor(() => expect(api.getContinueWatching).toHaveBeenCalledTimes(1))
  await waitFor(() => expect(screen.getByText(/family-safe\.mp4/)).toBeTruthy())
})

// AC: @family-mode panels stay usable when family mode is not configured
test('status failure still loads the continue watching list', async () => {
  api.getFamilyModeStatus.mockRejectedValue(new Error('not configured'))
  render(<ContinueWatching onImageClick={() => {}} />)
  await waitFor(() => expect(screen.getByText(/family-safe\.mp4/)).toBeTruthy())
})

function deferred() {
  let resolve
  const promise = new Promise(r => { resolve = r })
  return { promise, resolve }
}

test('lock clears cards immediately and ignores an older pending list', async () => {
  const old = deferred()
  const locked = deferred()
  api.getContinueWatching.mockResolvedValueOnce({ items: [watchItem(11, 'visible.mp4')] })
    .mockReturnValueOnce(old.promise).mockReturnValueOnce(locked.promise)
  render(<ContinueWatching onImageClick={() => {}} />)
  await screen.findByText('family-safe.mp4')
  act(() => emitLibraryEvent({ type: 'family_mode', is_locked: false }))
  await waitFor(() => expect(api.getContinueWatching).toHaveBeenCalledTimes(2))
  act(() => emitLibraryEvent({ type: 'family_mode', is_locked: true }))
  expect(screen.queryByText('family-safe.mp4')).toBeNull()
  await waitFor(() => expect(api.getContinueWatching).toHaveBeenCalledTimes(3))
  await act(async () => { old.resolve({ items: [watchItem(11, 'old.mp4')] }); await old.promise })
  expect(screen.queryByText('family-safe.mp4')).toBeNull()
  await act(async () => { locked.resolve({ items: [] }); await locked.promise })
  expect(screen.queryByText('family-safe.mp4')).toBeNull()
})

test('superseded hydration cannot restore a card after locking', async () => {
  const image = deferred()
  api.fetchImage.mockReturnValueOnce(image.promise)
  api.getContinueWatching.mockResolvedValueOnce({ items: [watchItem(11, 'old.mp4')] })
    .mockResolvedValueOnce({ items: [] })
  render(<ContinueWatching onImageClick={() => {}} />)
  await waitFor(() => expect(api.fetchImage).toHaveBeenCalled())
  act(() => emitLibraryEvent({ type: 'family_mode', is_locked: true }))
  await waitFor(() => expect(api.getContinueWatching).toHaveBeenCalledTimes(2))
  await act(async () => { image.resolve({ original_filename: 'old.mp4' }); await image.promise })
  expect(screen.queryByText('old.mp4')).toBeNull()
})

test('a lock event supersedes the initial status response', async () => {
  const status = deferred()
  api.getFamilyModeStatus.mockReturnValue(status.promise)
  render(<ContinueWatching onImageClick={() => {}} />)
  act(() => emitLibraryEvent({ type: 'family_mode', is_locked: true }))
  await waitFor(() => expect(api.getContinueWatching).toHaveBeenCalledTimes(1))
  await act(async () => { status.resolve({ is_locked: false }); await status.promise })
  expect(api.getContinueWatching).toHaveBeenCalledTimes(1)
})

test('unmounted pending loads do not hydrate or publish', async () => {
  const list = deferred()
  api.getContinueWatching.mockReturnValue(list.promise)
  const { unmount } = render(<ContinueWatching onImageClick={() => {}} />)
  await waitFor(() => expect(api.getContinueWatching).toHaveBeenCalled())
  unmount()
  await act(async () => { list.resolve({ items: [watchItem(11, 'old.mp4')] }); await list.promise })
  expect(api.fetchImage).not.toHaveBeenCalled()
})
