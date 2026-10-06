// SPDX-License-Identifier: GPL-3.0-only
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useNavigate } from 'react-router-dom'
const api = vi.hoisted(() => ({ browseRemoteSource: vi.fn(), getRemoteSources: vi.fn(), fetchDirectories: vi.fn(), importRemoteItem: vi.fn() }))
vi.mock('../api', () => api)
vi.mock('../components/Sidebar', () => ({ default: () => null }))
vi.mock('../components/MasonryGrid', () => ({ default: ({ images, onImageClick, onLoadMore, loading }) => <div className="masonry-container" aria-label="Regular masonry grid">
  {images.map(image => <button key={image.id} onClick={() => onImageClick(image)}>{image.remote_source_id}:{image.remote_post_id}</button>)}
  <button disabled={loading} onClick={onLoadMore}>Load next page</button>
</div> }))
vi.mock('../components/Lightbox', () => ({ default: ({ images, currentIndex, onNav, onClose, onTagClick, remoteActions }) => <div role="dialog" aria-label="Regular lightbox">
  <span>{images[currentIndex].remote_post_id}</span><button onClick={() => onNav(1)}>Next image</button><button onClick={() => onNav(-1)}>Previous image</button>
  <button onClick={onClose}>Close viewer</button><button onClick={() => onTagClick('viewer-tag')}>Search this tag</button>{remoteActions}
</div> }))
import SharedCollectionPage from './SharedCollectionPage'
let navigate
function Driver() { navigate = useNavigate(); return null }
const item = (id, source = 'one') => ({ source_id: source, remote_post_id: String(id), tags: [], media: [{ kind: 'original', mime_type: 'image/jpeg', width: 800, height: 600 }] })
const page = (items, pageNumber = 1, total = items.length) => ({ items, page: pageNumber, per_page: 60, total, stale: false })
const deferred = () => { let resolve; let reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no }); return { promise, resolve, reject } }
function mount(url = '/shared/one?media_type=image') {
  return render(<MemoryRouter initialEntries={[url]}><Driver /><Routes><Route path="/shared/:sourceId" element={<SharedCollectionPage />} /></Routes></MemoryRouter>)
}
beforeEach(() => {
  vi.resetAllMocks(); localStorage.clear(); sessionStorage.clear()
  api.getRemoteSources.mockResolvedValue({ sources: [{ source_id: 'one', display_name: 'Community One' }, { source_id: 'two', display_name: 'Community Two' }] })
  api.fetchDirectories.mockResolvedValue({ directories: [{ id: 9, name: 'Local pictures', enabled: true }] })
  api.importRemoteItem.mockResolvedValue({ image_id: 99 })
})
afterEach(cleanup)
test('appends overlapping pages and opens the shared lightbox with normal navigation', async () => {
  api.browseRemoteSource.mockResolvedValueOnce(page([item(1), item(2)], 1, 61)).mockResolvedValueOnce(page([item(2), item(3)], 2, 61))
  mount()
  await screen.findByRole('button', { name: 'one:1' })
  fireEvent.click(screen.getByRole('button', { name: 'Load next page' }))
  await screen.findByRole('button', { name: 'one:3' })
  expect(screen.getAllByRole('button', { name: 'one:2' })).toHaveLength(1)
  expect(api.browseRemoteSource).toHaveBeenLastCalledWith('one', expect.objectContaining({ page: 2, per_page: 60, media_type: 'image', q: '' }))
  fireEvent.click(screen.getByRole('button', { name: 'one:2' }))
  expect(screen.getByRole('dialog').textContent).toContain('2')
  fireEvent.click(screen.getByRole('button', { name: 'Next image' }))
  expect(screen.getByRole('dialog').textContent).toContain('3')
  fireEvent.click(screen.getByRole('button', { name: 'Close viewer' }))
  expect(screen.queryByRole('dialog')).toBeNull()
  expect(api.importRemoteItem).not.toHaveBeenCalled()
})
test('a late search response cannot replace the current submitted search', async () => {
  const old = deferred()
  api.browseRemoteSource.mockReturnValueOnce(old.promise).mockResolvedValueOnce(page([item('fresh')]))
  mount()
  fireEvent.change(screen.getByRole('textbox', { name: 'Search community tags' }), { target: { value: 'new-tag' } })
  fireEvent.click(screen.getByRole('button', { name: 'Search', exact: true }))
  await screen.findByRole('button', { name: 'one:fresh' })
  await act(async () => old.resolve(page([item('old')])))
  expect(screen.queryByRole('button', { name: 'one:old' })).toBeNull()
  expect(api.browseRemoteSource).toHaveBeenLastCalledWith('one', expect.objectContaining({ q: 'new-tag', page: 1 }))
})
test('changing source discards late errors from the previous source', async () => {
  const old = deferred()
  api.browseRemoteSource.mockReturnValueOnce(old.promise).mockResolvedValueOnce(page([item('new', 'two')]))
  mount()
  await act(async () => navigate('/shared/two?media_type=image'))
  await screen.findByRole('button', { name: 'two:new' })
  await act(async () => old.reject(new Error('Old source failed')))
  expect(screen.queryByText('Old source failed')).toBeNull()
  expect(screen.queryByRole('button', { name: 'one:new' })).toBeNull()
})
test('import uses the source identity and explicitly chosen local directory', async () => {
  api.browseRemoteSource.mockResolvedValue(page([item('42')]))
  mount()
  fireEvent.click(await screen.findByRole('button', { name: 'one:42' }))
  await screen.findByRole('option', { name: 'Local pictures' })
  expect(api.importRemoteItem).not.toHaveBeenCalled()
  fireEvent.change(screen.getByRole('combobox', { name: 'Import destination' }), { target: { value: '9' } })
  fireEvent.click(screen.getByRole('button', { name: 'Import a local copy' }))
  await waitFor(() => expect(api.importRemoteItem).toHaveBeenCalledWith('one', '42', { kind: 'original', directory_id: 9 }))
  await screen.findByText('Saved a local copy as image #99.')
})

test('changing Images to Videos discards a late image catalog response', async () => {
  const old = deferred()
  const video = item('video')
  video.media[0].mime_type = 'video/mp4'
  api.browseRemoteSource.mockReturnValueOnce(old.promise).mockResolvedValueOnce(page([video]))
  mount()
  await act(async () => navigate('/shared/one?media_type=video'))
  await screen.findByRole('button', { name: 'one:video' })
  await act(async () => old.resolve(page([item('image')])))
  expect(screen.queryByRole('button', { name: 'one:image' })).toBeNull()
  expect(api.browseRemoteSource).toHaveBeenLastCalledWith('one', expect.objectContaining({ media_type: 'video' }))
})

test('an empty filtered page continues to the next catalog page', async () => {
  api.browseRemoteSource.mockResolvedValueOnce(page([], 1, 61)).mockResolvedValueOnce(page([item('later')], 2, 61))
  mount()
  await screen.findByRole('button', { name: 'one:later' })
  expect(api.browseRemoteSource).toHaveBeenLastCalledWith('one', expect.objectContaining({ page: 2, media_type: 'image' }))
})

test('tile size shares the local gallery preference and viewer close preserves scroll position', async () => {
  localStorage.setItem('localbooru_tileSize', '4')
  api.browseRemoteSource.mockResolvedValue(page([item('42')]))
  mount()
  await screen.findByRole('button', { name: 'one:42' })
  expect(screen.getByRole('slider', { name: 'Tile size' }).value).toBe('4')
  fireEvent.change(screen.getByRole('slider', { name: 'Tile size' }), { target: { value: '2' } })
  expect(localStorage.getItem('localbooru_tileSize')).toBe('2')
  const content = document.querySelector('.masonry-container')
  content.scrollTop = 500
  fireEvent.scroll(content)
  const savedScroll = Object.entries(sessionStorage).find(([key]) => key.endsWith('_scroll'))
  expect(savedScroll?.[1]).toBe('500')
  fireEvent.click(screen.getByRole('button', { name: 'one:42' }))
  content.scrollTop = 0
  fireEvent.click(screen.getByRole('button', { name: 'Close viewer' }))
  expect(content.scrollTop).toBe(500)
})

test('saved shared-gallery pages and scroll restore when returning to a community', async () => {
  const sessionKey = `donutMediaCenter_shared_${encodeURIComponent(JSON.stringify(['one', 'image', '']))}`
  sessionStorage.setItem(`${sessionKey}_page`, '2')
  sessionStorage.setItem(`${sessionKey}_scroll`, '350')
  api.browseRemoteSource.mockResolvedValueOnce(page([item(1)], 1, 61)).mockResolvedValueOnce(page([item(2)], 2, 61))
  mount()
  await screen.findByRole('button', { name: 'one:2' })
  expect(screen.getByRole('button', { name: 'one:1' })).toBeTruthy()
  await waitFor(() => expect(document.querySelector('.masonry-container').scrollTop).toBe(350))
})

test('submitting the same search refreshes the first community page', async () => {
  api.browseRemoteSource.mockResolvedValueOnce(page([item('before')])).mockResolvedValueOnce(page([item('after')]))
  mount()
  await screen.findByRole('button', { name: 'one:before' })
  fireEvent.click(screen.getByRole('button', { name: 'Search', exact: true }))
  await screen.findByRole('button', { name: 'one:after' })
  expect(screen.queryByRole('button', { name: 'one:before' })).toBeNull()
  expect(api.browseRemoteSource).toHaveBeenCalledTimes(2)
})

test('pagination remains locked until saved pages finish restoring', async () => {
  const sessionKey = `donutMediaCenter_shared_${encodeURIComponent(JSON.stringify(['one', 'image', '']))}`
  sessionStorage.setItem(`${sessionKey}_page`, '2')
  sessionStorage.setItem(`${sessionKey}_scroll`, '450')
  const second = deferred()
  api.browseRemoteSource.mockResolvedValueOnce(page([item(1)], 1, 121)).mockReturnValueOnce(second.promise)
  mount()
  await screen.findByRole('button', { name: 'one:1' })
  expect(screen.getByRole('button', { name: 'Load next page' }).disabled).toBe(true)
  await act(async () => second.resolve(page([item(2)], 2, 121)))
  await screen.findByRole('button', { name: 'one:2' })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Load next page' }).disabled).toBe(false))
  expect(api.browseRemoteSource).toHaveBeenCalledTimes(2)
  await waitFor(() => expect(document.querySelector('.masonry-container').scrollTop).toBe(450))
})

test('a late import receipt cannot appear on the next remote item', async () => {
  const importing = deferred()
  api.importRemoteItem.mockReturnValueOnce(importing.promise)
  api.browseRemoteSource.mockResolvedValue(page([item('42'), item('43')]))
  mount()
  fireEvent.click(await screen.findByRole('button', { name: 'one:42' }))
  await screen.findByRole('option', { name: 'Local pictures' })
  fireEvent.change(screen.getByRole('combobox', { name: 'Import destination' }), { target: { value: '9' } })
  fireEvent.click(screen.getByRole('button', { name: 'Import a local copy' }))
  fireEvent.click(screen.getByRole('button', { name: 'Next image' }))
  await act(async () => importing.resolve({ image_id: 99 }))
  expect(screen.queryByText('Saved a local copy as image #99.')).toBeNull()
  expect(screen.getByRole('dialog').textContent).toContain('43')
})
