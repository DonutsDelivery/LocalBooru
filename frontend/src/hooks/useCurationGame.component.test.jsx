import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { act, cleanup, renderHook } from '@testing-library/react'
import { useCurationGame } from './useCurationGame'

const api = vi.hoisted(() => ({
  fetchImages: vi.fn(), setFavorite: vi.fn(), discardForCuration: vi.fn(),
  restoreCurationDiscard: vi.fn(), unfavoriteCurationItems: vi.fn(),
}))
vi.mock('../api', () => api)
vi.mock('../components/Toast', () => ({ toast: { error: vi.fn() } }))

let candidates
const items = [1, 2, 3].map(id => ({ id, directory_id: 1, library_id: 'synthetic', is_favorite: false }))
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  candidates = [...items]
  api.fetchImages.mockImplementation(async query => query.favorites_only
    ? { images: [], total: 0 } : { images: [...candidates], total: candidates.length })
  const remove = async item => { candidates = candidates.filter(other => other.id !== item.id) }
  api.setFavorite.mockImplementation(async (item, favorite) => {
    if (favorite) await remove(item)
    else candidates.push(item)
  })
  api.discardForCuration.mockImplementation(remove)
  api.restoreCurationDiscard.mockImplementation(async item => { candidates.push(item) })
})
afterEach(cleanup)

async function start() {
  const view = renderHook(() => useCurationGame({ loadedImages: items, filters: { media_type: 'image' } }))
  await act(async () => { await view.result.current.start() })
  return view.result
}

// AC: @curation-game ac-3
// AC: @curation-game ac-5
// AC: @curation-game ac-8
test('recovered Favorite/discard history reverses multiple decisions and their counts', async () => {
  const result = await start()
  await act(async () => { await result.current.keep() })
  await act(async () => { await result.current.discard() })
  expect(result.current.current.id).toBe(3)
  expect(result.current.progress).toBe(2)
  expect(result.current.actionHistory.map(action => action.kind)).toEqual(['keep', 'discard'])
  await act(async () => { await result.current.undo() })
  expect(result.current.current.id).toBe(2)
  expect(api.restoreCurationDiscard).toHaveBeenCalledWith(items[1])
  await act(async () => { await result.current.undo() })
  expect(result.current.queue.map(item => item.id)).toEqual([1, 2, 3])
  expect(api.setFavorite).toHaveBeenLastCalledWith(items[0], false)
  expect(result.current.lastAction).toBeNull()
  expect(result.current.actionHistory).toEqual([])
  expect(result.current.processed).toBe(0)
  expect(result.current.progress).toBe(0)
})

// AC: @curation-game ac-9
test('recovered candidate navigation rotates all candidates without committing a decision', async () => {
  const result = await start()
  act(() => result.current.navigate(1))
  expect(result.current.current.id).toBe(2)
  act(() => result.current.navigate(-1))
  expect(result.current.queue.map(item => item.id)).toEqual([1, 2, 3])
  expect(api.setFavorite).not.toHaveBeenCalled()
  expect(api.discardForCuration).not.toHaveBeenCalled()
  expect(result.current.processed).toBe(0)
})

// AC: @curation-gesture-decisions-recovery ac-3
test('an in-flight decision admits no duplicate decisions or undo', async () => {
  const result = await start()
  let release
  api.discardForCuration.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
  let first
  act(() => { first = result.current.discard() })
  await act(async () => { await result.current.discard(); await result.current.keep(); await result.current.undo() })
  expect(api.discardForCuration).toHaveBeenCalledOnce()
  expect(api.setFavorite).not.toHaveBeenCalled()
  candidates = items.slice(1)
  await act(async () => { release(); await first })
  expect(result.current.processed).toBe(1)
})

// AC: @curation-game ac-5
test('a failed restore keeps the action available for retry', async () => {
  const result = await start()
  await act(async () => { await result.current.discard() })
  api.restoreCurationDiscard.mockRejectedValueOnce(new Error('synthetic collision'))
  await act(async () => { await result.current.undo() })
  expect(result.current.current.id).toBe(2)
  expect(result.current.actionHistory).toHaveLength(1)
  await act(async () => { await result.current.undo() })
  expect(result.current.current.id).toBe(1)
  expect(result.current.progress).toBe(0)
})

// AC: @curation-gesture-decisions-recovery ac-4
// AC: @curation-gesture-decisions-recovery ac-5
test('failed refill preserves committed history; retry never repeats the mutation', async () => {
  const result = await start()
  api.fetchImages.mockRejectedValueOnce(new Error('synthetic unavailable'))
  await act(async () => { await result.current.discard() })
  expect(result.current.refillError).toBe('synthetic unavailable')
  expect(result.current.actionHistory).toHaveLength(1)
  expect(result.current.busy).toBe(false)
  await act(async () => { await result.current.retryRefill() })
  expect(api.discardForCuration).toHaveBeenCalledOnce()
  expect(result.current.refillError).toBeNull()
  await act(async () => { await result.current.undo() })
  expect(result.current.current.id).toBe(1)
})


// AC: @curation-game ac-7
test('empty runs complete and can unfavorite matching items to restart', async () => {
  candidates = []
  api.fetchImages.mockImplementation(async query => query.favorites_only
    ? { images: candidates.length ? [] : items, total: 3 }
    : { images: candidates, total: candidates.length })
  const view = renderHook(() => useCurationGame({ loadedImages: [], filters: {} }))
  await act(async () => { await view.result.current.start() })
  expect(view.result.current.complete).toBe(true)
  expect(view.result.current.matchingFavoriteCount).toBe(3)
  api.unfavoriteCurationItems.mockImplementation(async batch => { candidates = [...batch] })
  await act(async () => { await view.result.current.unfavoriteAllAndRestart() })
  expect(api.unfavoriteCurationItems).toHaveBeenCalledWith(items)
  expect(view.result.current.complete).toBe(false)
  expect(view.result.current.current.id).toBe(1)
  expect(view.result.current.actionHistory).toEqual([])
})

// AC: @curation-game ac-10
test('inactive game does not fetch or mutate the ordinary gallery', () => {
  const view = renderHook(() => useCurationGame({ loadedImages: items, filters: {} }))
  expect(view.result.current.active).toBe(false)
  expect(api.fetchImages).not.toHaveBeenCalled()
  expect(api.setFavorite).not.toHaveBeenCalled()
  expect(api.discardForCuration).not.toHaveBeenCalled()
})
