import { useCallback, useMemo, useState } from 'react'
import {
  discardForCuration,
  fetchImages,
  restoreCurationDiscard,
  setFavorite,
  unfavoriteCurationItems,
} from '../api'
import { toast } from '../components/Toast'
import { buildCurationQuery, mergeCandidates, seedCandidates } from '../utils/curationState'
import {
  getCurationProgress,
  loadCurationGoal,
  recordCurated,
  undoRecordedCurated,
} from '../utils/curationProgress'

const initialState = {
  active: false,
  query: null,
  queue: [],
  lastAction: null,
  processed: 0,
  loading: false,
  busy: false,
  complete: false,
  matchingFavoriteCount: 0,
}

export function useCurationGame({ loadedImages, filters, onGalleryRefresh }) {
  const [state, setState] = useState(initialState)
  const [progressVersion, setProgressVersion] = useState(0)
  const goal = useMemo(() => loadCurationGoal(), [progressVersion, state.active])
  const progress = useMemo(() => getCurationProgress(goal), [goal, progressVersion])

  const finishIfEmpty = useCallback(async (query, queue, lastAction = null) => {
    if (queue.length > 0) {
      setState(previous => ({ ...previous, queue, loading: false }))
      return
    }
    const result = await fetchImages({
      ...query,
      favorites_only: true,
      exclude_favorites: false,
      page: 1,
      per_page: 1,
    })
    setState(previous => ({
      ...previous,
      queue: [],
      loading: false,
      complete: true,
      lastAction: lastAction ?? previous.lastAction,
      matchingFavoriteCount: result.total || 0,
    }))
  }, [])

  const refill = useCallback(async (query, queue, lastAction = null) => {
    const result = await fetchImages({ ...query, page: 1 })
    const merged = mergeCandidates(queue, result.images || [])
    await finishIfEmpty(query, merged, lastAction)
  }, [finishIfEmpty])

  const start = useCallback(async (overrideQuery = null, overrideSeed = null) => {
    const query = buildCurationQuery(overrideQuery || filters)
    const seed = seedCandidates(overrideSeed || loadedImages)
    setState({ ...initialState, active: true, query, queue: seed, loading: true })
    try {
      await refill(query, seed)
    } catch (error) {
      toast.error(`Could not start Curation Game: ${error.message}`)
      setState(initialState)
    }
  }, [filters, loadedImages, refill])

  const applyAction = useCallback(async (kind) => {
    const item = state.queue[0]
    if (!item || state.busy) return
    setState(previous => ({ ...previous, busy: true }))
    try {
      if (kind === 'keep') await setFavorite(item, true)
      else await discardForCuration(item, localStorage.getItem('localbooru_dumpster_path'))
      const countedDate = recordCurated()
      const queue = state.queue.slice(1)
      const lastAction = { kind, item, countedDate }
      setProgressVersion(value => value + 1)
      setState(previous => ({
        ...previous,
        queue,
        lastAction,
        processed: previous.processed + 1,
        busy: false,
        complete: false,
      }))
      await refill(state.query, queue, lastAction)
    } catch (error) {
      toast.error(`${kind === 'keep' ? 'Keep' : 'Discard'} failed: ${error.message}`)
      setState(previous => ({ ...previous, busy: false }))
    }
  }, [state.busy, state.queue, state.query, refill])

  const undo = useCallback(async () => {
    if (!state.lastAction || state.busy) return
    const action = state.lastAction
    setState(previous => ({ ...previous, busy: true }))
    try {
      if (action.kind === 'keep') await setFavorite(action.item, false)
      else await restoreCurationDiscard(action.item)
      undoRecordedCurated(action.countedDate)
      setProgressVersion(value => value + 1)
      setState(previous => ({
        ...previous,
        queue: [action.item, ...previous.queue],
        lastAction: null,
        processed: Math.max(0, previous.processed - 1),
        busy: false,
        complete: false,
      }))
    } catch (error) {
      toast.error(`Undo failed: ${error.message}`)
      setState(previous => ({ ...previous, busy: false }))
    }
  }, [state.lastAction, state.busy])

  const unfavoriteAllAndRestart = useCallback(async () => {
    if (!state.query || state.busy) return
    setState(previous => ({ ...previous, busy: true }))
    try {
      while (true) {
        const result = await fetchImages({
          ...state.query,
          favorites_only: true,
          exclude_favorites: false,
          page: 1,
          per_page: 400,
        })
        if (!result.images?.length) break
        await unfavoriteCurationItems(result.images)
      }
      await start(state.query, [])
    } catch (error) {
      toast.error(`Could not restart: ${error.message}`)
      setState(previous => ({ ...previous, busy: false }))
    }
  }, [state.query, state.busy, start])

  const exit = useCallback(() => {
    setState(initialState)
    onGalleryRefresh?.()
  }, [onGalleryRefresh])

  return {
    ...state,
    current: state.queue[0] || null,
    goal,
    progress,
    start,
    keep: () => applyAction('keep'),
    discard: () => applyAction('discard'),
    undo,
    exit,
    unfavoriteAllAndRestart,
  }
}
