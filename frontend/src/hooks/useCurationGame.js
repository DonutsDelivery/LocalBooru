import { useCallback, useMemo, useRef, useState } from 'react'
import {
  discardForCuration,
  fetchImages,
  restoreCurationDiscard,
  setFavorite,
  unfavoriteCurationItems,
} from '../api'
import { toast } from '../components/Toast'
import {
  buildCurationQuery,
  commitCurationAction,
  createCurationActionLock,
  markCurationRefillFailure,
  mergeCandidates,
  seedCandidates,
} from '../utils/curationState'
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
  refillError: null,
  matchingFavoriteCount: 0,
}

export function useCurationGame({ loadedImages, filters, onGalleryRefresh }) {
  const [state, setState] = useState(initialState)
  const stateRef = useRef(state)
  const actionLock = useRef(createCurationActionLock())
  stateRef.current = state
  const [progressVersion, setProgressVersion] = useState(0)
  const goal = useMemo(() => loadCurationGoal(), [progressVersion, state.active])
  const progress = useMemo(() => getCurationProgress(goal), [goal, progressVersion])

  const finishIfEmpty = useCallback(async (query, queue, lastAction = null) => {
    if (queue.length > 0) {
      setState(previous => ({
        ...previous,
        queue,
        loading: false,
        busy: false,
        complete: false,
        refillError: null,
      }))
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
      busy: false,
      complete: true,
      refillError: null,
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
    if (!actionLock.current.tryAcquire()) return
    const query = buildCurationQuery(overrideQuery || filters)
    const seed = seedCandidates(overrideSeed || loadedImages)
    const starting = { ...initialState, active: true, query, queue: seed, loading: true, busy: true }
    stateRef.current = starting
    setState(starting)
    try {
      await refill(query, seed)
    } catch (error) {
      toast.error(`Could not start Curation Game: ${error.message}`)
      setState(previous => markCurationRefillFailure(previous, error))
    } finally {
      actionLock.current.release()
    }
  }, [filters, loadedImages, refill])

  const applyAction = useCallback(async (kind) => {
    if (!actionLock.current.tryAcquire()) return
    const snapshot = stateRef.current
    const item = snapshot.queue[0]
    if (!item) {
      actionLock.current.release()
      return
    }

    setState(previous => ({ ...previous, busy: true, refillError: null }))
    try {
      if (kind === 'keep') await setFavorite(item, true)
      else await discardForCuration(item, localStorage.getItem('localbooru_dumpster_path'))
    } catch (error) {
      toast.error(`${kind === 'keep' ? 'Keep' : 'Discard'} failed: ${error.message}`)
      setState(previous => ({ ...previous, busy: false }))
      actionLock.current.release()
      return
    }

    let countedDate = null
    try {
      countedDate = recordCurated()
    } catch (error) {
      toast.error(`Decision saved, but progress could not be recorded: ${error.message}`)
    }
    const committed = commitCurationAction(snapshot, kind, item, countedDate)
    stateRef.current = committed
    setProgressVersion(value => value + 1)
    setState(committed)

    try {
      await refill(snapshot.query, committed.queue, committed.lastAction)
    } catch (error) {
      toast.error(`Could not load the next curation item: ${error.message}`)
      setState(previous => markCurationRefillFailure(previous, error))
    } finally {
      actionLock.current.release()
    }
  }, [refill])

  const undo = useCallback(async () => {
    if (!actionLock.current.tryAcquire()) return
    const snapshot = stateRef.current
    if (!snapshot.lastAction) {
      actionLock.current.release()
      return
    }
    const action = snapshot.lastAction
    setState(previous => ({ ...previous, busy: true }))
    try {
      if (action.kind === 'keep') await setFavorite(action.item, false)
      else await restoreCurationDiscard(action.item)
      if (action.countedDate) {
        try {
          undoRecordedCurated(action.countedDate)
        } catch (error) {
          toast.error(`Undo succeeded, but progress could not be updated: ${error.message}`)
        }
      }
      setProgressVersion(value => value + 1)
      const next = {
        ...snapshot,
        queue: [action.item, ...snapshot.queue],
        lastAction: null,
        processed: Math.max(0, snapshot.processed - 1),
        busy: false,
        loading: false,
        complete: false,
        refillError: null,
      }
      stateRef.current = next
      setState(next)
    } catch (error) {
      toast.error(`Undo failed: ${error.message}`)
      setState(previous => ({ ...previous, busy: false }))
    } finally {
      actionLock.current.release()
    }
  }, [])

  const retryRefill = useCallback(async () => {
    if (!actionLock.current.tryAcquire()) return
    const snapshot = stateRef.current
    if (!snapshot.query) {
      actionLock.current.release()
      return
    }
    setState(previous => ({ ...previous, busy: true, loading: true, refillError: null }))
    try {
      await refill(snapshot.query, snapshot.queue, snapshot.lastAction)
    } catch (error) {
      toast.error(`Could not resume Curation Game: ${error.message}`)
      setState(previous => markCurationRefillFailure(previous, error))
    } finally {
      actionLock.current.release()
    }
  }, [refill])

  const unfavoriteAllAndRestart = useCallback(async () => {
    if (!actionLock.current.tryAcquire()) return
    const query = stateRef.current.query
    if (!query) {
      actionLock.current.release()
      return
    }
    setState(previous => ({ ...previous, busy: true }))
    try {
      while (true) {
        const result = await fetchImages({
          ...query,
          favorites_only: true,
          exclude_favorites: false,
          page: 1,
          per_page: 400,
        })
        if (!result.images?.length) break
        await unfavoriteCurationItems(result.images)
      }
      const restarting = { ...initialState, active: true, query, loading: true, busy: true }
      stateRef.current = restarting
      setState(restarting)
      await refill(query, [])
    } catch (error) {
      toast.error(`Could not restart: ${error.message}`)
      setState(previous => markCurationRefillFailure(previous, error))
    } finally {
      actionLock.current.release()
    }
  }, [refill])

  const exit = useCallback(() => {
    if (stateRef.current.busy) return
    stateRef.current = initialState
    actionLock.current.release()
    setState(initialState)
    onGalleryRefresh?.()
  }, [onGalleryRefresh])

  return {
    ...state,
    current: state.queue[0] || null,
    goal,
    progress,
    gestureVersion: progressVersion,
    start,
    keep: () => applyAction('keep'),
    discard: () => applyAction('discard'),
    undo,
    retryRefill,
    exit,
    unfavoriteAllAndRestart,
  }
}
