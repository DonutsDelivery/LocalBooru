import { useCallback, useMemo, useRef, useState } from 'react'
import {
  discardForCuration,
  fetchImages,
  restoreCurationDiscard,
  setFavorite,
  unfavoriteCurationItems,
} from '../api'
import { imageMatchesLocator, updateImagesByLocator } from '../utils/imageAdjustments.js'
import { toast } from '../components/Toast'
import {
  buildCurationQuery,
  commitCurationAction,
  createCurationActionLock,
  markCurationRefillFailure,
  mergeCandidates,
  rotateCurationCandidates,
  seedCandidates,
  setCurrentCurationFavorite,
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
  actionHistory: [],
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

  // alreadyFavorited re-commits a favorited item as kept without calling the API.
  const applyAction = useCallback(async (kind, { alreadyFavorited = false } = {}) => {
    if (!actionLock.current.tryAcquire()) return
    const snapshot = stateRef.current
    const item = snapshot.queue[0]
    if (!item || (alreadyFavorited && !item.is_favorite)) {
      actionLock.current.release()
      return
    }

    setState(previous => ({ ...previous, busy: true, refillError: null }))
    try {
      if (kind === 'keep') {
        if (!alreadyFavorited) await setFavorite(item, true)
      } else {
        // A discarded file must not stay favorited; undo re-favorites it.
        if (item.is_favorite) await setFavorite(item, false)
        try {
          await discardForCuration(item, localStorage.getItem('localbooru_dumpster_path'))
        } catch (error) {
          if (item.is_favorite) await setFavorite(item, true).catch(() => {})
          throw error
        }
      }
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
    const history = snapshot.actionHistory || []
    const action = history[history.length - 1] || snapshot.lastAction
    setState(previous => ({ ...previous, busy: true }))
    try {
      // Going back after a Favorite keeps it favorited and just reopens the decision.
      let restoredItem = action.item
      if (action.kind === 'discard') {
        await restoreCurationDiscard(action.item)
        if (action.item.is_favorite) {
          try {
            await setFavorite(action.item, true)
          } catch (error) {
            toast.error(`Restored, but could not re-favorite: ${error.message}`)
            restoredItem = { ...action.item, is_favorite: false }
          }
        }
      }
      if (action.countedDate) {
        try {
          undoRecordedCurated(action.countedDate)
        } catch (error) {
          toast.error(`Undo succeeded, but progress could not be updated: ${error.message}`)
        }
      }
      setProgressVersion(value => value + 1)
      const actionHistory = history.slice(0, -1)
      const next = {
        ...snapshot,
        queue: [restoredItem, ...snapshot.queue],
        actionHistory,
        lastAction: actionHistory[actionHistory.length - 1] || null,
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

  // Favorite on an undecided item favorites it and advances; on an already
  // favorited item it unfavorites and stays so the decision can be remade.
  const keep = useCallback(async () => {
    if (!stateRef.current.queue[0]?.is_favorite) return applyAction('keep')
    if (!actionLock.current.tryAcquire()) return
    const item = stateRef.current.queue[0]
    setState(previous => ({ ...previous, busy: true }))
    try {
      await setFavorite(item, false)
      setState(previous => {
        const next = { ...setCurrentCurationFavorite(previous, false), busy: false }
        stateRef.current = next
        return next
      })
    } catch (error) {
      toast.error(`Remove favorite failed: ${error.message}`)
      setState(previous => ({ ...previous, busy: false }))
    } finally {
      actionLock.current.release()
    }
  }, [applyAction])

  const navigate = useCallback((direction) => {
    const snapshot = stateRef.current
    if (snapshot.busy || snapshot.queue.length < 2) return

    const queue = rotateCurationCandidates(snapshot.queue, direction)
    if (queue === snapshot.queue) return

    const next = { ...snapshot, queue }
    stateRef.current = next
    setState(next)
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

  const updateImage = useCallback((locator, updates) => {
    setState(previous => {
      const lastAction = previous.lastAction && imageMatchesLocator(previous.lastAction.item, locator)
        ? { ...previous.lastAction, item: { ...previous.lastAction.item, ...updates } }
        : previous.lastAction
      const next = {
        ...previous,
        queue: updateImagesByLocator(previous.queue, locator, updates),
        lastAction,
      }
      stateRef.current = next
      return next
    })
  }, [])

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
    navigate,
    keep,
    keepFavorited: () => applyAction('keep', { alreadyFavorited: true }),
    discard: () => applyAction('discard'),
    undo,
    retryRefill,
    updateImage,
    exit,
    unfavoriteAllAndRestart,
  }
}
