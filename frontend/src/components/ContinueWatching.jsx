import { useState, useEffect, useRef } from 'react'
import { getContinueWatching, clearWatchHistory, fetchImage, getMediaUrl, getFamilyModeStatus, subscribeToLibraryEvents } from '../api'
import { canonicalWatchItem, watchHistoryIdentityKey, watchHistoryLocator } from '../utils/continueWatching.js'
import './ContinueWatching.css'

function formatTime(s) {
  if (!s || !isFinite(s)) return '0:00'
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = Math.floor(s % 60)
  if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
  return `${m}:${sec.toString().padStart(2, '0')}`
}

export default function ContinueWatching({ onImageClick }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  // Null until the backend lock state is known; the server filters the list.
  const [familyLocked, setFamilyLocked] = useState(null)
  const [lockRevision, setLockRevision] = useState(0)

  const loadGeneration = useRef(0)

  useEffect(() => {
    let alive = true
    let receivedEvent = false
    getFamilyModeStatus()
      .then(data => { if (alive && !receivedEvent) setFamilyLocked(!!data.is_locked) })
      .catch(() => { if (alive && !receivedEvent) setFamilyLocked(false) })
    const unsubscribe = subscribeToLibraryEvents(event => {
      if (!alive || event.type !== 'family_mode') return
      const isLocked = event.data?.is_locked ?? event.is_locked
      if (isLocked != null) {
        receivedEvent = true
        loadGeneration.current++
        setItems([])
        setLoading(true)
        setFamilyLocked(!!isLocked)
        setLockRevision(revision => revision + 1)
      }
    })
    return () => { alive = false; loadGeneration.current++; unsubscribe() }
  }, [])

  useEffect(() => {
    if (familyLocked === null) return
    const generation = ++loadGeneration.current
    loadItems(generation)
    return () => { loadGeneration.current++ }
  }, [familyLocked, lockRevision])

  async function loadItems(generation) {
    try {
      const data = await getContinueWatching()
      if (generation !== loadGeneration.current) return
      const watchItems = data.items || []

      // Hydrate each watch history item with image details (thumbnail, filename)
      const hydrated = await Promise.all(
        watchItems.map(async (item) => {
          try {
            const locator = watchHistoryLocator(item)
            const image = await fetchImage(locator.imageId, {
              directoryId: locator.directoryId,
              libraryId: locator.libraryId,
              optional: true,
            })
            return canonicalWatchItem(item, image)
          } catch (error) {
            // Deleted images are expected; surface identity/API regressions.
            if (error?.response?.status !== 404) {
              console.error('Failed to hydrate Continue Watching item:', error)
            }
            return null
          }
        })
      )

      if (generation !== loadGeneration.current) return
      setItems(hydrated.filter(Boolean))
    } catch (e) {
      // Silently fail - not critical
    }
    if (generation === loadGeneration.current) setLoading(false)
  }

  const handleDismiss = async (e, item) => {
    e.stopPropagation()
    try {
      const locator = watchHistoryLocator(item)
      await clearWatchHistory(locator.imageId, locator.directoryId, locator.libraryId)
      const dismissedKey = watchHistoryIdentityKey(item)
      setItems(prev => prev.filter(candidate => watchHistoryIdentityKey(candidate) !== dismissedKey))
    } catch (e) {
      console.error('Failed to dismiss:', e)
    }
  }

  const handleClearAll = async () => {
    try {
      await clearWatchHistory()
      setItems([])
    } catch (e) {
      console.error('Failed to clear all:', e)
    }
  }

  if (loading || items.length === 0) return null

  return (
    <div className="continue-watching">
      <div className="continue-watching-header">
        <h3>Continue Watching ({items.length})</h3>
        <button className="continue-watching-clear" onClick={handleClearAll}>Clear All</button>
      </div>
      <div className="continue-watching-row">
        {items.map(item => (
          <div
            key={watchHistoryIdentityKey(item)}
            className="continue-watching-card"
            onClick={() => onImageClick(item)}
          >
            <div className="continue-watching-thumb">
              <img src={getMediaUrl(item.thumbnail_url)} alt="" loading="lazy" />
              <div className="continue-watching-progress">
                <div className="continue-watching-progress-bar" style={{ width: `${(item.progress * 100).toFixed(0)}%` }} />
              </div>
              <span className="continue-watching-time">
                {formatTime(item.playback_position)} / {formatTime(item.duration)}
              </span>
              <button className="continue-watching-dismiss" onClick={(e) => handleDismiss(e, item)} title="Dismiss">
                <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
              </button>
            </div>
            <span className="continue-watching-name">{item.original_filename || item.filename}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
