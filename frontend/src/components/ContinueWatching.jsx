import { useState, useEffect } from 'react'
import { getContinueWatching, clearWatchHistory, getMediaUrl } from '../api'
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

  useEffect(() => {
    loadItems()
  }, [])

  async function loadItems() {
    try {
      const data = await getContinueWatching()
      setItems(data.items || [])
    } catch (e) {
      // Silently fail - not critical
    }
    setLoading(false)
  }

  const handleDismiss = async (e, imageId) => {
    e.stopPropagation()
    try {
      await clearWatchHistory(imageId)
      setItems(prev => prev.filter(item => item.id !== imageId))
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
            key={item.id}
            className="continue-watching-card"
            onClick={() => onImageClick(item.id)}
          >
            <div className="continue-watching-thumb">
              <img src={getMediaUrl(item.thumbnail_url)} alt="" loading="lazy" />
              <div className="continue-watching-progress">
                <div className="continue-watching-progress-bar" style={{ width: `${(item.progress * 100).toFixed(0)}%` }} />
              </div>
              <span className="continue-watching-time">
                {formatTime(item.playback_position)} / {formatTime(item.watch_duration)}
              </span>
              <button className="continue-watching-dismiss" onClick={(e) => handleDismiss(e, item.id)} title="Dismiss">
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
