import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { fetchCollection, updateCollection, removeFromCollection, getMediaUrl } from '../api'
import Sidebar from '../components/Sidebar'
import MasonryGrid from '../components/MasonryGrid'
import Lightbox from '../components/Lightbox'

export default function CollectionDetailPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [collection, setCollection] = useState(null)
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [lightboxIndex, setLightboxIndex] = useState(null)
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState('')

  const loadCollection = useCallback(async (pageNum = 1, append = false) => {
    try {
      const data = await fetchCollection(id, pageNum)
      setCollection(data)
      if (append) {
        setImages(prev => [...prev, ...data.images])
      } else {
        setImages(data.images || [])
      }
      setHasMore(data.has_more)
    } catch (e) {
      console.error('Failed to load collection:', e)
    }
    setLoading(false)
  }, [id])

  useEffect(() => {
    loadCollection()
  }, [loadCollection])

  const handleLoadMore = useCallback(() => {
    if (!hasMore || loading) return
    const nextPage = page + 1
    setPage(nextPage)
    loadCollection(nextPage, true)
  }, [hasMore, loading, page, loadCollection])

  const handleImageClick = (imageId) => {
    window.history.pushState({ lightbox: true, imageId }, '')
    setLightboxIndex(imageId)
  }

  const handleLightboxClose = useCallback(() => {
    if (window.history.state?.lightbox) {
      window.history.back()
    } else {
      setLightboxIndex(null)
    }
  }, [])

  // Handle popstate for lightbox
  useEffect(() => {
    const handlePopState = (e) => {
      if (lightboxIndex !== null && !e.state?.lightbox) {
        setLightboxIndex(null)
      }
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [lightboxIndex])

  const handleSaveName = async () => {
    if (!editName.trim()) return
    try {
      await updateCollection(id, { name: editName.trim() })
      setCollection(prev => ({ ...prev, name: editName.trim() }))
      setEditing(false)
    } catch (e) {
      console.error('Failed to update name:', e)
    }
  }

  const handleRemoveFromCollection = useCallback(async (imageId) => {
    try {
      await removeFromCollection(id, [imageId])
      setImages(prev => prev.filter(img => img.id !== imageId))
      setCollection(prev => prev ? { ...prev, item_count: Math.max(0, (prev.item_count || 1) - 1) } : prev)
    } catch (e) {
      console.error('Failed to remove from collection:', e)
    }
  }, [id])

  // Find lightbox index from imageId
  const lightboxImageIndex = lightboxIndex !== null ? images.findIndex(img => img.id === lightboxIndex) : -1

  return (
    <div className="app">
      <Sidebar />
      <main className="content with-sidebar">
        <div className="collections-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              className="collections-create-btn"
              style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-primary, #e0e0e0)' }}
              onClick={() => navigate('/collections')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
              Back
            </button>
            {editing ? (
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleSaveName() }}
                  autoFocus
                  style={{ padding: '6px 10px', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)', borderRadius: '6px', color: 'var(--text-primary)', fontSize: '1.1rem' }}
                />
                <button onClick={handleSaveName} style={{ padding: '6px 12px', background: '#4fc3f7', color: '#000', border: 'none', borderRadius: '6px', fontWeight: 600, cursor: 'pointer' }}>Save</button>
              </div>
            ) : (
              <h1 style={{ margin: 0, cursor: 'pointer' }} onClick={() => { setEditing(true); setEditName(collection?.name || '') }}>
                {collection?.name || 'Loading...'}
                {collection && <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginLeft: '8px' }}>({collection.item_count} items)</span>}
              </h1>
            )}
          </div>
        </div>

        {loading ? (
          <div className="collections-loading">Loading...</div>
        ) : images.length === 0 ? (
          <div className="collections-empty">
            <h2>Empty collection</h2>
            <p>Add images from the gallery lightbox.</p>
          </div>
        ) : (
          <MasonryGrid
            images={images}
            onImageClick={handleImageClick}
            onLoadMore={handleLoadMore}
            loading={loading}
            hasMore={hasMore}
            tileSize={3}
          />
        )}

        {lightboxImageIndex >= 0 && (
          <Lightbox
            images={images}
            currentIndex={lightboxImageIndex}
            total={images.length}
            onClose={handleLightboxClose}
            onNav={(dir) => {
              const newIdx = lightboxImageIndex + dir
              if (newIdx >= 0 && newIdx < images.length) {
                setLightboxIndex(images[newIdx].id)
              }
            }}
            onTagClick={() => {}}
            onImageUpdate={() => loadCollection()}
          />
        )}
      </main>
    </div>
  )
}
