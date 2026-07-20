import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { fetchCollection, updateCollection, removeFromCollection, getMediaUrl } from '../api'
import Sidebar from '../components/Sidebar'
import MasonryGrid from '../components/MasonryGrid'
import Lightbox from '../components/Lightbox'
import { adjustmentLocator, imageMatchesLocator, updateImagesByLocator } from '../utils/imageAdjustments.js'
import { useMobileDrawer } from '../hooks/useMobileDrawer'

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
  const drawer = useMobileDrawer()

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

  const handleImageClick = (image) => {
    const locator = adjustmentLocator(image)
    window.history.pushState({ lightbox: true, locator }, '')
    setLightboxIndex(locator)
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

  const lightboxImageIndex = lightboxIndex !== null
    ? images.findIndex(image => imageMatchesLocator(image, lightboxIndex))
    : -1

  return (
    <div className="app">
      <div className="main-container">
        {drawer.isOpen && <div className="sidebar-backdrop" onClick={drawer.close} />}
        <Sidebar mobileOpen={drawer.isOpen} onClose={drawer.close} />
        <main className="content with-sidebar">
        <div className="collections-header collection-detail-header">
          <div className="collection-detail-title-row">
            <button className="menu-btn mobile-only" onClick={drawer.open} aria-label="Open menu">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18"/>
              </svg>
            </button>
            <button
              className="collections-create-btn collection-back-btn"
              onClick={() => navigate('/collections')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
              Back
            </button>
            {editing ? (
              <div className="collection-name-editor">
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleSaveName() }}
                  autoFocus
                  className="collection-name-input"
                />
                <button onClick={handleSaveName} className="collection-save-btn">Save</button>
              </div>
            ) : (
              <h1 className="collection-editable-name" onClick={() => { setEditing(true); setEditName(collection?.name || '') }}>
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
                setLightboxIndex(adjustmentLocator(images[newIdx]))
              }
            }}
            onTagClick={() => {}}
            onImageUpdate={(locator, updates) => {
              setImages(previous => updateImagesByLocator(previous, locator, updates))
            }}
          />
        )}
        </main>
      </div>
    </div>
  )
}
