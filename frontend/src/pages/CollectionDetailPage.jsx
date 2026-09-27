import { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
import { fetchCollection, updateCollection, removeFromCollection } from '../api'
import Sidebar from '../components/Sidebar'
import MediaSectionsNav from '../components/MediaSectionsNav'
import MasonryGrid from '../components/MasonryGrid'
import Lightbox from '../components/Lightbox'
import { adjustmentLocator, imageMatchesLocator, updateImagesByLocator } from '../utils/imageAdjustments.js'
import { loadCollectionPages, restoreCollectionScroll, savedCollectionPage } from '../utils/collectionState.js'
import { useMobileDrawer } from '../hooks/useMobileDrawer'

export default function CollectionDetailPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const mediaType = searchParams.get('media_type') === 'video' ? 'video' : 'image'
  const [collection, setCollection] = useState(null)
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(true)
  const [loadedCollectionKey, setLoadedCollectionKey] = useState(null)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [lightboxIndex, setLightboxIndex] = useState(null)
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState('')
  const removingRef = useRef(false)
  const restoredScrollKeyRef = useRef(null)
  const drawer = useMobileDrawer()
  const collectionStateKey = `donutMediaCenter_collection_${mediaType}_${id}`

  const loadCollection = useCallback(async (pageNum = 1, append = false) => {
    try {
      const data = await fetchCollection(id, pageNum, 50, mediaType)
      setCollection(data)
      if (append) {
        setImages(prev => [...prev, ...data.images])
      } else {
        setImages(data.images || [])
      }
      setHasMore(data.has_more)
      setPage(pageNum)
      sessionStorage.setItem(`${collectionStateKey}_page`, String(pageNum))
      setLoading(false)
      return true
    } catch (e) {
      console.error('Failed to load collection:', e)
      setLoading(false)
      return false
    }
  }, [id, mediaType, collectionStateKey])

  useEffect(() => {
    let active = true
    const restore = async () => {
      try {
        const restored = await loadCollectionPages(
          nextPage => fetchCollection(id, nextPage, 50, mediaType),
          savedCollectionPage(sessionStorage, collectionStateKey),
          () => active,
        )
        if (!active || !restored) return
        setCollection(restored.collection)
        setImages(restored.images)
        setPage(restored.page)
        setHasMore(restored.hasMore)
        sessionStorage.setItem(`${collectionStateKey}_page`, String(restored.page))
      } catch (error) {
        console.error('Failed to restore collection:', error)
        if (active) {
          setCollection(null)
          setImages([])
          setPage(1)
          setHasMore(false)
        }
      } finally {
        if (active) {
          setLoadedCollectionKey(collectionStateKey)
          setLoading(false)
        }
      }
    }
    restore()
    return () => { active = false }
  }, [id, mediaType, collectionStateKey])

  useEffect(() => {
    if (loading || loadedCollectionKey !== collectionStateKey || restoredScrollKeyRef.current === collectionStateKey) return
    restoredScrollKeyRef.current = collectionStateKey
    requestAnimationFrame(() => {
      const container = document.querySelector('.collection-detail-page > .masonry-container')
      restoreCollectionScroll(sessionStorage, collectionStateKey, container)
    })
  }, [loading, loadedCollectionKey, collectionStateKey])

  useEffect(() => {
    sessionStorage.setItem(`donutMediaCenter_section_url_${mediaType}`, `${window.location.pathname}${window.location.search}`)
  }, [mediaType, searchParams])

  const handleLoadMore = useCallback(async () => {
    if (!hasMore || loading) return false
    const nextPage = page + 1
    setLoading(true)
    return loadCollection(nextPage, true)
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

  const handleRemoveFromCollection = useCallback(async (image) => {
    if (removingRef.current) return
    removingRef.current = true
    try {
      await removeFromCollection(id, [image.collection_legacy_member ? image.id : image])
      handleLightboxClose()
      setImages(prev => prev.filter(img => !(imageMatchesLocator(img, adjustmentLocator(image)) && img.collection_legacy_member === image.collection_legacy_member)))
      setCollection(prev => prev ? { ...prev, item_count: Math.max(0, (prev.item_count || 1) - 1) } : prev)
    } catch (e) {
      console.error('Failed to remove from collection:', e)
    } finally {
      removingRef.current = false
    }
  }, [id, handleLightboxClose])

  const lightboxImageIndex = lightboxIndex !== null
    ? images.findIndex(image => imageMatchesLocator(image, lightboxIndex))
    : -1

  return (
    <div className="app">
      <div className="main-container">
        {drawer.isOpen && <div className="sidebar-backdrop" onClick={drawer.close} />}
        <Sidebar mediaType={mediaType} mobileOpen={drawer.isOpen} onClose={drawer.close} />
        <main className="content with-sidebar collection-detail-page" onScrollCapture={(event) => {
          if (event.target.classList?.contains('masonry-container')) {
            sessionStorage.setItem(`${collectionStateKey}_scroll`, String(event.target.scrollTop))
          }
        }}>
        <MediaSectionsNav />
        <div className="collections-header collection-detail-header">
          <div className="collection-detail-title-row">
            <button className="menu-btn mobile-only" onClick={drawer.open} aria-label="Open menu">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18"/>
              </svg>
            </button>
            <button
              className="collections-create-btn collection-back-btn"
              onClick={() => navigate(`/collections?media_type=${mediaType}`)}
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

        {loadedCollectionKey !== collectionStateKey ? (
          <div className="collections-loading">Loading...</div>
        ) : images.length === 0 ? (
          <div className="collections-empty">
            <h2>Empty collection</h2>
            <p>Add {mediaType === 'video' ? 'videos' : 'images'} from the gallery lightbox.</p>
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
            onRemoveFromCollection={() => handleRemoveFromCollection(images[lightboxImageIndex])}
          />
        )}
        </main>
      </div>
    </div>
  )
}
