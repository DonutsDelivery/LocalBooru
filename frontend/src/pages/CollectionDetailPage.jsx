import './SharedCollectionPage.css'
import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useParams, useNavigate, useSearchParams, useLocation } from 'react-router-dom'
import { fetchCollection, updateCollection, removeFromCollection } from '../api'
import Sidebar from '../components/Sidebar'
import GalleryTileSizeControl from '../components/GalleryTileSizeControl'
import { useGalleryTileSize } from '../hooks/useGalleryTileSize'
import { collectionRequestFilters, collectionSidebarProps, readCollectionFilters, sidebarFilterParams, toggleCollectionTag } from '../utils/galleryCollectionFilters'
import MasonryGrid from '../components/MasonryGrid'
import Lightbox from '../components/Lightbox'
import { adjustmentLocator, imageMatchesLocator, updateImagesByLocator } from '../utils/imageAdjustments.js'
import { collectionGalleryTarget, loadCollectionPages, restoreCollectionScroll, savedCollectionPage } from '../utils/collectionState.js'
import { useMobileDrawer } from '../hooks/useMobileDrawer'

export default function CollectionDetailPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const filters = useMemo(() => readCollectionFilters(searchParams), [searchParams])
  const requestFilters = useMemo(() => collectionRequestFilters(filters), [filters])
  const [tileSize, setTileSize] = useGalleryTileSize()
  const mediaType = searchParams.get('media_type') === 'video' ? 'video' : 'image'
  const galleryReturnTo = collectionGalleryTarget(location, mediaType)
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
  const [notice, setNotice] = useState('')
  const [sidebarHover, setSidebarHover] = useState(false)
  const requestGeneration = useRef(0)
  const paginationBusy = useRef(false)
  const filterKey = JSON.stringify(requestFilters)
  const collectionStateKey = `donutMediaCenter_collection_${mediaType}_${id}${Object.keys(requestFilters).length ? `_${encodeURIComponent(filterKey)}` : ''}`
  const activeKey = useRef(collectionStateKey)
  activeKey.current = collectionStateKey

  const loadCollection = useCallback(async (pageNum = 1, append = false) => {
    const generation = requestGeneration.current
    try {
      const data = await fetchCollection(id, pageNum, 50, mediaType, requestFilters)
      if (generation !== requestGeneration.current || activeKey.current !== collectionStateKey) return false
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
      if (generation !== requestGeneration.current || activeKey.current !== collectionStateKey) return false
      setNotice(e?.response?.data?.error || e.message || 'Could not load this collection.')
      setLoading(false)
      return false
    }
  }, [id, mediaType, collectionStateKey, requestFilters])

  useEffect(() => {
    let active = true
    const generation = ++requestGeneration.current
    paginationBusy.current = true
    setLoading(true)
    setLightboxIndex(null)
    setEditing(false)
    setNotice('')
    const restore = async () => {
      try {
        const restored = await loadCollectionPages(
          nextPage => fetchCollection(id, nextPage, 50, mediaType, requestFilters),
          savedCollectionPage(sessionStorage, collectionStateKey),
          () => active && generation === requestGeneration.current && activeKey.current === collectionStateKey,
        )
        if (!active || generation !== requestGeneration.current || activeKey.current !== collectionStateKey || !restored) return
        setCollection(restored.collection)
        setImages(restored.images)
        setPage(restored.page)
        setHasMore(restored.hasMore)
        sessionStorage.setItem(`${collectionStateKey}_page`, String(restored.page))
      } catch (error) {
        if (active && generation === requestGeneration.current && activeKey.current === collectionStateKey) {
          setNotice(error?.response?.data?.error || error.message || 'Could not load this collection.')
          setCollection(null)
          setImages([])
          setPage(1)
          setHasMore(false)
        }
      } finally {
        if (active && generation === requestGeneration.current && activeKey.current === collectionStateKey) {
          paginationBusy.current = false
          setLoadedCollectionKey(collectionStateKey)
          setLoading(false)
        }
      }
    }
    restore()
    return () => { active = false; requestGeneration.current += 1; paginationBusy.current = false }
  }, [id, mediaType, collectionStateKey, requestFilters])

  useEffect(() => {
    if (loading || loadedCollectionKey !== collectionStateKey || restoredScrollKeyRef.current === collectionStateKey) return
    restoredScrollKeyRef.current = collectionStateKey
    requestAnimationFrame(() => {
      const container = document.querySelector('.collection-detail-page > .masonry-container')
      restoreCollectionScroll(sessionStorage, collectionStateKey, container)
    })
  }, [loading, loadedCollectionKey, collectionStateKey])

  const handleLoadMore = useCallback(async () => {
    if (!hasMore || loading || paginationBusy.current || loadedCollectionKey !== collectionStateKey) return false
    paginationBusy.current = true
    const nextPage = page + 1
    setLoading(true)
    const generation = requestGeneration.current
    try { return await loadCollection(nextPage, true) } finally { if (generation === requestGeneration.current) paginationBusy.current = false }
  }, [hasMore, loading, page, loadCollection, loadedCollectionKey, collectionStateKey])

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
    const generation = requestGeneration.current
    const name = editName.trim()
    try {
      await updateCollection(id, { name })
      if (generation !== requestGeneration.current || activeKey.current !== collectionStateKey) return
      setCollection(prev => prev ? { ...prev, name } : prev)
      setEditing(false)
    } catch (e) {
      console.error('Failed to update name:', e)
    }
  }

  const handleRemoveFromCollection = useCallback(async (image) => {
    if (removingRef.current) return
    removingRef.current = true
    const generation = requestGeneration.current
    try {
      await removeFromCollection(id, [image.collection_legacy_member ? image.id : image])
      if (generation !== requestGeneration.current || activeKey.current !== collectionStateKey) return
      handleLightboxClose()
      setImages(prev => prev.filter(img => !(imageMatchesLocator(img, adjustmentLocator(image)) && img.collection_legacy_member === image.collection_legacy_member)))
      setCollection(prev => prev ? { ...prev, item_count: Math.max(0, (prev.item_count || 1) - 1) } : prev)
    } catch (e) {
      console.error('Failed to remove from collection:', e)
    } finally {
      removingRef.current = false
    }
  }, [id, handleLightboxClose, collectionStateKey])

  const visibleCollection = loadedCollectionKey === collectionStateKey ? collection : null
  const lightboxImageIndex = lightboxIndex !== null
    && loadedCollectionKey === collectionStateKey ? images.findIndex(image => imageMatchesLocator(image, lightboxIndex))
    : -1

  return (
    <div className="app">
      {sidebarHover && <div className="sidebar-backdrop lightbox-backdrop" onClick={() => setSidebarHover(false)} />}
      <div className="main-container">
        {drawer.isOpen && <div className="sidebar-backdrop" onClick={drawer.close} />}
        <Sidebar {...collectionSidebarProps(filters)} defaultSort="collection" mediaType={mediaType} mobileOpen={drawer.isOpen}
          onClose={() => { drawer.close(); setSidebarHover(false) }} currentCollection={visibleCollection}
          selectedImage={images[lightboxImageIndex]} lightboxMode={lightboxImageIndex >= 0} lightboxHover={sidebarHover} onMouseLeave={() => setSidebarHover(false)}
          onSearch={(...args) => setSearchParams(sidebarFilterParams(searchParams, false, ...args))}
          onTagClick={tag => setSearchParams(toggleCollectionTag(searchParams, tag))}
          onRenameCollection={visibleCollection ? () => { setEditing(true); setEditName(visibleCollection.name) } : undefined}
          sidebarActions={<><button className="media-collection-link collection-sidebar-action" onClick={() => navigate(`/collections?media_type=${mediaType}`, { state: { galleryReturnTo } })}>← Back to Collections</button>
            {editing ? <div className="collection-name-editor"><input aria-label="Collection name" value={editName} onChange={event => setEditName(event.target.value)} onKeyDown={event => { if (event.key === 'Enter') handleSaveName() }} autoFocus className="collection-name-input" /><button onClick={handleSaveName} className="collection-save-btn">Save</button></div> : <h3>{visibleCollection?.name || 'Collection'}</h3>}
          </>} />
        <main className="content with-sidebar collection-detail-page" onScrollCapture={(event) => {
          if (loadedCollectionKey === collectionStateKey && event.target.classList?.contains('masonry-container')) {
            sessionStorage.setItem(`${collectionStateKey}_scroll`, String(event.target.scrollTop))
          }
        }}>
        <button className="menu-btn mobile-only collection-mobile-menu" onClick={drawer.open} aria-label="Open menu"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12h18M3 6h18M3 18h18" /></svg></button>
        <div className="floating-controls"><GalleryTileSizeControl tileSize={tileSize} onChange={setTileSize} /></div>
        {notice && <p className="collection-filter-notice" role="alert">{notice}</p>}

        {loadedCollectionKey !== collectionStateKey ? (
          <div className="collections-loading">Loading...</div>
        ) : images.length === 0 ? (
          <div className="collections-empty">
            <h2>{Object.keys(requestFilters).length ? 'No matching items' : 'Empty collection'}</h2>
            <p>{Object.keys(requestFilters).length ? 'Try changing the sidebar filters.' : <>Add {mediaType === 'video' ? 'videos' : 'images'} from the gallery lightbox.</>}</p>
          </div>
        ) : (
          <MasonryGrid
            images={images}
            onImageClick={handleImageClick}
            onLoadMore={handleLoadMore}
            loading={loading}
            hasMore={hasMore}
            tileSize={tileSize}
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
            onTagClick={tag => setSearchParams(toggleCollectionTag(searchParams, tag))}
            onSidebarHover={setSidebarHover}
            sidebarOpen={drawer.isOpen || sidebarHover}
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
