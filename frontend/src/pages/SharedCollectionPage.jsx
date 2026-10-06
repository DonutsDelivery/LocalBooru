// SPDX-License-Identifier: GPL-3.0-only
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useLocation, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import RemoteCreatorCollections from '../components/RemoteCreatorCollections'
import GalleryTileSizeControl from '../components/GalleryTileSizeControl'
import { useGalleryTileSize } from '../hooks/useGalleryTileSize'
import { collectionRequestFilters, collectionSidebarProps, readCollectionFilters, REMOTE_SORT_VALUES, sidebarFilterParams, toggleCollectionTag } from '../utils/galleryCollectionFilters'
import MasonryGrid from '../components/MasonryGrid'
import Lightbox from '../components/Lightbox'
import { browseRemoteSource, fetchDirectories, fetchRemoteSourceTags, getRemoteSources, importRemoteItem } from '../api'
import { useMobileDrawer } from '../hooks/useMobileDrawer'
import { collectionGalleryTarget } from '../utils/collectionState'
import { appendRemoteGalleryItems, remoteGalleryItem, remotePageHasMore } from '../utils/remoteGallery'
import './SharedCollectionPage.css'

const message = error => error?.response?.data?.detail || error?.response?.data?.error || error?.message || 'Could not load this community.'
const emptyResult = scope => ({ scope, images: [], page: 0, hasMore: true, stale: false, offline: false, partial: false, loading: true, error: '', nextCursor: null })

export default function SharedCollectionPage() {
  const { sourceId } = useParams()
  const [params, setParams] = useSearchParams()
  const mediaType = params.get('media_type') === 'video' ? 'video' : 'image'
  const filters = useMemo(() => readCollectionFilters(params, true), [params])
  const requestFilterKey = JSON.stringify(collectionRequestFilters(filters, true))
  const requestFilters = useMemo(() => JSON.parse(requestFilterKey), [requestFilterKey])
  const creatorView = ['available', 'following', 'offline', 'all'].includes(params.get('creator_view')) ? params.get('creator_view') : 'available'
  const creatorQuery = params.get('creator_query') || ''
  const query = filters.tags
  const navigate = useNavigate()
  const location = useLocation()
  const galleryReturnTo = collectionGalleryTarget(location, mediaType)
  return <SharedCommunityGallery key={`${sourceId}:${mediaType}`} sourceId={sourceId} mediaType={mediaType} query={query} filters={filters} requestFilters={requestFilters} creatorView={creatorView} creatorQuery={creatorQuery}
    onCreatorDirectoryChange={(view, query) => { const next = new URLSearchParams(params); if (view === 'available') next.delete('creator_view'); else next.set('creator_view', view); if (query) next.set('creator_query', query); else next.delete('creator_query'); setParams(next, { replace: true }) }}
    onCreatorSelect={(id, offline) => { const next = new URLSearchParams(params); if (id != null) next.set('creator_id', id); else next.delete('creator_id'); if (offline) next.set('offline_only', 'true'); else next.delete('offline_only'); setParams(next) }}
    onFilters={(...args) => setParams(sidebarFilterParams(params, true, ...args))}
    onTag={tag => setParams(toggleCollectionTag(params, tag, true))}
    onBack={() => navigate(`/collections?media_type=${mediaType}`, { state: { galleryReturnTo } })}
    settingsReturnTo={`${location.pathname}${location.search}`} />
}

function SharedCommunityGallery({ sourceId, mediaType, query, filters, requestFilters, creatorView, creatorQuery, onCreatorDirectoryChange, onCreatorSelect, onFilters, onTag, onBack, settingsReturnTo }) {
  const extraFilters = { ...requestFilters }; delete extraFilters.q
  const scope = JSON.stringify([sourceId, mediaType, query, ...(Object.keys(extraFilters).length ? [extraFilters] : [])])
  const sessionKey = `donutMediaCenter_shared_${encodeURIComponent(scope)}`
  const [result, setResult] = useState(() => emptyResult(scope))
  const resultRef = useRef(result)
  const generation = useRef(0)
  const activeScope = useRef(scope)
  activeScope.current = scope
  const busyRef = useRef(false)
  const restoringRef = useRef(true)
  const [restoring, setRestoring] = useState(true)
  const [sourceName, setSourceName] = useState('')
  const [facets, setFacets] = useState({ tags: [], available_ratings: [] })
  const tagContext = useMemo(() => ({ ...requestFilters, media_type: mediaType }), [requestFilters, mediaType])
  const [selectedId, setSelectedId] = useState(null)
  const selectedRef = useRef(selectedId)
  selectedRef.current = selectedId
  const [sidebarHover, setSidebarHover] = useState(false)
  const [tileSize, setTileSize] = useGalleryTileSize()
  const [directories, setDirectories] = useState([])
  const [directoryId, setDirectoryId] = useState('')
  const [importBusy, setImportBusy] = useState(false)
  const importGeneration = useRef(0)
  const [importNotice, setImportNotice] = useState('')
  const contentRef = useRef(null)
  const scrollContainer = () => contentRef.current?.querySelector('.masonry-container')
  const lightboxScroll = useRef(0)
  const drawer = useMobileDrawer()
  const current = result.scope === scope ? result : emptyResult(scope)
  const images = useMemo(() => current.images.map(image => ({ ...image, remote_source_name: sourceName || image.remote_source_name })), [current.images, sourceName])
  const selectedIndex = images.findIndex(image => image.id === selectedId)
  const selected = images[selectedIndex]

  useEffect(() => {
    let active = true
    getRemoteSources().then(data => {
      if (active) setSourceName(data.sources?.find(source => source.source_id === sourceId)?.display_name || '')
    }).catch(() => {})
    return () => { active = false }
  }, [sourceId])

  useEffect(() => {
    let active = true
    setFacets(previous => ({ ...previous, tags: [] }))
    fetchRemoteSourceTags(sourceId, '', 100, tagContext).then(data => { if (active) setFacets(data) }).catch(() => {})
    return () => { active = false }
  }, [sourceId, tagContext])

  const loadPage = useCallback(async (page, append = false) => {
    if (busyRef.current) return null
    busyRef.current = true
    const request = ++generation.current
    const previous = resultRef.current.scope === scope ? resultRef.current : emptyResult(scope)
    setResult({ ...previous, loading: true, error: '' })
    try {
      const data = await browseRemoteSource(sourceId, { ...requestFilters, q: requestFilters.q || '', media_type: mediaType, page, per_page: 60, ...(append && previous.nextCursor ? { cursor: previous.nextCursor } : {}) })
      if (request !== generation.current || activeScope.current !== scope) return null
      if (Array.isArray(data.available_ratings)) setFacets(previous => ({ ...previous, available_ratings: data.available_ratings }))
      const incoming = (data.items || []).map(item => remoteGalleryItem(item, { sourceId, offlineOnly: Boolean(requestFilters.offline_only || data.offline) }))
      const next = { scope, images: append ? appendRemoteGalleryItems(previous.images, incoming) : incoming,
        page, hasMore: remotePageHasMore({ ...data, page, per_page: data.per_page || 60 }),
        stale: Boolean(data.stale), offline: Boolean(data.offline), partial: Boolean(data.partial || (append && previous.partial)), loading: false, error: '', nextCursor: data.next_cursor }
      resultRef.current = next
      setResult(next)
      sessionStorage.setItem(`${sessionKey}_page`, String(page))
      return next
    } catch (error) {
      if (request !== generation.current || activeScope.current !== scope) return null
      const next = { ...previous, loading: false, error: message(error) }
      resultRef.current = next
      setResult(next)
      return null
    } finally {
      if (request === generation.current) busyRef.current = false
    }
  }, [sourceId, mediaType, query, scope, sessionKey, requestFilters])

  // A new source, media category or submitted search owns a fresh result list.
  useEffect(() => {
    generation.current += 1
    busyRef.current = false
    const next = emptyResult(scope)
    resultRef.current = next
    setResult(next)
    setSelectedId(null)
    setImportNotice('')
    setImportBusy(false)
    const savedPage = Number(sessionStorage.getItem(`${sessionKey}_page`))
    const savedScrollValue = sessionStorage.getItem(`${sessionKey}_scroll`)
    const savedScroll = Number(savedScrollValue)
    const lastPage = Number.isSafeInteger(savedPage) && savedPage > 0 ? Math.min(savedPage, 100) : 1
    let active = true
    restoringRef.current = true
    setRestoring(true)
    if (scrollContainer()) scrollContainer().scrollTop = 0
    const restore = async () => {
      try {
        for (let page = 1; page <= lastPage; page += 1) {
          const loaded = await loadPage(page, page > 1)
          if (!active || !loaded) return
          if (!loaded.hasMore) break
        }
        if (active && savedScrollValue !== null && Number.isFinite(savedScroll)) requestAnimationFrame(() => {
          if (active && scrollContainer()) scrollContainer().scrollTop = Math.max(0, savedScroll)
        })
      } finally {
        if (active) { restoringRef.current = false; setRestoring(false) }
      }
    }
    restore()
    return () => { active = false; generation.current += 1; busyRef.current = false }
  }, [scope, query, sessionKey, loadPage])

  const loadMore = useCallback(async () => {
    const previous = resultRef.current
    if (previous.scope !== scope || !previous.hasMore || previous.loading || previous.error || busyRef.current || restoringRef.current) return false
    return Boolean(await loadPage(previous.page + 1, true))
  }, [scope, loadPage])

  // A filtered catalog page can be empty while later pages still contain media.
  useEffect(() => {
    if (!restoring && !current.loading && !current.error && current.hasMore && images.length === 0) loadMore()
  }, [restoring, current.loading, current.error, current.hasMore, current.page, images.length, loadMore])

  useEffect(() => {
    importGeneration.current += 1
    setImportBusy(false)
    setImportNotice('')
    if (!selectedId) return
    let active = true
    fetchDirectories(true).then(data => {
      if (active) setDirectories((data.directories || []).filter(directory => directory.enabled !== false && directory.path_exists !== false))
    }).catch(error => { if (active) setImportNotice(message(error)) })
    return () => { active = false }
  }, [selectedId])

  const openImage = image => {
    lightboxScroll.current = scrollContainer()?.scrollTop || 0
    setSelectedId(image.id)
    setImportNotice('')
  }
  const closeImage = () => {
    setSelectedId(null)
    setSidebarHover(false)
    if (scrollContainer()) scrollContainer().scrollTop = lightboxScroll.current
  }
  const navigateImage = async direction => {
    const initialId = selectedId
    const requestScope = scope
    let index = selectedIndex + direction
    if (index >= images.length && current.hasMore) {
      const next = await loadPage(current.page + 1, true)
      if (!next || activeScope.current !== requestScope || selectedRef.current !== initialId) return
      if (next.images[index]) setSelectedId(next.images[index].id)
      return
    }
    if (images.length) { index = (index + images.length) % images.length; setSelectedId(images[index].id) }
  }
  const importImage = async () => {
    if (!selected || !directoryId || importBusy) return
    const importRequest = ++importGeneration.current
    const importingId = selected.id
    const importingScope = scope
    setImportBusy(true)
    setImportNotice('')
    try {
      const original = selected.remote_item
      const result = await importRemoteItem(sourceId, selected.remote_post_id, {
        kind: original.media.find(media => media.kind === 'original')?.kind || original.media[0]?.kind,
        directory_id: Number(directoryId),
      })
      if (importGeneration.current === importRequest && activeScope.current === importingScope && selectedRef.current === importingId) setImportNotice(`Saved a local copy as image #${result.image_id}.`)
    } catch (error) {
      if (importGeneration.current === importRequest && activeScope.current === importingScope && selectedRef.current === importingId) setImportNotice(message(error))
    } finally { if (importGeneration.current === importRequest && activeScope.current === importingScope) setImportBusy(false) }
  }

  const searchRemoteTags = useCallback(async (query, limit) => {
    const data = await fetchRemoteSourceTags(sourceId, query, limit, tagContext)
    return data.tags || []
  }, [sourceId, tagContext])
  const searchCommunity = (...args) => {
    const next = sidebarFilterParams(new URLSearchParams(settingsReturnTo.split('?')[1] || ''), true, ...args)
    const nextFilters = collectionRequestFilters(readCollectionFilters(next, true), true)
    if (JSON.stringify(nextFilters) === JSON.stringify(requestFilters)) {
      generation.current += 1
      busyRef.current = false
      setSelectedId(null)
      if (scrollContainer()) scrollContainer().scrollTop = 0
      loadPage(1)
    } else onFilters(...args)
  }

  const remoteActions = <div className="shared-import-controls">
    <label>Save to local directory<select aria-label="Import destination" value={directoryId} onChange={event => setDirectoryId(event.target.value)}><option value="">Choose a directory</option>{directories.map(directory => <option key={directory.id} value={directory.id}>{directory.name || directory.path}</option>)}</select></label>
    <button onClick={importImage} disabled={!directoryId || importBusy}>{importBusy ? 'Importing…' : 'Import a local copy'}</button>
    {importNotice && <p role="status">{importNotice}</p>}
  </div>

  return <div className="app">
    {sidebarHover && <div className="sidebar-backdrop lightbox-backdrop" onClick={() => setSidebarHover(false)} />}
    <div className="main-container">
    {drawer.isOpen && <div className="sidebar-backdrop" onClick={drawer.close} />}
    <Sidebar {...collectionSidebarProps(filters)} filterScope="remote" supportedSortValues={REMOTE_SORT_VALUES} availableRatings={facets.available_ratings || []}
      tags={facets.tags || []} searchSuggestions={searchRemoteTags} onSearch={searchCommunity}
      sidebarActions={<><h3>{sourceName || 'Shared community'}</h3><button className="media-collection-link collection-sidebar-action" onClick={onBack}>← Back to Collections</button><Link className="media-collection-link" to="/settings/fediverse" state={{ returnTo: settingsReturnTo }}>Community settings</Link></>}
      sidebarCollections={<RemoteCreatorCollections sourceId={sourceId} mediaType={mediaType} selectedCreatorId={filters.creatorId} localCopies={filters.offlineOnly} view={creatorView} query={creatorQuery} onDirectoryChange={onCreatorDirectoryChange} onSelectCreator={(id, offline) => { closeImage(); drawer.close(); onCreatorSelect(id, offline) }} />}
      mediaType={mediaType} mobileOpen={drawer.isOpen} onClose={() => { drawer.close(); setSidebarHover(false) }}
      onTagClick={onTag} selectedImage={selected} lightboxMode={Boolean(selected)} lightboxHover={sidebarHover} onMouseLeave={() => setSidebarHover(false)} />
    <main ref={contentRef} className="content with-sidebar shared-gallery-page" onScrollCapture={event => { if (result.scope === scope && event.target.classList?.contains('masonry-container')) sessionStorage.setItem(`${sessionKey}_scroll`, String(event.target.scrollTop)) }}>
      <button className="menu-btn mobile-only collection-mobile-menu" onClick={drawer.open} aria-label="Open menu"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12h18M3 6h18M3 18h18" /></svg></button>
      {current.offline ? <p className="shared-gallery-notice" role="status">{current.partial ? 'Offline copies · Some content is not available on this device.' : 'Offline copies · Showing downloaded content.'}</p> : current.stale && <p className="shared-gallery-notice" role="status">Offline · showing saved results.</p>}
      {current.error && <div className="shared-gallery-notice" role="alert">{current.error} <button onClick={() => loadPage(current.page ? current.page + 1 : 1, current.page > 0)}>Retry</button></div>}
      <div className="floating-controls"><GalleryTileSizeControl tileSize={tileSize} onChange={setTileSize} /></div>
      <MasonryGrid images={images} onImageClick={openImage} onLoadMore={loadMore} loading={current.loading || restoring} hasMore={current.hasMore && !current.error} tileSize={tileSize} />
    </main>
  </div>
    {selectedIndex >= 0 && <Lightbox images={images} currentIndex={selectedIndex} total={images.length} onClose={closeImage} onNav={navigateImage}
      onTagClick={tag => { closeImage(); onTag(tag) }} onImageUpdate={() => {}}
      onSidebarHover={setSidebarHover} sidebarOpen={drawer.isOpen || sidebarHover} remoteActions={remoteActions} />}
  </div>
}
