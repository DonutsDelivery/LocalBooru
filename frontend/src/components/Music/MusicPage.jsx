import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import SidebarNavigation from '../SidebarNavigation'
import {
  addMusicCollectionItem, createMusicCollection, fetchLibraries, fetchMusicAlbum,
  fetchMusicAlbums, fetchMusicCollection, fetchMusicCollections, fetchMusicFacets,
  fetchMusicTracks, getMediaUrl, removeMusicCollectionItem, setMusicFavorite,
} from '../../api'
import { useMusicPlayer } from './MusicPlayer'
import PersistentMusicPlayer from './PersistentMusicPlayer'
import './Music.css'

const defaultFacets = { artists: [], albums: [], genres: [], years: [], folders: [] }
const labelFor = (value, fallback) => value || fallback
const itemKey = item => `${item.library_id || ''}:${item.id}`
const musicItemTitle = (item, mode) => mode === 'albums' ? item.display_title || item.title : item.title
function matchesCollectionFilters(item, filters) {
  const query = filters.query.trim().toLocaleLowerCase()
  if (query && ![item.display_title, item.title, item.artist, item.album].some(value => value?.toLocaleLowerCase().includes(query))) return false
  if (filters.artist && item.artist !== filters.artist) return false
  if (filters.album && item.album !== filters.album && item.title !== filters.album) return false
  if (filters.genre && item.genre !== filters.genre) return false
  if (filters.year && String(item.year) !== String(filters.year)) return false
  if (filters.favorites && !item.is_favorite) return false
  if (filters.folder && String(item.directory_id) !== String(filters.folder)) return false
  return true
}

function MusicArtwork({ item }) {
  return item.artwork_url
    ? <img src={getMediaUrl(item.artwork_url)} alt="" loading="lazy" />
    : <div className="music-card-fallback" aria-label="No artwork">♫</div>
}

function MusicFilterDropdown({ name, label, value, options, openFilter, setOpenFilter, onChange }) {
  const triggerRef = useRef(null)
  const [menuPosition, setMenuPosition] = useState(null)
  const open = openFilter === name
  const selected = options.find(option => String(option.value) === String(value)) || options[0]

  useLayoutEffect(() => {
    if (!open) return
    const positionMenu = () => {
      const rect = triggerRef.current?.getBoundingClientRect()
      if (!rect) return
      const below = window.innerHeight - rect.bottom - 8
      const above = rect.top - 8
      const placeAbove = below < 220 && above > below
      setMenuPosition({
        left: Math.max(8, Math.min(rect.left, window.innerWidth - rect.width - 8)),
        width: rect.width,
        maxHeight: Math.min(260, Math.max(80, placeAbove ? above : below)),
        top: placeAbove ? undefined : rect.bottom + 4,
        bottom: placeAbove ? window.innerHeight - rect.top + 4 : undefined,
      })
    }
    positionMenu()
    window.addEventListener('resize', positionMenu)
    window.addEventListener('scroll', positionMenu, true)
    return () => {
      window.removeEventListener('resize', positionMenu)
      window.removeEventListener('scroll', positionMenu, true)
    }
  }, [open])

  return <div className="music-filter-dropdown">
    <span className="music-filter-label">{label}</span>
    <button ref={triggerRef} type="button" className="music-filter-trigger"
      aria-label={`${label} filter`} aria-haspopup="listbox" aria-expanded={open}
      onClick={() => setOpenFilter(current => current === name ? null : name)}
      onKeyDown={event => {
        if (event.key === 'Escape') { setOpenFilter(null); triggerRef.current?.focus() }
        if (event.key === 'ArrowDown' && !open) { event.preventDefault(); setOpenFilter(name) }
      }}>
      <span>{selected.label}</span><span className="music-filter-chevron" aria-hidden="true">▾</span>
    </button>
    {open && menuPosition && createPortal(<div className="music-filter-options" role="listbox" aria-label={label} style={menuPosition}
      onKeyDown={event => {
        if (event.key === 'Escape') { setOpenFilter(null); triggerRef.current?.focus() }
      }}>
      {options.map(option => <button key={String(option.value)} type="button" role="option"
        aria-selected={String(option.value) === String(value)}
        onClick={() => { onChange(option.value); setOpenFilter(null); triggerRef.current?.focus() }}>
        {option.label}
      </button>)}
    </div>, document.body)}
  </div>
}

export default function MusicPage() {
  const { browse, setBrowse, updateBrowse, startAlbum, startSong, openAlbum, queueTrack, session } = useMusicPlayer()
  const mode = browse.mode
  const filters = browse.byMode[mode]
  const scrollRef = useRef(null)
  const loadMoreRef = useRef(null)
  const loadedQueryRef = useRef(null)
  const paginationPendingRef = useRef(false)
  const albumPlayRequestRef = useRef(0)
  const restoredRef = useRef('')
  const [albums, setAlbums] = useState([])
  const [tracks, setTracks] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [facets, setFacets] = useState(defaultFacets)
  const [collections, setCollections] = useState([])
  const [libraries, setLibraries] = useState([])
  const [newCollection, setNewCollection] = useState('')
  const [addingCollection, setAddingCollection] = useState(false)
  const [collectionTarget, setCollectionTarget] = useState('')
  const [openFilter, setOpenFilter] = useState(null)
  const [directMembers, setDirectMembers] = useState(new Set())
  const [refresh, setRefresh] = useState(0)

  const { scroll: savedScroll, page = 1, ...queryFilters } = filters
  const queryKey = JSON.stringify({ mode, filters: queryFilters, page, refresh })
  const activeItems = mode === 'albums' ? albums : tracks

  useEffect(() => {
    fetchLibraries().then(data => setLibraries(data.libraries || [])).catch(() => {})
  }, [])

  useEffect(() => {
    const closeOutside = event => {
      if (!event.target.closest?.('.music-filter-dropdown, .music-filter-options')) setOpenFilter(null)
    }
    document.addEventListener('pointerdown', closeOutside)
    return () => document.removeEventListener('pointerdown', closeOutside)
  }, [])

  useEffect(() => {
    let alive = true
    Promise.allSettled([fetchMusicFacets({ library_id: filters.library || undefined }), fetchMusicCollections(filters.library || undefined)]).then(results => {
      if (!alive) return
      if (results[0].status === 'fulfilled') setFacets({ ...defaultFacets, ...results[0].value })
      if (results[1].status === 'fulfilled') setCollections(results[1].value.collections || [])
    })
    return () => { alive = false }
  }, [filters.library, refresh])

  useEffect(() => {
    let alive = true
    loadedQueryRef.current = null
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        if (filters.collection) {
          const result = await fetchMusicCollection(filters.collection, filters.library || undefined)
          let collectionAlbums = result.albums || []
          let collectionTracks = result.tracks || []
          const direct = new Set([
            ...collectionAlbums.map(album => `album:${itemKey(album)}`),
            ...collectionTracks.map(track => `track:${itemKey(track)}`),
          ])
          if (mode === 'songs' && collectionAlbums.length) {
            const details = await Promise.all(collectionAlbums.map(album => fetchMusicAlbum(album.id, album.library_id)))
            collectionTracks = [...collectionTracks, ...details.flatMap(detail => detail.tracks || [])]
          }
          if (mode === 'albums' && collectionTracks.length) {
            const ids = new Set(collectionAlbums.map(itemKey))
            const hasNamedAlbum = track => track.album_id && track.album?.trim().toLowerCase() !== 'unknown album'
            const albumRefs = collectionTracks.filter(hasNamedAlbum).map(track => ({
              id: track.album_id, title: track.album, artist: track.artist,
              artwork_url: track.artwork_url, library_id: track.library_id,
            }))
            for (const album of albumRefs) if (!ids.has(itemKey(album))) { ids.add(itemKey(album)); collectionAlbums.push(album) }
            // A track with no album still needs an Albums-mode entry so its
            // collection never disappears when switching music views.
            for (const track of collectionTracks.filter(track => !hasNamedAlbum(track))) {
              collectionAlbums.push({
                id: `single-${track.id}`, title: track.title || 'Untitled track',
                artist: track.artist, artwork_url: track.artwork_url,
                library_id: track.library_id, track_count: 1, genre: track.genre,
                year: track.year, directory_id: track.directory_id,
                is_favorite: track.is_favorite, _standaloneTrack: track,
              })
            }
          }
          const unique = (items) => [...new Map(items.map(item => [itemKey(item), item])).values()]
          collectionAlbums = unique(collectionAlbums)
          collectionTracks = unique(collectionTracks)
          if (!alive) return
          setDirectMembers(direct)
          const visibleAlbums = collectionAlbums.filter(album => matchesCollectionFilters(album, filters))
          const visibleTracks = collectionTracks.filter(track => matchesCollectionFilters(track, filters))
          setAlbums(visibleAlbums)
          setTracks(visibleTracks)
          setTotal(mode === 'albums' ? visibleAlbums.length : visibleTracks.length)
        } else {
          const params = {
            q: filters.query || undefined, artist: filters.artist || undefined,
            album: filters.album || undefined, genre: filters.genre || undefined,
            year: filters.year || undefined, favorites_only: filters.favorites || undefined,
            directory_id: filters.folder || undefined, library_id: filters.library || undefined,
            per_page: 60,
          }
          const results = await Promise.all(Array.from({ length: page }, (_, index) =>
            mode === 'albums'
              ? fetchMusicAlbums({ ...params, page: index + 1 })
              : fetchMusicTracks({ ...params, page: index + 1 })
          ))
          if (!alive) return
          if (mode === 'albums') setAlbums(results.flatMap(result => result.albums || []))
          else setTracks(results.flatMap(result => result.tracks || []))
          setTotal(results[0]?.total || 0)
        }
        loadedQueryRef.current = queryKey
      } catch (cause) {
        if (alive) setError(cause.response?.data?.detail || 'Could not load the music library.')
      } finally {
        if (alive) {
          paginationPendingRef.current = false
          setLoading(false)
        }
      }
    }
    load()
    return () => { alive = false }
  // queryKey captures changes to all filters, page, mode, and refresh.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryKey])

  useEffect(() => {
    if (loading || !activeItems.length) return
    const marker = `${mode}:${filters.library}:${filters.collection}:${page}:${activeItems.length}`
    if (restoredRef.current === marker) return
    restoredRef.current = marker
    requestAnimationFrame(() => { if (scrollRef.current) scrollRef.current.scrollTop = savedScroll || 0 })
  }, [mode, filters.library, filters.collection, page, activeItems.length, loading, savedScroll])

  useEffect(() => {
    const root = scrollRef.current
    const marker = loadMoreRef.current
    if (!root || !marker || loading || error || filters.collection
        || !activeItems.length || activeItems.length >= total
        || loadedQueryRef.current !== queryKey) return
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting || paginationPendingRef.current || loadedQueryRef.current !== queryKey) return
      paginationPendingRef.current = true
      updateBrowse(mode, { page: page + 1, scroll: root.scrollTop })
    }, { root, rootMargin: '600px 0px' })
    observer.observe(marker)
    return () => observer.disconnect()
  }, [activeItems.length, error, filters.collection, loading, mode, page, queryKey, total, updateBrowse])

  const updateFilter = (key, value) => {
    updateBrowse(mode, { [key]: value, scroll: 0, page: 1 })
    if (scrollRef.current) scrollRef.current.scrollTop = 0
  }

  const switchMode = nextMode => {
    if (nextMode === mode) return
    if (scrollRef.current) updateBrowse(mode, { scroll: scrollRef.current.scrollTop })
    setBrowse(previous => ({ ...previous, mode: nextMode }))
  }

  const openItem = useCallback(async item => {
    albumPlayRequestRef.current += 1
    if (mode === 'songs') { startSong(item); return }
    if (item._standaloneTrack) { openAlbum(item, [item._standaloneTrack]); return }
    try {
      const detail = await fetchMusicAlbum(item.id, item.library_id)
      openAlbum(detail.album || item, detail.tracks || [])
    } catch { setError('Could not open this album.') }
  }, [mode, openAlbum, startSong])

  const playAlbum = useCallback(async item => {
    const request = ++albumPlayRequestRef.current
    try {
      const detail = item._standaloneTrack
        ? { album: item, tracks: [item._standaloneTrack] }
        : await fetchMusicAlbum(item.id, item.library_id)
      if (request !== albumPlayRequestRef.current) return
      const tracks = detail.tracks || []
      if (!tracks.length) {
        setError('This album has no playable tracks.')
        return
      }
      startAlbum(detail.album || item, tracks, null, { openViewer: false })
    } catch { if (request === albumPlayRequestRef.current) setError('Could not play this album.') }
  }, [startAlbum])

  const createCollection = async event => {
    event.preventDefault()
    const name = newCollection.trim()
    if (!name) return
    setAddingCollection(true)
    try {
      const created = await createMusicCollection(name, filters.library || undefined)
      setCollections(previous => [...previous, created.collection || created])
      setNewCollection('')
    } catch { setError('Could not create the collection.') }
    finally { setAddingCollection(false) }
  }

  const changeCollectionMembership = async (item, add) => {
    if (!filters.collection && !collectionTarget) return
    const collectionId = filters.collection || collectionTarget
    const member = item._standaloneTrack || item
    const memberType = item._standaloneTrack ? 'track' : mode === 'albums' ? 'album' : 'track'
    try {
      if (add) await addMusicCollectionItem(collectionId, memberType, member.id, member.library_id)
      else await removeMusicCollectionItem(collectionId, memberType, member.id, member.library_id)
      setRefresh(value => value + 1)
    } catch { setError('Could not update the collection.') }
  }

  const toggleFavorite = async (event, track) => {
    event.stopPropagation()
    try {
      await setMusicFavorite(track, !track.is_favorite)
      setTracks(previous => previous.map(item => itemKey(item) === itemKey(track) ? { ...item, is_favorite: !track.is_favorite } : item))
      if (filters.favorites) setRefresh(value => value + 1)
    } catch { setError('Could not update the favorite.') }
  }

  return <div className="music-page">
    <aside className="music-sidebar">
      <div className="music-sidebar-content">
      <SidebarNavigation />
      <h1>Music</h1>
      <p>Your local albums and songs</p>
      <div className="music-side-section">
        <h2>Browse</h2>
        <button className={!filters.collection ? 'active' : ''} onClick={() => updateFilter('collection', '')}>All music</button>
        <div className="music-collections-title">Collections</div>
        {collections.map(collection => <button key={collection.id} className={String(filters.collection) === String(collection.id) ? 'active' : ''}
          onClick={() => updateFilter('collection', String(collection.id))}>{collection.name} <small>{collection.item_count ?? ''}</small></button>)}
        <form className="music-new-collection" onSubmit={createCollection}>
          <input value={newCollection} onChange={event => setNewCollection(event.target.value)} placeholder="New collection" aria-label="New music collection" />
          <button disabled={addingCollection || !newCollection.trim()} aria-label="Create music collection">＋</button>
        </form>
      </div>
      <div className="music-side-section music-filters">
        <h2>Filters</h2>
        {libraries.length > 1 && <MusicFilterDropdown name="library" label="Library" value={filters.library} options={[{ value: '', label: 'Primary library' }, ...libraries.map(library => ({ value: library.uuid || library.id, label: library.name }))]} openFilter={openFilter} setOpenFilter={setOpenFilter} onChange={value => updateFilter('library', value)} />}
        <MusicFilterDropdown name="artist" label="Artist" value={filters.artist} options={[{ value: '', label: 'All artists' }, ...facets.artists.map(value => ({ value, label: value }))]} openFilter={openFilter} setOpenFilter={setOpenFilter} onChange={value => updateFilter('artist', value)} />
        <MusicFilterDropdown name="album" label="Album" value={filters.album} options={[{ value: '', label: 'All albums' }, ...facets.albums.map(value => ({ value, label: value }))]} openFilter={openFilter} setOpenFilter={setOpenFilter} onChange={value => updateFilter('album', value)} />
        <MusicFilterDropdown name="genre" label="Genre" value={filters.genre} options={[{ value: '', label: 'All genres' }, ...facets.genres.map(value => ({ value, label: value }))]} openFilter={openFilter} setOpenFilter={setOpenFilter} onChange={value => updateFilter('genre', value)} />
        <MusicFilterDropdown name="year" label="Year" value={filters.year} options={[{ value: '', label: 'All years' }, ...facets.years.map(value => ({ value, label: String(value) }))]} openFilter={openFilter} setOpenFilter={setOpenFilter} onChange={value => updateFilter('year', value)} />
        <MusicFilterDropdown name="folder" label="Folder" value={filters.folder} options={[{ value: '', label: 'All folders' }, ...facets.folders.map(folder => ({ value: folder.id, label: folder.name }))]} openFilter={openFilter} setOpenFilter={setOpenFilter} onChange={value => updateFilter('folder', value)} />
        <label className="music-check"><input type="checkbox" checked={filters.favorites} onChange={event => updateFilter('favorites', event.target.checked)} /> Favorites</label>
      </div>
      </div>
      <PersistentMusicPlayer />
    </aside>
    <main className="music-main">
      <div className="music-toolbar">
        <div className="music-mode-switch" role="group" aria-label="Music view">
          <button className={mode === 'albums' ? 'active' : ''} onClick={() => switchMode('albums')}>Albums</button>
          <button className={mode === 'songs' ? 'active' : ''} onClick={() => switchMode('songs')}>Songs</button>
        </div>
        <input type="search" value={filters.query} onChange={event => updateFilter('query', event.target.value)} placeholder={`Search ${mode}`} aria-label={`Search ${mode}`} />
        <span className="music-result-count">{total.toLocaleString()} {mode}</span>
      </div>
      {error && <div className="music-error" role="alert">{error} <button onClick={() => setRefresh(value => value + 1)}>Retry</button></div>}
      <div className="music-scroll" ref={scrollRef} onScroll={event => {
        const top = event.currentTarget.scrollTop
        // Keep scroll position in provider so route switches restore this view.
        if (Math.abs(top - (filters.scroll || 0)) > 60) updateBrowse(mode, { scroll: top })
      }}>
        {!loading && !activeItems.length && <div className="music-empty"><span>♫</span><h2>No {mode} found</h2><p>Try another search or add a music folder in Directories.</p></div>}
        <div className="music-masonry">
          {activeItems.map(item => <article className="music-card" key={itemKey(item)}>
            <button className="music-card-open" onClick={() => openItem(item)} aria-label={`Open ${musicItemTitle(item, mode)}`}>
              <MusicArtwork item={item} />
              <span className="music-card-details"><strong>{labelFor(musicItemTitle(item, mode), mode === 'albums' ? 'Unknown album' : 'Untitled track')}</strong><small>{labelFor(item.artist, 'Unknown artist')}</small></span>
            </button>
            <div className="music-card-actions">
              {mode === 'albums' && <button className="music-card-play" onClick={() => playAlbum(item)} aria-label={`Play ${musicItemTitle(item, mode)}`} title="Play album">▶</button>}
              {mode === 'songs' && <button onClick={event => toggleFavorite(event, item)} aria-label={item.is_favorite ? 'Remove favorite' : 'Add favorite'} title="Favorite">{item.is_favorite ? '♥' : '♡'}</button>}
              {session && <button onClick={() => mode === 'songs' ? queueTrack(item) : openItem(item)} aria-label={mode === 'songs' ? `Queue ${item.title}` : `Open ${musicItemTitle(item, mode)}`} title={mode === 'songs' ? 'Add to queue' : 'Open album'}>{mode === 'songs' ? '＋ Queue' : 'Tracks'}</button>}
              {filters.collection ? directMembers.has(`${item._standaloneTrack ? 'track' : mode === 'albums' ? 'album' : 'track'}:${itemKey(item._standaloneTrack || item)}`)
                ? <button onClick={() => changeCollectionMembership(item, false)} title="Remove from collection">Remove</button>
                : <small>{mode === 'songs' ? 'From album' : 'Contains a collection song'}</small>
                : collections.length > 0 && <span className="music-add-to-collection"><select value={collectionTarget} onChange={event => setCollectionTarget(event.target.value)} aria-label="Choose music collection"><option value="">Collection…</option>{collections.map(collection => <option value={collection.id} key={collection.id}>{collection.name}</option>)}</select><button disabled={!collectionTarget} onClick={() => changeCollectionMembership(item, true)} aria-label="Add to collection">＋</button></span>}
            </div>
          </article>)}
        </div>
        {!filters.collection && activeItems.length < total && <div ref={loadMoreRef} className="music-load-sentinel" aria-live="polite">{loading && activeItems.length ? 'Loading more music…' : ''}</div>}
        {loading && !activeItems.length && <div className="music-loading">Loading music…</div>}
      </div>
    </main>
  </div>
}
