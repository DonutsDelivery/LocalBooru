import { useCallback, useEffect, useRef, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  addMusicCollectionItem, createMusicCollection, fetchLibraries, fetchMusicAlbum,
  fetchMusicAlbums, fetchMusicCollection, fetchMusicCollections, fetchMusicFacets,
  fetchMusicTracks, getMediaUrl, removeMusicCollectionItem, setMusicFavorite,
} from '../../api'
import { useMusicPlayer } from './MusicPlayer'
import './Music.css'

const defaultFacets = { artists: [], albums: [], genres: [], years: [], folders: [] }
const labelFor = (value, fallback) => value || fallback
const itemKey = item => `${item.library_id || ''}:${item.id}`
function matchesCollectionFilters(item, filters) {
  const query = filters.query.trim().toLocaleLowerCase()
  if (query && ![item.title, item.artist, item.album].some(value => value?.toLocaleLowerCase().includes(query))) return false
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

export default function MusicPage() {
  const { browse, setBrowse, updateBrowse, startSong, openAlbum, queueTrack, session } = useMusicPlayer()
  const mode = browse.mode
  const filters = browse.byMode[mode]
  const scrollRef = useRef(null)
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
  const [directMembers, setDirectMembers] = useState(new Set())
  const [refresh, setRefresh] = useState(0)

  const { scroll: savedScroll, page = 1, ...queryFilters } = filters
  const queryKey = JSON.stringify({ mode, filters: queryFilters, page, refresh })
  const activeItems = mode === 'albums' ? albums : tracks

  useEffect(() => {
    fetchLibraries().then(data => setLibraries(data.libraries || [])).catch(() => {})
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
            const albumRefs = collectionTracks.filter(track => track.album_id).map(track => ({
              id: track.album_id, title: track.album, artist: track.artist,
              artwork_url: track.artwork_url, library_id: track.library_id,
            }))
            for (const album of albumRefs) if (!ids.has(itemKey(album))) { ids.add(itemKey(album)); collectionAlbums.push(album) }
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
            page: 1, per_page: 60 * page,
          }
          const result = mode === 'albums' ? await fetchMusicAlbums(params) : await fetchMusicTracks(params)
          if (!alive) return
          if (mode === 'albums') setAlbums(result.albums || [])
          else setTracks(result.tracks || [])
          setTotal(result.total || 0)
        }
      } catch (cause) {
        if (alive) setError(cause.response?.data?.detail || 'Could not load the music library.')
      } finally {
        if (alive) setLoading(false)
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
    if (mode === 'songs') { startSong(item); return }
    try {
      const detail = await fetchMusicAlbum(item.id, item.library_id)
      openAlbum(detail.album || item, detail.tracks || [])
    } catch { setError('Could not open this album.') }
  }, [mode, openAlbum, startSong])

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
    try {
      if (add) await addMusicCollectionItem(collectionId, mode === 'albums' ? 'album' : 'track', item.id, item.library_id)
      else await removeMusicCollectionItem(collectionId, mode === 'albums' ? 'album' : 'track', item.id, item.library_id)
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
        {libraries.length > 1 && <label>Library<select value={filters.library} onChange={event => updateFilter('library', event.target.value)}>
          <option value="">Primary library</option>{libraries.map(library => <option key={library.uuid || library.id} value={library.uuid || library.id}>{library.name}</option>)}
        </select></label>}
        <label>Artist<select value={filters.artist} onChange={event => updateFilter('artist', event.target.value)}><option value="">All artists</option>{facets.artists.map(value => <option key={value}>{value}</option>)}</select></label>
        <label>Album<select value={filters.album} onChange={event => updateFilter('album', event.target.value)}><option value="">All albums</option>{facets.albums.map(value => <option key={value}>{value}</option>)}</select></label>
        <label>Genre<select value={filters.genre} onChange={event => updateFilter('genre', event.target.value)}><option value="">All genres</option>{facets.genres.map(value => <option key={value}>{value}</option>)}</select></label>
        <label>Year<select value={filters.year} onChange={event => updateFilter('year', event.target.value)}><option value="">All years</option>{facets.years.map(value => <option key={value}>{value}</option>)}</select></label>
        <label>Folder<select value={filters.folder} onChange={event => updateFilter('folder', event.target.value)}><option value="">All folders</option>{facets.folders.map(folder => <option key={folder.id} value={folder.id}>{folder.name}</option>)}</select></label>
        <label className="music-check"><input type="checkbox" checked={filters.favorites} onChange={event => updateFilter('favorites', event.target.checked)} /> Favorites</label>
      </div>
    </aside>
    <main className="music-main">
      <nav className="music-media-nav" aria-label="Media sections">
        <NavLink to="/" end>Images</NavLink><NavLink to="/videos">Videos</NavLink><NavLink to="/music">Music</NavLink>
      </nav>
      <div className="music-toolbar">
        <div className="music-mode-switch" role="group" aria-label="Music view">
          <button className={mode === 'albums' ? 'active' : ''} onClick={() => switchMode('albums')}>Albums</button>
          <button className={mode === 'songs' ? 'active' : ''} onClick={() => switchMode('songs')}>Songs</button>
        </div>
        <input type="search" value={filters.query} onChange={event => updateFilter('query', event.target.value)} placeholder={`Search ${mode}`} aria-label={`Search ${mode}`} />
        <span className="music-result-count">{total.toLocaleString()} {mode}</span>
      </div>
      {error && <div className="music-error" role="alert">{error}</div>}
      <div className="music-scroll" ref={scrollRef} onScroll={event => {
        const top = event.currentTarget.scrollTop
        // Keep scroll position in provider so route switches restore this view.
        if (Math.abs(top - (filters.scroll || 0)) > 60) updateBrowse(mode, { scroll: top })
      }}>
        {!loading && !activeItems.length && <div className="music-empty"><span>♫</span><h2>No {mode} found</h2><p>Try another search or add a music folder in Directories.</p></div>}
        <div className="music-masonry">
          {activeItems.map(item => <article className="music-card" key={itemKey(item)}>
            <button className="music-card-open" onClick={() => openItem(item)} aria-label={`Open ${item.title}`}>
              <MusicArtwork item={item} />
              <span className="music-card-details"><strong>{labelFor(item.title, mode === 'albums' ? 'Unknown album' : 'Untitled track')}</strong><small>{labelFor(item.artist, 'Unknown artist')}</small></span>
            </button>
            <div className="music-card-actions">
              {mode === 'songs' && <button onClick={event => toggleFavorite(event, item)} aria-label={item.is_favorite ? 'Remove favorite' : 'Add favorite'} title="Favorite">{item.is_favorite ? '♥' : '♡'}</button>}
              {session && <button onClick={() => mode === 'songs' ? queueTrack(item) : openItem(item)} aria-label={mode === 'songs' ? `Queue ${item.title}` : `Open ${item.title}`} title={mode === 'songs' ? 'Add to queue' : 'Open album'}>{mode === 'songs' ? '＋ Queue' : 'Tracks'}</button>}
              {filters.collection ? directMembers.has(`${mode === 'albums' ? 'album' : 'track'}:${itemKey(item)}`)
                ? <button onClick={() => changeCollectionMembership(item, false)} title="Remove from collection">Remove</button>
                : <small>{mode === 'songs' ? 'From album' : 'Contains a collection song'}</small>
                : collections.length > 0 && <span className="music-add-to-collection"><select value={collectionTarget} onChange={event => setCollectionTarget(event.target.value)} aria-label="Choose music collection"><option value="">Collection…</option>{collections.map(collection => <option value={collection.id} key={collection.id}>{collection.name}</option>)}</select><button disabled={!collectionTarget} onClick={() => changeCollectionMembership(item, true)} aria-label="Add to collection">＋</button></span>}
            </div>
          </article>)}
        </div>
        {!filters.collection && activeItems.length < total && <button className="music-load-more" disabled={loading} onClick={() => updateBrowse(mode, { page: page + 1 })}>{loading ? 'Loading…' : 'Load more'}</button>}
        {loading && !activeItems.length && <div className="music-loading">Loading music…</div>}
      </div>
    </main>
  </div>
}
