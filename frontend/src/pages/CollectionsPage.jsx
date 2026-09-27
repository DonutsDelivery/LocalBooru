import { useState, useEffect, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { fetchCollections, createCollection, deleteCollection, getMediaUrl } from '../api'
import Sidebar from '../components/Sidebar'
import { useMobileDrawer } from '../hooks/useMobileDrawer'
import './CollectionsPage.css'

export default function CollectionsPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const mediaType = searchParams.get('media_type') === 'video' ? 'video' : 'image'
  const [collections, setCollections] = useState([])
  const [loading, setLoading] = useState(true)
  const [loadedMediaType, setLoadedMediaType] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)
  const restoredScrollTypeRef = useRef(null)
  const drawer = useMobileDrawer()

  useEffect(() => {
    let active = true
    fetchCollections(mediaType)
      .then(data => {
        if (active) setCollections(data.collections || [])
      })
      .catch(error => console.error('Failed to load collections:', error))
      .finally(() => {
        if (active) {
          setLoadedMediaType(mediaType)
          setLoading(false)
        }
      })
    return () => { active = false }
  }, [mediaType])

  useEffect(() => {
    if (loading || loadedMediaType !== mediaType || restoredScrollTypeRef.current === mediaType) return
    restoredScrollTypeRef.current = mediaType
    requestAnimationFrame(() => {
      const grid = document.querySelector('.collections-page .collections-grid')
      if (grid) grid.scrollTop = Number(sessionStorage.getItem(`donutMediaCenter_collections_scroll_${mediaType}`) || 0)
    })
  }, [loading, loadedMediaType, mediaType])

  const handleCreate = async () => {
    if (!newName.trim() || creating) return
    setCreating(true)
    try {
      const result = await createCollection(newName.trim(), null, mediaType)
      setCollections(prev => [result, ...prev])
      setNewName('')
      setShowCreate(false)
    } catch (e) {
      console.error('Failed to create collection:', e)
    }
    setCreating(false)
  }

  const handleDelete = async (e, id) => {
    e.stopPropagation()
    if (!confirm(`Delete this collection? ${mediaType === 'video' ? 'Videos' : 'Images'} will not be deleted.`)) return
    try {
      await deleteCollection(id)
      setCollections(prev => prev.filter(c => c.id !== id))
    } catch (e) {
      console.error('Failed to delete:', e)
    }
  }

  return (
    <div className="app">
      <div className="main-container">
        {drawer.isOpen && <div className="sidebar-backdrop" onClick={drawer.close} />}
        <Sidebar mediaType={mediaType} mobileOpen={drawer.isOpen} onClose={drawer.close} />
        <main className="content with-sidebar collections-page">
        <div className="collections-header">
          <div className="collections-title-row">
            <button className="menu-btn mobile-only" onClick={drawer.open} aria-label="Open menu">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12h18M3 6h18M3 18h18"/>
              </svg>
            </button>
            <h1>{mediaType === 'video' ? 'Video' : 'Image'} Collections</h1>
          </div>
          <button className="collections-create-btn" onClick={() => setShowCreate(!showCreate)}>
            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>
            New Collection
          </button>
        </div>

        {showCreate && (
          <div className="collections-create-form">
            <input
              type="text"
              placeholder="Collection name..."
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleCreate() }}
              autoFocus
            />
            <button onClick={handleCreate} disabled={!newName.trim() || creating}>
              {creating ? 'Creating...' : 'Create'}
            </button>
            <button className="cancel" onClick={() => { setShowCreate(false); setNewName('') }}>Cancel</button>
          </div>
        )}

        {loading || loadedMediaType !== mediaType ? (
          <div className="collections-loading">Loading collections...</div>
        ) : collections.length === 0 ? (
          <div className="collections-empty">
            <h2>No collections yet</h2>
            <p>Create a collection to organize your {mediaType === 'video' ? 'videos' : 'images'}.</p>
          </div>
        ) : (
          <div className="collections-grid" onScroll={(event) => {
            sessionStorage.setItem(`donutMediaCenter_collections_scroll_${mediaType}`, String(event.currentTarget.scrollTop))
          }}>
            {collections.map(c => (
              <div
                key={c.id}
                className="collection-card"
                role="button"
                tabIndex={0}
                onClick={() => navigate(`/collections/${c.id}?media_type=${mediaType}`)}
                onKeyDown={(event) => {
                  if (event.target !== event.currentTarget) return
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    navigate(`/collections/${c.id}?media_type=${mediaType}`)
                  }
                }}
              >
                <div className="collection-card-cover">
                  {c.cover_thumbnail_url ? (
                    <img src={getMediaUrl(c.cover_thumbnail_url)} alt="" loading="lazy" />
                  ) : (
                    <div className="collection-card-empty">
                      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>
                    </div>
                  )}
                  <button className="collection-card-delete" onClick={(e) => handleDelete(e, c.id)} title="Delete collection" aria-label={`Delete ${c.name}`}>
                    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
                  </button>
                </div>
                <div className="collection-card-info">
                  <span className="collection-card-name">{c.name}</span>
                  <span className="collection-card-count">{c.item_count} items</span>
                </div>
              </div>
            ))}
          </div>
        )}
        </main>
      </div>
    </div>
  )
}
