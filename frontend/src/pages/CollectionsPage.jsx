import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchCollections, createCollection, deleteCollection, getMediaUrl } from '../api'
import Sidebar from '../components/Sidebar'
import './CollectionsPage.css'

export default function CollectionsPage() {
  const navigate = useNavigate()
  const [collections, setCollections] = useState([])
  const [loading, setLoading] = useState(true)
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [creating, setCreating] = useState(false)

  useEffect(() => {
    loadCollections()
  }, [])

  async function loadCollections() {
    try {
      const data = await fetchCollections()
      setCollections(data.collections || [])
    } catch (e) {
      console.error('Failed to load collections:', e)
    }
    setLoading(false)
  }

  const handleCreate = async () => {
    if (!newName.trim() || creating) return
    setCreating(true)
    try {
      const result = await createCollection(newName.trim())
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
    if (!confirm('Delete this collection? Images will not be deleted.')) return
    try {
      await deleteCollection(id)
      setCollections(prev => prev.filter(c => c.id !== id))
    } catch (e) {
      console.error('Failed to delete:', e)
    }
  }

  return (
    <div className="app">
      <Sidebar />
      <main className="content with-sidebar">
        <div className="collections-header">
          <h1>Collections</h1>
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

        {loading ? (
          <div className="collections-loading">Loading collections...</div>
        ) : collections.length === 0 ? (
          <div className="collections-empty">
            <h2>No collections yet</h2>
            <p>Create a collection to organize your media into albums.</p>
          </div>
        ) : (
          <div className="collections-grid">
            {collections.map(c => (
              <div key={c.id} className="collection-card" onClick={() => navigate(`/collections/${c.id}`)}>
                <div className="collection-card-cover">
                  {c.cover_thumbnail_url ? (
                    <img src={getMediaUrl(c.cover_thumbnail_url)} alt="" loading="lazy" />
                  ) : (
                    <div className="collection-card-empty">
                      <svg viewBox="0 0 24 24" fill="currentColor"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>
                    </div>
                  )}
                  <button className="collection-card-delete" onClick={(e) => handleDelete(e, c.id)} title="Delete collection">
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
  )
}
