/**
 * DirectoriesPage - Directory management page
 * Extracted from App.jsx
 */
import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import ComfyUIConfigModal from '../components/ComfyUIConfigModal'
import { getLibraryStats, updateDirectory, tagUntagged, clearDirectoryTagQueue, fetchLibraries, addLibrary, mountLibrary, unmountLibrary, removeLibrary, listParentDirectories, removeParentDirectory } from '../api'
import { getDesktopAPI, isDesktopApp } from '../tauriAPI'
import { toast } from '../components/Toast'
import { useAddonStatus } from '../hooks/useAddonStatus'

// Helper to create composite key for directory (avoids ID collisions across libraries)
const makeDirKey = (dir) => `${dir.library_id || 'primary'}:${dir.id}`
const parseDirKey = (key) => {
  const idx = key.lastIndexOf(':')
  const libId = key.substring(0, idx)
  return {
    dirId: parseInt(key.substring(idx + 1)),
    libraryId: libId === 'primary' ? undefined : libId
  }
}

function DirectoriesPage() {
  const navigate = useNavigate()
  const [directories, setDirectories] = useState([])
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState({})
  const [pruning, setPruning] = useState({})
  const [comfyuiConfigDir, setComfyuiConfigDir] = useState(null)
  const [stats, setStats] = useState(null)
  const [relocating, setRelocating] = useState({})
  const [selectedDirs, setSelectedDirs] = useState(new Set())
  const [batchLoading, setBatchLoading] = useState(false)
  const [repairing, setRepairing] = useState({})
  const [taggingActive, setTaggingActive] = useState({})
  const { installed: taggerInstalled } = useAddonStatus('auto-tagger')
  const { installed: ageDetectorInstalled } = useAddonStatus('age-detector')
  const [libraries, setLibraries] = useState([])
  const [parentDirs, setParentDirs] = useState([])
  const [showAddLibrary, setShowAddLibrary] = useState(false)
  const [newLibraryPath, setNewLibraryPath] = useState('')
  const [newLibraryName, setNewLibraryName] = useState('')
  const [newLibraryCreateNew, setNewLibraryCreateNew] = useState(false)
  const [activeLibrary, setActiveLibrary] = useState(null) // null = all libraries

  const refreshDirectories = async () => {
    const { fetchDirectories } = await import('../api')
    const data = await fetchDirectories()
    const dirs = data.directories || []
    setDirectories(dirs)
    // Sync tagging button state with actual queue
    const activeState = {}
    for (const dir of dirs) {
      if (dir.pending_tag_tasks > 0) {
        activeState[makeDirKey(dir)] = true
      }
    }
    setTaggingActive(activeState)
  }

  const refreshLibraries = async () => {
    try {
      const data = await fetchLibraries()
      setLibraries(data.libraries || [])
    } catch (e) {
      console.error('Failed to fetch libraries:', e)
    }
  }

  const refreshParentDirs = async () => {
    try {
      const data = await listParentDirectories()
      setParentDirs(data.parents || [])
    } catch (e) {
      console.error('Failed to fetch parent directories:', e)
    }
  }

  useEffect(() => {
    refreshDirectories()
      .catch(console.error)
      .finally(() => setLoading(false))
    getLibraryStats().then(setStats).catch(console.error)
    refreshLibraries()
    refreshParentDirs()
  }, [])

  // Resolve active library UUID (primary lib uses its real UUID, not 'primary')
  const activeLibraryUuid = activeLibrary === 'primary'
    ? libraries.find(l => l.is_primary)?.uuid
    : activeLibrary

  // Filtered lists based on active library tab
  const filteredDirectories = activeLibraryUuid
    ? directories.filter(d => d.library_id === activeLibraryUuid)
    : directories
  const filteredParentDirs = activeLibraryUuid
    ? parentDirs.filter(p => p.library_id === activeLibraryUuid)
    : parentDirs

  const handleAddDirectory = async () => {
    const api = getDesktopAPI()
    if (api?.addDirectory) {
      const path = await api.addDirectory()
      if (path) {
        const { addDirectory } = await import('../api')
        const libraryId = activeLibraryUuid && !libraries.find(l => l.uuid === activeLibraryUuid)?.is_primary ? activeLibraryUuid : undefined
        await addDirectory(path, { library_id: libraryId })
        await refreshDirectories()
      }
    } else {
      toast.warning('Directory picker only available in desktop app')
    }
  }

  const handleAddParentDirectory = async () => {
    const api = getDesktopAPI()
    if (api?.addDirectory) {
      const path = await api.addDirectory()
      if (path) {
        const { addParentDirectory } = await import('../api')
        const libraryId = activeLibraryUuid && !libraries.find(l => l.uuid === activeLibraryUuid)?.is_primary ? activeLibraryUuid : undefined
        const result = await addParentDirectory(path, { library_id: libraryId })
        toast.success(result.message)
        await refreshDirectories()
        await refreshParentDirs()
      }
    } else {
      toast.warning('Directory picker only available in desktop app')
    }
  }

  const handleRemoveParent = async (parent, removeChildren) => {
    try {
      const result = await removeParentDirectory(parent.path, {
        removeChildren,
        libraryId: parent.library_id
      })
      toast.success(result.message)
      await refreshParentDirs()
      await refreshDirectories()
    } catch (error) {
      toast.error(`Failed to remove parent: ${error.message}`)
    }
  }

  const handleRescan = async (dir) => {
    const key = makeDirKey(dir)
    setScanning(prev => ({ ...prev, [key]: true }))
    try {
      const { scanDirectory } = await import('../api')
      await scanDirectory(dir.id, dir.library_id)
      await refreshDirectories()
    } catch (error) {
      console.error('Scan failed:', error)
      toast.error('Scan failed: ' + error.message)
    } finally {
      setScanning(prev => ({ ...prev, [key]: false }))
    }
  }

  const handleRemove = async (dir) => {
    const dirName = dir.name || dir.path
    if (!confirm(`Remove "${dirName}" from watch list?\n\nImages will be removed from library.\nActual files on disk will NOT be deleted.`)) {
      return
    }
    try {
      const { removeDirectory } = await import('../api')
      await removeDirectory(dir.id, false, dir.library_id)
      await refreshDirectories()
    } catch (error) {
      console.error('Remove failed:', error)
      toast.error('Remove failed: ' + error.message)
    }
  }

  const handlePrune = async (dir) => {
    const key = makeDirKey(dir)
    const dirName = dir.name || dir.path
    const nonFavorited = dir.image_count - dir.favorited_count
    const savedDumpsterPath = localStorage.getItem('localbooru_dumpster_path') || null
    const dumpsterInfo = savedDumpsterPath ? `\nDumpster: ${savedDumpsterPath}` : ''
    if (!confirm(`Prune "${dirName}"?\n\nThis will move ${nonFavorited} non-favorited images to the dumpster folder.\nFavorited images (${dir.favorited_count}) will be kept.${dumpsterInfo}`)) {
      return
    }
    setPruning(prev => ({ ...prev, [key]: true }))
    try {
      const { pruneDirectory } = await import('../api')
      const result = await pruneDirectory(dir.id, savedDumpsterPath, dir.library_id)
      toast.success(`Pruned ${result.pruned} images to:\n${result.dumpster_path}`)
      await refreshDirectories()
      getLibraryStats().then(setStats).catch(console.error)
    } catch (error) {
      console.error('Prune failed:', error)
      toast.error('Prune failed: ' + error.message)
    } finally {
      setPruning(prev => ({ ...prev, [key]: false }))
    }
  }

  const handleRelocate = async (dir) => {
    const key = makeDirKey(dir)
    const api = getDesktopAPI()
    if (api?.addDirectory) {
      const newPath = await api.addDirectory()
      if (newPath && newPath !== dir.path) {
        if (!confirm(`Update directory location?\n\nFrom: ${dir.path}\nTo: ${newPath}\n\nThis will update all file references.`)) {
          return
        }
        setRelocating(prev => ({ ...prev, [key]: true }))
        try {
          const { updateDirectoryPath } = await import('../api')
          const result = await updateDirectoryPath(dir.id, newPath, dir.library_id)
          toast.success(`Directory relocated.\n${result.files_updated} file references updated.`)
          await refreshDirectories()
        } catch (error) {
          console.error('Relocate failed:', error)
          toast.error('Relocate failed: ' + (error.response?.data?.detail || error.message))
        } finally {
          setRelocating(prev => ({ ...prev, [key]: false }))
        }
      }
    } else {
      toast.warning('Directory picker only available in desktop app')
    }
  }

  // Selection handlers
  const toggleSelectDir = (dirId) => {
    setSelectedDirs(prev => {
      const newSet = new Set(prev)
      if (newSet.has(dirId)) {
        newSet.delete(dirId)
      } else {
        newSet.add(dirId)
      }
      return newSet
    })
  }

  const selectAllDirs = () => {
    setSelectedDirs(new Set(filteredDirectories.map(d => makeDirKey(d))))
  }

  const clearSelection = () => {
    setSelectedDirs(new Set())
  }

  // Batch action handlers
  const handleBatchRescan = async () => {
    if (selectedDirs.size === 0) return
    setBatchLoading(true)
    const { scanDirectory } = await import('../api')
    const dirKeys = Array.from(selectedDirs)

    // Mark all as scanning
    setScanning(prev => {
      const next = { ...prev }
      dirKeys.forEach(key => next[key] = true)
      return next
    })

    try {
      // Run rescans in parallel
      await Promise.all(dirKeys.map(key => {
        const { dirId, libraryId } = parseDirKey(key)
        return scanDirectory(dirId, libraryId).catch(e => {
          console.error(`Scan failed for ${key}:`, e)
        })
      }))
      await refreshDirectories()
    } finally {
      setScanning({})
      setBatchLoading(false)
      clearSelection()
    }
  }

  const handleBatchPrune = async () => {
    if (selectedDirs.size === 0) return
    const selectedList = directories.filter(d => selectedDirs.has(makeDirKey(d)))
    const totalNonFavorited = selectedList.reduce((sum, d) => sum + (d.image_count - d.favorited_count), 0)
    const totalFavorited = selectedList.reduce((sum, d) => sum + d.favorited_count, 0)
    const savedDumpsterPath = localStorage.getItem('localbooru_dumpster_path') || null
    const dumpsterInfo = savedDumpsterPath ? `\nDumpster: ${savedDumpsterPath}` : ''

    if (!confirm(`Prune ${selectedDirs.size} directories?\n\nThis will move ${totalNonFavorited} non-favorited images to the dumpster folder.\nFavorited images (${totalFavorited}) will be kept.${dumpsterInfo}`)) {
      return
    }

    setBatchLoading(true)
    const { pruneDirectory } = await import('../api')
    const dirKeys = Array.from(selectedDirs)

    // Mark all as pruning
    setPruning(prev => {
      const next = { ...prev }
      dirKeys.forEach(key => next[key] = true)
      return next
    })

    try {
      let totalPruned = 0
      for (const key of dirKeys) {
        try {
          const { dirId, libraryId } = parseDirKey(key)
          const result = await pruneDirectory(dirId, savedDumpsterPath, libraryId)
          totalPruned += result.pruned
        } catch (e) {
          console.error(`Prune failed for ${key}:`, e)
        }
      }
      toast.success(`Pruned ${totalPruned} images total`)
      await refreshDirectories()
      getLibraryStats().then(setStats).catch(console.error)
    } finally {
      setPruning({})
      setBatchLoading(false)
      clearSelection()
    }
  }

  const handleBatchRemove = async () => {
    if (selectedDirs.size === 0) return
    const selectedList = directories.filter(d => selectedDirs.has(makeDirKey(d)))
    const totalImages = selectedList.reduce((sum, d) => sum + (d.image_count || 0), 0)

    // Only show first 5 names to avoid huge dialogs
    const maxNames = 5
    const namesList = selectedList.slice(0, maxNames).map(d => d.name || d.path)
    const remaining = selectedList.length - maxNames
    let namesDisplay = '- ' + namesList.join('\n- ')
    if (remaining > 0) {
      namesDisplay += `\n... and ${remaining} more`
    }

    if (!confirm(`Remove ${selectedDirs.size} directories (${totalImages.toLocaleString()} images) from watch list?\n\n${namesDisplay}\n\nImages will be removed from library.\nActual files on disk will NOT be deleted.\n\nThis may take a while for large libraries.`)) {
      return
    }

    setBatchLoading(true)

    try {
      const { bulkDeleteDirectories } = await import('../api')
      // Group by library for separate bulk calls
      const byLibrary = {}
      for (const key of selectedDirs) {
        const { dirId, libraryId } = parseDirKey(key)
        const libKey = libraryId || ''
        if (!byLibrary[libKey]) byLibrary[libKey] = []
        byLibrary[libKey].push(dirId)
      }

      let totalDeleted = 0
      let totalImageCount = 0
      for (const [libKey, dirIds] of Object.entries(byLibrary)) {
        const libraryId = libKey || undefined
        console.log(`[Bulk Remove] Deleting ${dirIds.length} directories from library ${libKey || 'primary'}...`)
        const result = await bulkDeleteDirectories(dirIds, false, libraryId)
        totalDeleted += result.deleted
        totalImageCount += result.image_count || 0
      }
      console.log(`[Bulk Remove] Deleted ${totalDeleted} directories, ${totalImageCount} images`)

      await refreshDirectories()
    } catch (e) {
      console.error('Bulk remove failed:', e)
      toast.error(`Remove failed: ${e.response?.data?.detail || e.message || 'Unknown error'}`)
    } finally {
      setBatchLoading(false)
      clearSelection()
    }
  }

  const handleRepair = async (dir) => {
    const key = makeDirKey(dir)
    setRepairing(prev => ({ ...prev, [key]: true }))
    try {
      const { repairDirectoryPaths } = await import('../api')
      const result = await repairDirectoryPaths(dir.id, dir.library_id)
      toast.success(`Repair complete:\n${result.valid} files OK\n${result.repaired} paths fixed\n${result.removed} missing removed`)
      await refreshDirectories()
    } catch (error) {
      console.error('Repair failed:', error)
      toast.error('Repair failed: ' + (error.response?.data?.detail || error.message))
    } finally {
      setRepairing(prev => ({ ...prev, [key]: false }))
    }
  }

  const handleBatchRepair = async () => {
    if (selectedDirs.size === 0) return
    setBatchLoading(true)
    try {
      const { bulkRepairDirectories } = await import('../api')
      // Group by library for separate bulk calls
      const byLibrary = {}
      for (const key of selectedDirs) {
        const { dirId, libraryId } = parseDirKey(key)
        const libKey = libraryId || ''
        if (!byLibrary[libKey]) byLibrary[libKey] = []
        byLibrary[libKey].push(dirId)
      }

      let totalsValid = 0, totalsRepaired = 0, totalsRemoved = 0, totalsOrphan = 0
      for (const [libKey, dirIds] of Object.entries(byLibrary)) {
        const libraryId = libKey || undefined
        const result = await bulkRepairDirectories(dirIds, libraryId)
        totalsValid += result.totals.valid || 0
        totalsRepaired += result.totals.repaired || 0
        totalsRemoved += result.totals.removed || 0
        totalsOrphan += result.totals.orphan_thumbnails || 0
      }
      toast.success(`Batch repair complete:\n${totalsValid} files OK\n${totalsRepaired} paths fixed\n${totalsRemoved} missing removed${totalsOrphan > 0 ? `\n${totalsOrphan} orphan thumbnails cleaned` : ''}`)
      await refreshDirectories()
    } catch (e) {
      console.error('Batch repair failed:', e)
      toast.error(`Batch repair failed: ${e.response?.data?.detail || e.message || 'Unknown error'}`)
    } finally {
      setBatchLoading(false)
    }
  }

  const handleAddLibrary = async () => {
    if (!newLibraryPath.trim()) return
    try {
      await addLibrary(
        newLibraryPath.trim(),
        newLibraryName.trim() || newLibraryPath.trim().split('/').pop(),
        true,
        newLibraryCreateNew
      )
      setNewLibraryPath('')
      setNewLibraryName('')
      setNewLibraryCreateNew(false)
      setShowAddLibrary(false)
      await refreshLibraries()
      await refreshDirectories()
    } catch (e) {
      toast.error(e.response?.data?.message || e.message)
    }
  }

  const handleMountLibrary = async (uuid) => {
    try {
      await mountLibrary(uuid)
      await refreshLibraries()
      await refreshDirectories()
    } catch (e) {
      toast.error(e.response?.data?.message || e.message)
    }
  }

  const handleUnmountLibrary = async (uuid) => {
    try {
      await unmountLibrary(uuid)
      await refreshLibraries()
      await refreshDirectories()
    } catch (e) {
      toast.error(e.response?.data?.message || e.message)
    }
  }

  const handleRemoveLibrary = async (uuid) => {
    if (!confirm('Remove this library? (Files will not be deleted)')) return
    try {
      await removeLibrary(uuid)
      await refreshLibraries()
      await refreshDirectories()
    } catch (e) {
      toast.error(e.response?.data?.message || e.message)
    }
  }

  return (
    <div className="app">
      <div className="main-container">
        <Sidebar stats={stats} />
        <main className="content with-sidebar">
          <div className="page directories-page">
            <div className="page-header">
              <button className="back-btn mobile-only" onClick={() => navigate('/')} aria-label="Back to gallery">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M19 12H5M12 19l-7-7 7-7"/>
                </svg>
              </button>
              <h1>Watch Directories</h1>
            </div>
            <p>Add folders to automatically import and tag images.</p>

            <div className="directory-buttons">
              <button onClick={handleAddDirectory} className="add-directory-btn">
                + Add Directory
              </button>
              <button onClick={handleAddParentDirectory} className="add-directory-btn">
                + Add Parent Directory
              </button>
            </div>

            {/* Library Tabs */}
            <div className="library-tabs" style={{ display: 'flex', gap: '4px', marginBottom: '16px', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0', overflowX: 'auto' }}>
              <button
                className={`library-tab ${!activeLibrary ? 'active' : ''}`}
                onClick={() => setActiveLibrary(null)}
                style={{
                  padding: '8px 16px',
                  background: !activeLibrary ? 'var(--accent)' : 'transparent',
                  color: !activeLibrary ? '#fff' : 'var(--text-secondary)',
                  border: 'none',
                  borderRadius: '6px 6px 0 0',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  fontWeight: !activeLibrary ? 600 : 400,
                  whiteSpace: 'nowrap',
                }}
              >
                All Libraries
              </button>
              {libraries.filter(l => l.mounted).map(lib => {
                const isActive = activeLibrary === (lib.is_primary ? 'primary' : lib.uuid)
                return (
                <button
                  key={lib.uuid}
                  className={`library-tab ${isActive ? 'active' : ''}`}
                  onClick={() => setActiveLibrary(lib.is_primary ? 'primary' : lib.uuid)}
                  style={{
                    padding: '8px 16px',
                    background: isActive ? 'var(--accent)' : 'transparent',
                    color: isActive ? '#fff' : 'var(--text-secondary)',
                    border: 'none',
                    borderRadius: '6px 6px 0 0',
                    cursor: 'pointer',
                    fontSize: '0.9rem',
                    fontWeight: isActive ? 600 : 400,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {lib.name}
                </button>
              )})}
              <button
                onClick={() => setShowAddLibrary(!showAddLibrary)}
                style={{
                  padding: '8px 12px',
                  background: 'transparent',
                  color: 'var(--text-secondary)',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  marginLeft: 'auto',
                }}
              >
                {showAddLibrary ? 'Cancel' : '+'}
              </button>
            </div>

            {/* Add Library Form */}
            {showAddLibrary && (
              <div style={{ background: 'var(--bg-secondary)', padding: '12px', borderRadius: '8px', marginBottom: '16px' }}>
                <input
                  type="text"
                  placeholder="Path to folder containing library.db"
                  value={newLibraryPath}
                  onChange={e => setNewLibraryPath(e.target.value)}
                  style={{ width: '100%', marginBottom: '8px', padding: '6px 10px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }}
                />
                <input
                  type="text"
                  placeholder="Library name (optional)"
                  value={newLibraryName}
                  onChange={e => setNewLibraryName(e.target.value)}
                  style={{ width: '100%', marginBottom: '8px', padding: '6px 10px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-primary)', color: 'var(--text-primary)' }}
                />
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  <input
                    type="checkbox"
                    checked={newLibraryCreateNew}
                    onChange={e => setNewLibraryCreateNew(e.target.checked)}
                  />
                  Create new empty library at this path
                </label>
                <button onClick={handleAddLibrary} className="btn btn-primary btn-sm">
                  {newLibraryCreateNew ? 'Create Library' : 'Mount Library'}
                </button>
              </div>
            )}

            {/* Library management controls for the selected tab */}
            {activeLibrary && (() => {
              const lib = libraries.find(l => activeLibrary === 'primary' ? l.is_primary : l.uuid === activeLibrary)
              return lib && !lib.is_primary ? (
                <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{lib.path}</span>
                  {lib.mounted ? (
                    <button onClick={() => handleUnmountLibrary(lib.uuid)} className="btn btn-sm" style={{ fontSize: '0.8rem' }}>Unmount</button>
                  ) : (
                    <button onClick={() => handleMountLibrary(lib.uuid)} className="btn btn-primary btn-sm" style={{ fontSize: '0.8rem' }} disabled={!lib.accessible}>Mount</button>
                  )}
                  <button onClick={() => handleRemoveLibrary(lib.uuid)} className="btn btn-sm" style={{ fontSize: '0.8rem', color: 'var(--color-error, #e74c3c)' }}>Remove</button>
                  {!lib.accessible && <span style={{ fontSize: '0.8rem', color: 'var(--color-error, #e74c3c)' }}>Path not accessible</span>}
                </div>
              ) : null
            })()}

            {/* Unmounted libraries (shown only on "All" tab) */}
            {!activeLibrary && libraries.some(l => !l.mounted && !l.is_primary) && (
              <div style={{ marginBottom: '16px' }}>
                <h3 style={{ margin: '0 0 8px', fontSize: '0.95rem', color: 'var(--text-secondary)' }}>Unmounted Libraries</h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {libraries.filter(l => !l.mounted && !l.is_primary).map(lib => (
                    <div key={lib.uuid} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: 'var(--bg-secondary)', borderRadius: '6px', opacity: 0.7 }}>
                      <div>
                        <span style={{ fontWeight: 500 }}>{lib.name}</span>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginLeft: '8px' }}>{lib.path}</span>
                      </div>
                      <div style={{ display: 'flex', gap: '6px' }}>
                        <button onClick={() => handleMountLibrary(lib.uuid)} className="btn btn-primary btn-sm" style={{ fontSize: '0.8rem' }} disabled={!lib.accessible}>Mount</button>
                        <button onClick={() => handleRemoveLibrary(lib.uuid)} className="btn btn-sm" style={{ fontSize: '0.8rem', color: 'var(--color-error, #e74c3c)' }}>Remove</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Parent Directories (filtered by active library) */}
            {filteredParentDirs.length > 0 && (
              <div style={{ marginBottom: '24px' }}>
                <h2 style={{ margin: '0 0 12px', fontSize: '1.1rem' }}>Parent Directories</h2>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '12px' }}>
                  These folders are watched for new subdirectories, which are automatically added.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {filteredParentDirs.map(parent => (
                    <div
                      key={`${parent.library_id}:${parent.path}`}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        padding: '10px 14px',
                        background: 'var(--bg-secondary)',
                        borderRadius: '8px',
                        border: '1px solid var(--glass-border)',
                      }}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                        <line x1="12" y1="11" x2="12" y2="17"/>
                        <line x1="9" y1="14" x2="15" y2="14"/>
                      </svg>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: '0.9rem', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {parent.path.split('/').pop() || parent.path}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {parent.path} · {parent.child_count} {parent.child_count === 1 ? 'subdirectory' : 'subdirectories'}
                          {!activeLibrary && parent.library_name && <span> · {parent.library_name}</span>}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
                        <button
                          onClick={() => handleRemoveParent(parent, false)}
                          className="btn btn-sm"
                          style={{ fontSize: '0.75rem' }}
                          title="Stop watching for new subdirectories (keep existing)"
                        >
                          Stop Watching
                        </button>
                        <button
                          onClick={() => handleRemoveParent(parent, true)}
                          className="btn btn-sm"
                          style={{ fontSize: '0.75rem', color: 'var(--color-error, #e74c3c)' }}
                          title="Remove parent and all child directories"
                        >
                          Remove All
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {loading ? (
              <p>Loading...</p>
            ) : filteredDirectories.length === 0 ? (
              <p className="empty-state">{activeLibrary ? 'No directories in this library yet.' : 'No directories added yet. Add a folder to get started!'}</p>
            ) : (
              <>
              <div className="directory-list-header">
                <span className="directory-count">{filteredDirectories.length} directories</span>
                <div className="selection-buttons">
                  <button className="select-btn" onClick={selectAllDirs}>Select All</button>
                  <button className="select-btn" onClick={clearSelection} disabled={selectedDirs.size === 0}>Unselect All</button>
                </div>
              </div>
              <ul className="directory-list">
                {filteredDirectories.map(dir => {
                  const dirKey = `${dir.library_id || 'primary'}:${dir.id}`
                  return (
                  <li key={dirKey} className={`directory-item ${dir.enabled ? '' : 'disabled'} ${selectedDirs.has(dirKey) ? 'selected' : ''}`}>
                    {/* Header row: checkbox, name/path/count, status */}
                    <div className="directory-header">
                      <label className="directory-checkbox">
                        <input
                          type="checkbox"
                          checked={selectedDirs.has(dirKey)}
                          onChange={() => toggleSelectDir(dirKey)}
                        />
                      </label>
                      <div className="directory-info">
                        <strong>{dir.name}</strong>
                        <span className="directory-path">{dir.path}</span>
                        <span className="directory-stats">{dir.image_count} images</span>
                      </div>
                      <div className="directory-status">
                        {!dir.path_exists && <span className="warning">Path not found</span>}
                        {dir.enabled ? '✓ Active' : 'Disabled'}
                      </div>
                    </div>

                    {/* Stats row: read-only metrics */}
                    <div className="directory-stats-row">
                      <span className="stat" title="Images with age detection">
                        Age: {dir.age_detected_pct}%
                      </span>
                      <span className="stat" title="Images with booru tags">
                        Tagged: {dir.tagged_pct}%
                      </span>
                      <span className="stat" title="Favorited images">
                        Favorites: {dir.favorited_count}
                      </span>
                    </div>

                    {/* Toggles row: actionable settings */}
                    <div className="directory-toggles">
                      {taggerInstalled && (
                      <button
                        className={`toggle-btn tag-btn ${taggingActive[dirKey] ? 'active' : ''}`}
                        onClick={async () => {
                          const isActive = taggingActive[dirKey]
                          if (isActive) {
                            setTaggingActive(prev => ({ ...prev, [dirKey]: false }))
                            try {
                              await clearDirectoryTagQueue(dir.id)
                            } catch (err) {
                              console.error('Failed to clear queue:', err)
                            }
                          } else {
                            setTaggingActive(prev => ({ ...prev, [dirKey]: true }))
                            try {
                              await tagUntagged(dir.id)
                            } catch (err) {
                              console.error('Failed to start tagging:', err)
                              setTaggingActive(prev => ({ ...prev, [dirKey]: false }))
                            }
                          }
                        }}
                        title={taggingActive[dirKey] ? "Stop tagging and clear queue" : "Start tagging untagged images"}
                      >
                        {taggingActive[dirKey] ? '⏹' : '▶'} Tag
                      </button>
                      )}
                      {ageDetectorInstalled && (
                      <button
                        className={`toggle-btn ${dir.auto_age_detect ? 'active' : ''}`}
                        onClick={() => {
                          const newValue = !dir.auto_age_detect
                          setDirectories(dirs => dirs.map(d =>
                            makeDirKey(d) === dirKey ? {...d, auto_age_detect: newValue} : d
                          ))
                          updateDirectory(dir.id, { auto_age_detect: newValue }, dir.library_id)
                            .catch(err => {
                              console.error('Failed to update:', err)
                              refreshDirectories()
                            })
                        }}
                        title="Auto-detect ages on new images"
                      >
                        {dir.auto_age_detect ? '☑' : '☐'} Age Detect
                      </button>
                      )}
                      <button
                        className={`toggle-btn ${dir.public_access ? 'active' : ''}`}
                        onClick={() => {
                          const newValue = !dir.public_access
                          setDirectories(dirs => dirs.map(d =>
                            makeDirKey(d) === dirKey ? {...d, public_access: newValue} : d
                          ))
                          updateDirectory(dir.id, { public_access: newValue }, dir.library_id)
                            .catch(err => {
                              console.error('Failed to update:', err)
                              refreshDirectories()
                            })
                        }}
                        title="Allow public network access to this directory"
                      >
                        {dir.public_access ? '☑' : '☐'} Public
                      </button>
                      <button
                        className={`toggle-btn ${dir.show_images ? 'active' : ''}`}
                        onClick={() => {
                          const newValue = !dir.show_images
                          setDirectories(dirs => dirs.map(d =>
                            makeDirKey(d) === dirKey ? {...d, show_images: newValue} : d
                          ))
                          updateDirectory(dir.id, { show_images: newValue }, dir.library_id)
                            .catch(err => {
                              console.error('Failed to update:', err)
                              refreshDirectories()
                            })
                        }}
                        title="Show images from this directory in gallery"
                      >
                        {dir.show_images ? '☑' : '☐'} Images
                      </button>
                      <button
                        className={`toggle-btn ${dir.show_videos ? 'active' : ''}`}
                        onClick={() => {
                          const newValue = !dir.show_videos
                          setDirectories(dirs => dirs.map(d =>
                            makeDirKey(d) === dirKey ? {...d, show_videos: newValue} : d
                          ))
                          updateDirectory(dir.id, { show_videos: newValue }, dir.library_id)
                            .catch(err => {
                              console.error('Failed to update:', err)
                              refreshDirectories()
                            })
                        }}
                        title="Show videos from this directory in gallery"
                      >
                        {dir.show_videos ? '☑' : '☐'} Videos
                      </button>
                      <button
                        className={`toggle-btn ${dir.family_safe ? 'active' : ''}`}
                        onClick={() => {
                          const newValue = !dir.family_safe
                          setDirectories(dirs => dirs.map(d =>
                            makeDirKey(d) === dirKey ? {...d, family_safe: newValue} : d
                          ))
                          updateDirectory(dir.id, { family_safe: newValue }, dir.library_id)
                            .catch(err => {
                              console.error('Failed to update:', err)
                              refreshDirectories()
                            })
                        }}
                        title="Mark as family-safe (shown when family mode is locked)"
                      >
                        {dir.family_safe ? '☑' : '☐'} Family Safe
                      </button>
                      <button
                        className={`toggle-btn ${dir.lan_visible ? 'active' : ''}`}
                        onClick={() => {
                          const newValue = !dir.lan_visible
                          setDirectories(dirs => dirs.map(d =>
                            makeDirKey(d) === dirKey ? {...d, lan_visible: newValue} : d
                          ))
                          updateDirectory(dir.id, { lan_visible: newValue }, dir.library_id)
                            .catch(err => {
                              console.error('Failed to update:', err)
                              refreshDirectories()
                            })
                        }}
                        title="Allow LAN network access to this directory"
                      >
                        {dir.lan_visible ? '☑' : '☐'} LAN
                      </button>
                    </div>

                    {/* Actions row: buttons grouped by purpose */}
                    <div className="directory-actions">
                      <div className="action-group">
                        <button
                          className="action-btn"
                          onClick={() => handleRescan(dir)}
                          disabled={scanning[dirKey]}
                        >
                          {scanning[dirKey] ? 'Scanning...' : 'Rescan'}
                        </button>
                        <button
                          className="action-btn"
                          onClick={() => handleRepair(dir)}
                          disabled={repairing[dirKey]}
                          title="Fix moved files and remove missing entries"
                        >
                          {repairing[dirKey] ? 'Repairing...' : 'Repair'}
                        </button>
                        <button
                          className="action-btn"
                          onClick={() => handlePrune(dir)}
                          disabled={pruning[dirKey] || dir.image_count === 0}
                          title="Move non-favorited images to dumpster"
                        >
                          {pruning[dirKey] ? 'Pruning...' : 'Prune'}
                        </button>
                      </div>
                      <div className="action-group">
                        <button
                          className="action-btn secondary"
                          onClick={() => setComfyuiConfigDir(dir)}
                          title="Configure ComfyUI metadata extraction"
                        >
                          ComfyUI
                        </button>
                        <button
                          className="action-btn secondary"
                          onClick={() => handleRelocate(dir)}
                          disabled={relocating[dirKey]}
                          title="Change directory location (if folder was moved)"
                        >
                          {relocating[dirKey] ? 'Relocating...' : 'Edit Path'}
                        </button>
                      </div>
                      <button
                        className="action-btn danger"
                        onClick={() => handleRemove(dir)}
                      >
                        Remove
                      </button>
                    </div>
                  </li>
                )})}
              </ul>
              </>
            )}

            {/* Batch action bar */}
            {selectedDirs.size > 0 && (
              <div className="batch-action-bar directory-batch-bar">
                <div className="batch-action-count">
                  {selectedDirs.size} selected
                  <button className="batch-select-link" onClick={selectAllDirs}>Select All</button>
                </div>
                <div className="batch-action-buttons">
                  <button
                    className="batch-btn"
                    onClick={handleBatchRescan}
                    disabled={batchLoading}
                  >
                    Rescan All
                  </button>
                  <button
                    className="batch-btn"
                    onClick={handleBatchRepair}
                    disabled={batchLoading}
                  >
                    Repair All
                  </button>
                  <button
                    className="batch-btn"
                    onClick={handleBatchPrune}
                    disabled={batchLoading}
                  >
                    Prune All
                  </button>
                  <button
                    className="batch-btn danger"
                    onClick={handleBatchRemove}
                    disabled={batchLoading}
                  >
                    Remove All
                  </button>
                  <button
                    className="batch-btn secondary"
                    onClick={clearSelection}
                    disabled={batchLoading}
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* ComfyUI Configuration Modal */}
      {comfyuiConfigDir && (
        <ComfyUIConfigModal
          directoryId={comfyuiConfigDir.id}
          directoryName={comfyuiConfigDir.name || comfyuiConfigDir.path}
          libraryId={comfyuiConfigDir.library_id}
          onClose={() => setComfyuiConfigDir(null)}
          onSave={refreshDirectories}
        />
      )}
    </div>
  )
}

export default DirectoriesPage
