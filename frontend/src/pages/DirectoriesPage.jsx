/**
 * DirectoriesPage - Directory management page
 * Extracted from App.jsx
 */
import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Sidebar'
import ComfyUIConfigModal from '../components/ComfyUIConfigModal'
import { getLibraryStats, updateDirectory, tagUntagged, clearDirectoryTagQueue } from '../api'
import { getDesktopAPI, isDesktopApp } from '../tauriAPI'

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

  const refreshDirectories = async () => {
    const { fetchDirectories } = await import('../api')
    const data = await fetchDirectories()
    const dirs = data.directories || []
    setDirectories(dirs)
    // Sync tagging button state with actual queue
    const activeState = {}
    for (const dir of dirs) {
      if (dir.pending_tag_tasks > 0) {
        activeState[dir.id] = true
      }
    }
    setTaggingActive(activeState)
  }

  useEffect(() => {
    refreshDirectories()
      .catch(console.error)
      .finally(() => setLoading(false))
    getLibraryStats().then(setStats).catch(console.error)
  }, [])

  const handleAddDirectory = async () => {
    const api = getDesktopAPI()
    if (api?.addDirectory) {
      const path = await api.addDirectory()
      if (path) {
        const { addDirectory } = await import('../api')
        await addDirectory(path)
        await refreshDirectories()
      }
    } else {
      alert('Directory picker only available in desktop app')
    }
  }

  const handleAddParentDirectory = async () => {
    const api = getDesktopAPI()
    if (api?.addDirectory) {
      const path = await api.addDirectory()
      if (path) {
        const { addParentDirectory } = await import('../api')
        const result = await addParentDirectory(path)
        alert(result.message)
        await refreshDirectories()
      }
    } else {
      alert('Directory picker only available in desktop app')
    }
  }

  const handleRescan = async (dirId) => {
    setScanning(prev => ({ ...prev, [dirId]: true }))
    try {
      const { scanDirectory } = await import('../api')
      await scanDirectory(dirId)
      await refreshDirectories()
    } catch (error) {
      console.error('Scan failed:', error)
      alert('Scan failed: ' + error.message)
    } finally {
      setScanning(prev => ({ ...prev, [dirId]: false }))
    }
  }

  const handleRemove = async (dirId, dirName) => {
    if (!confirm(`Remove "${dirName}" from watch list?\n\nImages will be removed from library.\nActual files on disk will NOT be deleted.`)) {
      return
    }
    try {
      const { removeDirectory } = await import('../api')
      await removeDirectory(dirId, false)
      await refreshDirectories()
    } catch (error) {
      console.error('Remove failed:', error)
      alert('Remove failed: ' + error.message)
    }
  }

  const handlePrune = async (dirId, dirName, favoritedCount) => {
    const nonFavorited = directories.find(d => d.id === dirId)?.image_count - favoritedCount
    const savedDumpsterPath = localStorage.getItem('localbooru_dumpster_path') || null
    const dumpsterInfo = savedDumpsterPath ? `\nDumpster: ${savedDumpsterPath}` : ''
    if (!confirm(`Prune "${dirName}"?\n\nThis will move ${nonFavorited} non-favorited images to the dumpster folder.\nFavorited images (${favoritedCount}) will be kept.${dumpsterInfo}`)) {
      return
    }
    setPruning(prev => ({ ...prev, [dirId]: true }))
    try {
      const { pruneDirectory } = await import('../api')
      const result = await pruneDirectory(dirId, savedDumpsterPath)
      alert(`Pruned ${result.pruned} images to:\n${result.dumpster_path}`)
      await refreshDirectories()
      getLibraryStats().then(setStats).catch(console.error)
    } catch (error) {
      console.error('Prune failed:', error)
      alert('Prune failed: ' + error.message)
    } finally {
      setPruning(prev => ({ ...prev, [dirId]: false }))
    }
  }

  const handleRelocate = async (dirId, dirName, currentPath) => {
    const api = getDesktopAPI()
    if (api?.addDirectory) {
      const newPath = await api.addDirectory()
      if (newPath && newPath !== currentPath) {
        if (!confirm(`Update directory location?\n\nFrom: ${currentPath}\nTo: ${newPath}\n\nThis will update all file references.`)) {
          return
        }
        setRelocating(prev => ({ ...prev, [dirId]: true }))
        try {
          const { updateDirectoryPath } = await import('../api')
          const result = await updateDirectoryPath(dirId, newPath)
          alert(`Directory relocated.\n${result.files_updated} file references updated.`)
          await refreshDirectories()
        } catch (error) {
          console.error('Relocate failed:', error)
          alert('Relocate failed: ' + (error.response?.data?.detail || error.message))
        } finally {
          setRelocating(prev => ({ ...prev, [dirId]: false }))
        }
      }
    } else {
      alert('Directory picker only available in desktop app')
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
    setSelectedDirs(new Set(directories.map(d => d.id)))
  }

  const clearSelection = () => {
    setSelectedDirs(new Set())
  }

  // Batch action handlers
  const handleBatchRescan = async () => {
    if (selectedDirs.size === 0) return
    setBatchLoading(true)
    const { scanDirectory } = await import('../api')
    const dirIds = Array.from(selectedDirs)

    // Mark all as scanning
    setScanning(prev => {
      const next = { ...prev }
      dirIds.forEach(id => next[id] = true)
      return next
    })

    try {
      // Run rescans in parallel
      await Promise.all(dirIds.map(id => scanDirectory(id).catch(e => {
        console.error(`Scan failed for ${id}:`, e)
      })))
      await refreshDirectories()
    } finally {
      setScanning({})
      setBatchLoading(false)
      clearSelection()
    }
  }

  const handleBatchPrune = async () => {
    if (selectedDirs.size === 0) return
    const selectedList = directories.filter(d => selectedDirs.has(d.id))
    const totalNonFavorited = selectedList.reduce((sum, d) => sum + (d.image_count - d.favorited_count), 0)
    const totalFavorited = selectedList.reduce((sum, d) => sum + d.favorited_count, 0)
    const savedDumpsterPath = localStorage.getItem('localbooru_dumpster_path') || null
    const dumpsterInfo = savedDumpsterPath ? `\nDumpster: ${savedDumpsterPath}` : ''

    if (!confirm(`Prune ${selectedDirs.size} directories?\n\nThis will move ${totalNonFavorited} non-favorited images to the dumpster folder.\nFavorited images (${totalFavorited}) will be kept.${dumpsterInfo}`)) {
      return
    }

    setBatchLoading(true)
    const { pruneDirectory } = await import('../api')
    const dirIds = Array.from(selectedDirs)

    // Mark all as pruning
    setPruning(prev => {
      const next = { ...prev }
      dirIds.forEach(id => next[id] = true)
      return next
    })

    try {
      let totalPruned = 0
      for (const id of dirIds) {
        try {
          const result = await pruneDirectory(id, savedDumpsterPath)
          totalPruned += result.pruned
        } catch (e) {
          console.error(`Prune failed for ${id}:`, e)
        }
      }
      alert(`Pruned ${totalPruned} images total`)
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
    const selectedList = directories.filter(d => selectedDirs.has(d.id))
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
      const dirIds = Array.from(selectedDirs)
      console.log(`[Bulk Remove] Deleting ${dirIds.length} directories with ${totalImages} images...`)

      const result = await bulkDeleteDirectories(dirIds, false)
      console.log(`[Bulk Remove] Deleted ${result.deleted} directories, ${result.image_count} images`)

      await refreshDirectories()
    } catch (e) {
      console.error('Bulk remove failed:', e)
      alert(`Remove failed: ${e.response?.data?.detail || e.message || 'Unknown error'}`)
    } finally {
      setBatchLoading(false)
      clearSelection()
    }
  }

  const handleRepair = async (dirId) => {
    setRepairing(prev => ({ ...prev, [dirId]: true }))
    try {
      const { repairDirectoryPaths } = await import('../api')
      const result = await repairDirectoryPaths(dirId)
      alert(`Repair complete:\n${result.valid} files OK\n${result.repaired} paths fixed\n${result.removed} missing removed`)
      await refreshDirectories()
    } catch (error) {
      console.error('Repair failed:', error)
      alert('Repair failed: ' + (error.response?.data?.detail || error.message))
    } finally {
      setRepairing(prev => ({ ...prev, [dirId]: false }))
    }
  }

  const handleBatchRepair = async () => {
    if (selectedDirs.size === 0) return
    setBatchLoading(true)
    try {
      const { bulkRepairDirectories } = await import('../api')
      const result = await bulkRepairDirectories(Array.from(selectedDirs))
      alert(`Batch repair complete:\n${result.totals.valid} files OK\n${result.totals.repaired} paths fixed\n${result.totals.removed} missing removed`)
      await refreshDirectories()
    } catch (e) {
      console.error('Batch repair failed:', e)
      alert(`Batch repair failed: ${e.response?.data?.detail || e.message || 'Unknown error'}`)
    } finally {
      setBatchLoading(false)
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

            {loading ? (
              <p>Loading...</p>
            ) : directories.length === 0 ? (
              <p className="empty-state">No directories added yet. Add a folder to get started!</p>
            ) : (
              <>
              <div className="directory-list-header">
                <span className="directory-count">{directories.length} directories</span>
                <div className="selection-buttons">
                  <button className="select-btn" onClick={selectAllDirs}>Select All</button>
                  <button className="select-btn" onClick={clearSelection} disabled={selectedDirs.size === 0}>Unselect All</button>
                </div>
              </div>
              <ul className="directory-list">
                {directories.map(dir => (
                  <li key={dir.id} className={`directory-item ${dir.enabled ? '' : 'disabled'} ${selectedDirs.has(dir.id) ? 'selected' : ''}`}>
                    {/* Header row: checkbox, name/path/count, status */}
                    <div className="directory-header">
                      <label className="directory-checkbox">
                        <input
                          type="checkbox"
                          checked={selectedDirs.has(dir.id)}
                          onChange={() => toggleSelectDir(dir.id)}
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
                      <button
                        className={`toggle-btn tag-btn ${taggingActive[dir.id] ? 'active' : ''}`}
                        onClick={async () => {
                          const isActive = taggingActive[dir.id]
                          if (isActive) {
                            setTaggingActive(prev => ({ ...prev, [dir.id]: false }))
                            try {
                              await clearDirectoryTagQueue(dir.id)
                            } catch (err) {
                              console.error('Failed to clear queue:', err)
                            }
                          } else {
                            setTaggingActive(prev => ({ ...prev, [dir.id]: true }))
                            try {
                              await tagUntagged(dir.id)
                            } catch (err) {
                              console.error('Failed to start tagging:', err)
                              setTaggingActive(prev => ({ ...prev, [dir.id]: false }))
                            }
                          }
                        }}
                        title={taggingActive[dir.id] ? "Stop tagging and clear queue" : "Start tagging untagged images"}
                      >
                        {taggingActive[dir.id] ? '⏹' : '▶'} Tag
                      </button>
                      <button
                        className={`toggle-btn ${dir.auto_age_detect ? 'active' : ''}`}
                        onClick={() => {
                          const newValue = !dir.auto_age_detect
                          setDirectories(dirs => dirs.map(d =>
                            d.id === dir.id ? {...d, auto_age_detect: newValue} : d
                          ))
                          updateDirectory(dir.id, { auto_age_detect: newValue })
                            .catch(err => {
                              console.error('Failed to update:', err)
                              refreshDirectories()
                            })
                        }}
                        title="Auto-detect ages on new images"
                      >
                        {dir.auto_age_detect ? '☑' : '☐'} Age Detect
                      </button>
                      <button
                        className={`toggle-btn ${dir.public_access ? 'active' : ''}`}
                        onClick={() => {
                          const newValue = !dir.public_access
                          setDirectories(dirs => dirs.map(d =>
                            d.id === dir.id ? {...d, public_access: newValue} : d
                          ))
                          updateDirectory(dir.id, { public_access: newValue })
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
                            d.id === dir.id ? {...d, show_images: newValue} : d
                          ))
                          updateDirectory(dir.id, { show_images: newValue })
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
                            d.id === dir.id ? {...d, show_videos: newValue} : d
                          ))
                          updateDirectory(dir.id, { show_videos: newValue })
                            .catch(err => {
                              console.error('Failed to update:', err)
                              refreshDirectories()
                            })
                        }}
                        title="Show videos from this directory in gallery"
                      >
                        {dir.show_videos ? '☑' : '☐'} Videos
                      </button>
                    </div>

                    {/* Actions row: buttons grouped by purpose */}
                    <div className="directory-actions">
                      <div className="action-group">
                        <button
                          className="action-btn"
                          onClick={() => handleRescan(dir.id)}
                          disabled={scanning[dir.id]}
                        >
                          {scanning[dir.id] ? 'Scanning...' : 'Rescan'}
                        </button>
                        <button
                          className="action-btn"
                          onClick={() => handleRepair(dir.id)}
                          disabled={repairing[dir.id]}
                          title="Fix moved files and remove missing entries"
                        >
                          {repairing[dir.id] ? 'Repairing...' : 'Repair'}
                        </button>
                        <button
                          className="action-btn"
                          onClick={() => handlePrune(dir.id, dir.name || dir.path, dir.favorited_count)}
                          disabled={pruning[dir.id] || dir.image_count === 0}
                          title="Move non-favorited images to dumpster"
                        >
                          {pruning[dir.id] ? 'Pruning...' : 'Prune'}
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
                          onClick={() => handleRelocate(dir.id, dir.name || dir.path, dir.path)}
                          disabled={relocating[dir.id]}
                          title="Change directory location (if folder was moved)"
                        >
                          {relocating[dir.id] ? 'Relocating...' : 'Edit Path'}
                        </button>
                      </div>
                      <button
                        className="action-btn danger"
                        onClick={() => handleRemove(dir.id, dir.name || dir.path)}
                      >
                        Remove
                      </button>
                    </div>
                  </li>
                ))}
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
          onClose={() => setComfyuiConfigDir(null)}
          onSave={refreshDirectories}
        />
      )}
    </div>
  )
}

export default DirectoriesPage
