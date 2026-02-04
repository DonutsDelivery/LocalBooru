import { useState, useEffect } from 'react'
import {
  getMigrationInfo,
  getMigrationDirectories,
  validateMigration,
  startMigration,
  validateImport,
  startImport,
  getMigrationStatus,
  deleteSourceData,
  cleanupMigration,
  subscribeToMigrationEvents
} from '../api'
import { getDesktopAPI, isDesktopApp } from '../tauriAPI'
import './MigrationSettings.css'

function formatBytes(bytes) {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

export default function MigrationSettings() {
  const [info, setInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [migrating, setMigrating] = useState(false)
  const [progress, setProgress] = useState(null)
  const [result, setResult] = useState(null)
  const [validation, setValidation] = useState(null)
  const [error, setError] = useState(null)

  // Directory selection state
  const [directories, setDirectories] = useState([])
  const [selectedDirIds, setSelectedDirIds] = useState(new Set())
  const [loadingDirs, setLoadingDirs] = useState(false)
  const [dirMode, setDirMode] = useState(null) // Which mode we loaded directories for

  useEffect(() => {
    loadInfo()
  }, [])

  // Subscribe to migration events when migrating
  useEffect(() => {
    if (!migrating) return

    const unsubscribe = subscribeToMigrationEvents((event) => {
      if (event.type === 'migration_progress') {
        setProgress(event.data)
      } else if (event.type === 'migration_completed') {
        setMigrating(false)
        setResult({ success: true, ...event.data })
        loadInfo()
      } else if (event.type === 'migration_error') {
        setMigrating(false)
        setResult({ success: false, error: event.data.error })
      }
    })

    return unsubscribe
  }, [migrating])

  // Also poll for status (fallback if SSE fails)
  useEffect(() => {
    if (!migrating) return

    const interval = setInterval(async () => {
      try {
        const status = await getMigrationStatus()
        if (status.progress) {
          setProgress(status.progress)
        }
        if (!status.running && status.result) {
          setMigrating(false)
          setResult(status.result)
          loadInfo()
        }
      } catch (e) {
        console.error('Failed to get migration status:', e)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [migrating])

  async function loadInfo() {
    try {
      setLoading(true)
      setError(null)
      const data = await getMigrationInfo()
      setInfo(data)
    } catch (e) {
      setError('Failed to load migration info: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  async function loadDirectories(mode) {
    try {
      setLoadingDirs(true)
      setError(null)
      const data = await getMigrationDirectories(mode)
      if (data.success) {
        setDirectories(data.directories)
        setDirMode(mode)
        // Select all by default
        setSelectedDirIds(new Set(data.directories.map(d => d.id)))
      } else {
        setError('Failed to load directories: ' + data.error)
      }
    } catch (e) {
      setError('Failed to load directories: ' + e.message)
    } finally {
      setLoadingDirs(false)
    }
  }

  function toggleDirectory(id) {
    setSelectedDirIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
    // Clear validation when selection changes
    setValidation(null)
  }

  function selectAll() {
    setSelectedDirIds(new Set(directories.map(d => d.id)))
    setValidation(null)
  }

  function deselectAll() {
    setSelectedDirIds(new Set())
    setValidation(null)
  }

  // Calculate selected counts
  const selectedDirs = directories.filter(d => selectedDirIds.has(d.id))
  const selectedImageCount = selectedDirs.reduce((sum, d) => sum + d.image_count, 0)
  const selectedThumbSize = selectedDirs.reduce((sum, d) => sum + d.thumbnail_size, 0)

  async function handleValidate(mode) {
    try {
      setError(null)
      // Load directories if not already loaded for this mode
      if (dirMode !== mode) {
        await loadDirectories(mode)
      }

      const directoryIds = selectedDirIds.size > 0 ? Array.from(selectedDirIds) : null

      // Determine if this should be an import (destination has data) or migration
      const isImport = (mode === 'system_to_portable' && info.portable_has_data) ||
                       (mode === 'portable_to_system' && info.system_has_data)

      let result
      if (isImport && directoryIds && directoryIds.length > 0) {
        // Import: add directories to existing database
        result = await validateImport(mode, directoryIds)
        result.isImport = true
      } else {
        // Migration: requires empty destination
        result = await validateMigration(mode, directoryIds)
        result.isImport = false
      }

      setValidation({ mode, ...result })
    } catch (e) {
      setError('Validation failed: ' + e.message)
    }
  }

  async function handleStartMigration(mode) {
    const dirCount = selectedDirIds.size
    const isSelective = dirCount > 0 && dirCount < directories.length

    // Determine if this should be an import (destination has data) or migration
    const isImport = (mode === 'system_to_portable' && info.portable_has_data) ||
                     (mode === 'portable_to_system' && info.system_has_data)

    const actionWord = isImport ? 'import' : 'migration'
    let confirmMsg = `Start ${actionWord}?\n\nThis will ${isImport ? 'add selected directories to' : 'copy data to'} the ${mode === 'system_to_portable' ? 'portable' : 'system'} location.`
    if (isSelective || isImport) {
      confirmMsg += `\n\n${dirCount} of ${directories.length} directories selected (${selectedImageCount} images)`
    }
    if (isImport && validation?.images_to_skip > 0) {
      confirmMsg += `\n\nNote: ${validation.images_to_skip} duplicate images will be skipped.`
    }

    if (!confirm(confirmMsg)) {
      return
    }

    try {
      setError(null)
      setProgress(null)
      setResult(null)
      setMigrating(true)

      const directoryIds = selectedDirIds.size > 0 ? Array.from(selectedDirIds) : null

      let response
      if (isImport && directoryIds && directoryIds.length > 0) {
        response = await startImport(mode, directoryIds)
        response.isImport = true
      } else {
        response = await startMigration(mode, directoryIds)
        response.isImport = false
      }

      if (!response.success) {
        setMigrating(false)
        setError(response.error)
      }
    } catch (e) {
      setMigrating(false)
      setError('Failed to start: ' + e.message)
    }
  }

  async function handleDeleteSource(mode) {
    if (!confirm('Delete source data?\n\nThis will permanently delete the original data. Only do this after verifying the migration was successful.\n\nThis cannot be undone!')) {
      return
    }

    try {
      setError(null)
      const response = await deleteSourceData(mode)
      if (response.success) {
        alert(response.message)
        loadInfo()
      } else {
        setError(response.error)
      }
    } catch (e) {
      setError('Failed to delete source: ' + e.message)
    }
  }

  async function handleCleanup(mode) {
    if (!confirm('Clean up partial migration?\n\nThis will remove partially copied data from the destination.')) {
      return
    }

    try {
      setError(null)
      const response = await cleanupMigration(mode)
      if (response.success) {
        alert(response.message)
        setResult(null)
        setValidation(null)
        loadInfo()
      } else {
        setError(response.error)
      }
    } catch (e) {
      setError('Failed to cleanup: ' + e.message)
    }
  }

  if (loading) {
    return <div className="migration-settings loading">Loading...</div>
  }

  if (!info) {
    return <div className="migration-settings error">{error || 'Failed to load migration info'}</div>
  }

  const isPortable = info.current_mode === 'portable'
  // Allow migration even if destination has data (for selective/merge migrations)
  const canMigrateToPortable = info.portable_path && info.system_has_data
  const canMigrateToSystem = info.portable_has_data

  return (
    <div className="migration-settings">
      <section>
        <h2>Data Location</h2>
        <p className="setting-description">
          LocalBooru can store data in your system directory (AppData/home folder) or in a portable folder next to the application.
        </p>

        <div className="current-mode">
          <strong>Current Mode:</strong>{' '}
          <span className={`mode-badge ${info.current_mode}`}>
            {info.current_mode === 'portable' ? 'Portable' : 'System'}
          </span>
        </div>

        <div className="paths-info">
          <div className="path-row">
            <label>System Location:</label>
            <code>{info.system_path}</code>
            {info.system_has_data && (
              <span className="data-badge">
                {formatBytes(info.system_data_size)} data
              </span>
            )}
          </div>

          {info.portable_path ? (
            <div className="path-row">
              <label>Portable Location:</label>
              <code>{info.portable_path}</code>
              {info.portable_has_data && (
                <span className="data-badge">
                  {formatBytes(info.portable_data_size)} data
                </span>
              )}
            </div>
          ) : (
            <div className="path-row">
              <label>Portable Location:</label>
              <span className="not-available">Not available (run from portable installation)</span>
            </div>
          )}
        </div>
      </section>

      <section>
        <h2>Migrate Data</h2>

        {error && <div className="error-message">{error}</div>}

        {migrating && (
          <div className="migration-progress">
            <h3>Migration in Progress</h3>
            {progress && (
              <>
                <div className="progress-bar-container">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${progress.percent || 0}%` }}
                  />
                </div>
                <div className="progress-details">
                  <span>{progress.phase}</span>
                  <span>{Math.round(progress.percent || 0)}%</span>
                  <span>{progress.files_copied} / {progress.total_files} files</span>
                </div>
                {progress.current_file && (
                  <div className="current-file">
                    Copying: {progress.current_file}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {result && (
          <div className={`migration-result ${result.success ? 'success' : 'error'}`}>
            {result.success ? (
              <>
                <h3>{result.import === true ? 'Import' : 'Migration'} Complete!</h3>
                {result.import === true ? (
                  <>
                    <p>Imported {result.directories_imported || 0} directories with {result.images_imported || 0} images</p>
                    {(result.images_skipped || 0) > 0 && (
                      <p>{result.images_skipped} duplicate images were skipped</p>
                    )}
                    <p>Tags: {result.tags_created || 0} created, {result.tags_reused || 0} reused</p>
                    <p>Copied {result.files_copied || 0} files ({formatBytes(result.bytes_copied || 0)})</p>
                  </>
                ) : (
                  <p>Copied {result.files_copied || 0} files ({formatBytes(result.bytes_copied || 0)})</p>
                )}
                <p><strong>Important:</strong> Restart LocalBooru to use the new data location.</p>
                <div className="result-actions">
                  {isDesktopApp() && (
                    <button
                      onClick={async () => {
                        if (!confirm('Restart LocalBooru?\n\nThe app will restart and use the new data location.')) return
                        try {
                          const api = getDesktopAPI()
                          if (api?.restartBackend) {
                            await api.restartBackend()
                          }
                          // Reload the page after backend restarts
                          setTimeout(() => window.location.reload(), 2000)
                        } catch (e) {
                          setError('Failed to restart: ' + e.message)
                        }
                      }}
                      className="primary-btn"
                    >
                      Restart Now
                    </button>
                  )}
                  <button
                    onClick={() => handleDeleteSource(validation?.mode || (isPortable ? 'portable_to_system' : 'system_to_portable'))}
                    className="danger-btn"
                  >
                    Delete Source Data
                  </button>
                </div>
              </>
            ) : (
              <>
                <h3>Migration Failed</h3>
                <p className="error">{result.error}</p>
                <button onClick={() => handleCleanup(validation?.mode || (isPortable ? 'portable_to_system' : 'system_to_portable'))}>
                  Clean Up Partial Data
                </button>
              </>
            )}
          </div>
        )}

        {!migrating && !result && (
          <div className="migration-options">
            {/* System to Portable */}
            <div className={`migration-option ${canMigrateToPortable ? '' : 'disabled'}`}>
              <h3>System → Portable</h3>
              <p>Copy all data to the portable location for a self-contained installation.</p>

              {!info.portable_path && (
                <p className="warning">Run LocalBooru from a portable installation to enable this option.</p>
              )}
              {info.portable_has_data && (
                <p className="info">Portable location has existing data. Selected directories will be merged.</p>
              )}
              {!info.system_has_data && (
                <p className="warning">No data in system location to migrate.</p>
              )}

              {canMigrateToPortable && (
                <>
                  <button onClick={() => handleValidate('system_to_portable')}>
                    {loadingDirs && dirMode !== 'system_to_portable' ? 'Loading...' : 'Check Migration'}
                  </button>

                  {/* Directory selection - show after loading directories */}
                  {dirMode === 'system_to_portable' && directories.length > 0 && (
                    <DirectorySelector
                      directories={directories}
                      selectedIds={selectedDirIds}
                      onToggle={toggleDirectory}
                      onSelectAll={selectAll}
                      onDeselectAll={deselectAll}
                      selectedImageCount={selectedImageCount}
                      selectedThumbSize={selectedThumbSize}
                    />
                  )}

                  {validation?.mode === 'system_to_portable' && (
                    <div className="validation-result">
                      {validation.valid ? (
                        <>
                          <p className="success">
                            Ready to {validation.isImport ? 'import' : 'migrate'} {formatBytes(validation.bytes_to_copy)} ({validation.files_to_copy} files)
                            {(validation.selective || validation.isImport) && ` from ${validation.directory_count} directories`}
                          </p>
                          {validation.isImport && (
                            <p className="info">
                              {validation.images_to_import} images to import
                              {validation.images_to_skip > 0 && `, ${validation.images_to_skip} duplicates will be skipped`}
                            </p>
                          )}
                          <button
                            onClick={() => handleStartMigration('system_to_portable')}
                            className="primary-btn"
                            disabled={selectedDirIds.size === 0}
                          >
                            Start {validation.isImport ? 'Import' : 'Migration'}
                          </button>
                        </>
                      ) : (
                        <p className="error">{validation.error}</p>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Portable to System */}
            <div className={`migration-option ${canMigrateToSystem ? '' : 'disabled'}`}>
              <h3>Portable → System</h3>
              <p>Copy all data to the system location for a standard installation.</p>

              {info.system_has_data && (
                <p className="info">System location has existing data. Selected directories will be merged.</p>
              )}
              {!info.portable_has_data && (
                <p className="warning">No data in portable location to migrate.</p>
              )}

              {canMigrateToSystem && (
                <>
                  <button onClick={() => handleValidate('portable_to_system')}>
                    {loadingDirs && dirMode !== 'portable_to_system' ? 'Loading...' : 'Check Migration'}
                  </button>

                  {/* Directory selection - show after loading directories */}
                  {dirMode === 'portable_to_system' && directories.length > 0 && (
                    <DirectorySelector
                      directories={directories}
                      selectedIds={selectedDirIds}
                      onToggle={toggleDirectory}
                      onSelectAll={selectAll}
                      onDeselectAll={deselectAll}
                      selectedImageCount={selectedImageCount}
                      selectedThumbSize={selectedThumbSize}
                    />
                  )}

                  {validation?.mode === 'portable_to_system' && (
                    <div className="validation-result">
                      {validation.valid ? (
                        <>
                          <p className="success">
                            Ready to {validation.isImport ? 'import' : 'migrate'} {formatBytes(validation.bytes_to_copy)} ({validation.files_to_copy} files)
                            {(validation.selective || validation.isImport) && ` from ${validation.directory_count} directories`}
                          </p>
                          {validation.isImport && (
                            <p className="info">
                              {validation.images_to_import} images to import
                              {validation.images_to_skip > 0 && `, ${validation.images_to_skip} duplicates will be skipped`}
                            </p>
                          )}
                          <button
                            onClick={() => handleStartMigration('portable_to_system')}
                            className="primary-btn"
                            disabled={selectedDirIds.size === 0}
                          >
                            Start {validation.isImport ? 'Import' : 'Migration'}
                          </button>
                        </>
                      ) : (
                        <p className="error">{validation.error}</p>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </section>

      <section>
        <h2>Notes</h2>
        <ul className="migration-notes">
          <li>Migration copies your database, thumbnails, settings, and cached data.</li>
          <li>Your original image files are NOT moved - they stay in your watch directories.</li>
          <li>You can select which watch directories to include.</li>
          <li><strong>Import vs Migration:</strong> If the destination already has data, selected directories will be imported (merged) into the existing database. Duplicate images are automatically skipped.</li>
          <li>After migration/import, restart LocalBooru to use the new data location.</li>
          <li>You can delete the source data after verifying the migration was successful.</li>
        </ul>
      </section>
    </div>
  )
}

// Directory selection component
function DirectorySelector({
  directories,
  selectedIds,
  onToggle,
  onSelectAll,
  onDeselectAll,
  selectedImageCount,
  selectedThumbSize
}) {
  return (
    <div className="directory-selector">
      <div className="directory-selector-header">
        <h4>Select Watch Directories to Migrate</h4>
        <div className="directory-selector-actions">
          <button type="button" onClick={onSelectAll} className="small-btn">Select All</button>
          <button type="button" onClick={onDeselectAll} className="small-btn">Deselect All</button>
        </div>
      </div>

      <div className="directory-list">
        {directories.map(dir => (
          <label key={dir.id} className={`directory-item ${!dir.path_accessible ? 'inaccessible' : ''}`}>
            <input
              type="checkbox"
              checked={selectedIds.has(dir.id)}
              onChange={() => onToggle(dir.id)}
            />
            <div className="directory-info">
              <span className="directory-name">{dir.name}</span>
              <span className="directory-path">{dir.path}</span>
              <span className="directory-stats">
                {dir.image_count} images, {formatBytes(dir.thumbnail_size)} thumbnails
              </span>
              {dir.warning && (
                <span className="directory-warning">
                  {dir.warning}
                </span>
              )}
            </div>
          </label>
        ))}
      </div>

      <div className="directory-summary">
        <strong>Selected:</strong> {selectedIds.size} of {directories.length} directories
        ({selectedImageCount} images, {formatBytes(selectedThumbSize)} thumbnails)
      </div>
    </div>
  )
}
