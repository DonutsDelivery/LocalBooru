import { useEffect, useMemo, useRef, useState } from 'react'
import {
  absorbWd14Sidecars,
  exportWd14Sidecars,
  fetchDirectories,
  fetchLibraries,
  importWd14Sidecars,
} from '../api'
import {
  buildWd14DirectoryOptions,
  buildWd14RequestDirectories,
  getWd14Failures,
  getWd14SummaryItems,
  wd14ConfirmationMessage,
  wd14StatusLabel,
} from './wd14SidecarUi'
import './WD14SidecarSettings.css'

const OPERATION_LABELS = {
  import: 'Import',
  absorb: 'Absorb',
  export: 'Export',
}

function errorMessage(error) {
  const detail = error.response?.data?.message
    || error.response?.data?.error
    || error.response?.data?.detail
  if (typeof detail === 'string') return detail
  return error.message || 'The operation failed'
}

export default function WD14SidecarSettings() {
  const [directories, setDirectories] = useState([])
  const [selectedKeys, setSelectedKeys] = useState(new Set())
  const [overwrite, setOverwrite] = useState(false)
  const [loading, setLoading] = useState(true)
  const [busyOperation, setBusyOperation] = useState(null)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const operationLock = useRef(false)

  useEffect(() => {
    let active = true

    Promise.all([fetchLibraries(), fetchDirectories(true)])
      .then(([libraryData, directoryData]) => {
        if (!active) return
        setDirectories(buildWd14DirectoryOptions(
          directoryData.directories || [],
          libraryData.libraries || [],
        ))
      })
      .catch((loadError) => {
        if (active) setError(`Failed to load mounted directories: ${errorMessage(loadError)}`)
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => { active = false }
  }, [])

  const requestDirectories = useMemo(
    () => buildWd14RequestDirectories(directories, selectedKeys),
    [directories, selectedKeys],
  )
  const failures = useMemo(() => getWd14Failures(result?.results), [result])

  function toggleDirectory(key) {
    setSelectedKeys(current => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  function selectAll() {
    setSelectedKeys(new Set(
      directories.filter(directory => directory.accessible).map(directory => directory.key),
    ))
  }

  async function runOperation(operation) {
    if (operationLock.current || busyOperation || requestDirectories.length === 0) return

    const confirmation = wd14ConfirmationMessage(
      operation,
      operation === 'export' && overwrite,
      requestDirectories.length,
    )
    if (confirmation && !confirm(confirmation)) return

    operationLock.current = true
    setBusyOperation(operation)
    setError(null)
    setResult(null)

    try {
      let response
      if (operation === 'import') {
        response = await importWd14Sidecars(requestDirectories)
      } else if (operation === 'absorb') {
        response = await absorbWd14Sidecars(requestDirectories)
      } else {
        response = await exportWd14Sidecars(requestDirectories, overwrite)
      }
      setResult(response)
    } catch (operationError) {
      setError(`${OPERATION_LABELS[operation]} failed: ${errorMessage(operationError)}`)
    } finally {
      operationLock.current = false
      setBusyOperation(null)
    }
  }

  const actionsDisabled = loading || busyOperation !== null || requestDirectories.length === 0

  return (
    <section className="wd14-sidecar-settings">
      <header>
        <h2>WD14 Text Sidecars</h2>
        <p className="settings-description">
          Exchange comma-separated tags with same-stem .txt files beside indexed media.
          Operations are limited to mounted, registered LocalBooru directories.
        </p>
      </header>

      {error && <div className="wd14-message error" role="alert">{error}</div>}

      <div className="wd14-directory-section">
        <div className="wd14-section-heading">
          <div>
            <h3>Directories</h3>
            <span>{selectedKeys.size} of {directories.length} selected</span>
          </div>
          <div className="wd14-selection-actions">
            <button type="button" onClick={selectAll} disabled={loading || busyOperation !== null}>
              Select all
            </button>
            <button
              type="button"
              onClick={() => setSelectedKeys(new Set())}
              disabled={loading || busyOperation !== null || selectedKeys.size === 0}
            >
              Clear
            </button>
          </div>
        </div>

        {loading ? (
          <p className="settings-description">Loading mounted directories...</p>
        ) : directories.length === 0 ? (
          <p className="settings-description">No mounted watch directories are available.</p>
        ) : (
          <div className="wd14-directory-list">
            {directories.map(directory => (
              <label
                key={directory.key}
                className={`wd14-directory ${directory.accessible ? '' : 'inaccessible'}`}
              >
                <input
                  type="checkbox"
                  checked={selectedKeys.has(directory.key)}
                  disabled={!directory.accessible || busyOperation !== null}
                  onChange={() => toggleDirectory(directory.key)}
                />
                <span className="wd14-directory-details">
                  <strong>{directory.name}</strong>
                  <span>{directory.libraryName} · Directory {directory.directoryId}</span>
                  <code>{directory.path}</code>
                  <span>{directory.imageCount} indexed media{!directory.accessible && ' · Path unavailable'}</span>
                </span>
              </label>
            ))}
          </div>
        )}
      </div>

      <div className="wd14-operation-section">
        <div className="wd14-export-option">
          <label>
            <input
              type="checkbox"
              checked={overwrite}
              disabled={busyOperation !== null}
              onChange={event => setOverwrite(event.target.checked)}
            />
            Overwrite existing sidecars during Export
          </label>
          <span>Off by default. Existing .txt files are otherwise skipped.</span>
        </div>

        <div className="wd14-operation-actions">
          {Object.entries(OPERATION_LABELS).map(([operation, label]) => (
            <button
              key={operation}
              type="button"
              className={operation === 'absorb' ? 'danger' : 'primary'}
              disabled={actionsDisabled}
              onClick={() => runOperation(operation)}
            >
              {busyOperation === operation ? `${label} running...` : label}
            </button>
          ))}
        </div>
        <p className="wd14-operation-note">
          Import adds tags. Absorb adds tags and then removes only fully committed sidecars.
          Export writes the current LocalBooru tags.
        </p>
      </div>

      {result && (
        <div className="wd14-results" aria-live="polite">
          <h3>{OPERATION_LABELS[result.operation] || 'Operation'} results</h3>
          <div className="wd14-summary-grid">
            {getWd14SummaryItems(result.summary).map(([label, value]) => (
              <div key={label} className="wd14-summary-item">
                <span>{label}</span>
                <strong>{value}</strong>
              </div>
            ))}
          </div>

          {failures.length > 0 && (
            <div className="wd14-failures">
              <h4>Sidecars requiring attention</h4>
              <ul>
                {failures.map((failure, index) => (
                  <li key={`${failure.sidecar_path}-${index}`}>
                    <div>
                      <strong>{wd14StatusLabel(failure.status)}</strong>
                      <code>{failure.sidecar_path}</code>
                    </div>
                    <p>{failure.error || 'The sidecar could not be completed safely.'}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
