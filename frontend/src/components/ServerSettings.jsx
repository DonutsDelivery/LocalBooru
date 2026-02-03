import { useState, useEffect } from 'react'
import {
  getServers,
  addServer,
  updateServer,
  removeServer,
  getActiveServerId,
  setActiveServerId,
  testServerConnection,
  isMobileApp
} from '../serverManager'
import { updateServerConfig } from '../api'
import './ServerSettings.css'

// Dynamic import for barcode scanner (only on mobile)
let BarcodeScanner = null
if (isMobileApp()) {
  import('@capacitor-mlkit/barcode-scanning').then(module => {
    BarcodeScanner = module.BarcodeScanner
  })
}

export default function ServerSettings({ onServerChange }) {
  const [servers, setServers] = useState([])
  const [activeServerId, setActiveServerIdState] = useState(null)
  const [showAddModal, setShowAddModal] = useState(false)
  const [editingServer, setEditingServer] = useState(null)
  const [loading, setLoading] = useState(true)
  const [scanning, setScanning] = useState(false)
  const [scanError, setScanError] = useState(null)

  // Load servers on mount
  useEffect(() => {
    loadServers()
  }, [])

  async function loadServers() {
    setLoading(true)
    const [serverList, activeId] = await Promise.all([
      getServers(),
      getActiveServerId()
    ])
    setServers(serverList)
    setActiveServerIdState(activeId)
    setLoading(false)
  }

  async function handleSetActive(id) {
    await setActiveServerId(id)
    setActiveServerIdState(id)
    await updateServerConfig()
    onServerChange?.()
  }

  async function handleDelete(id) {
    if (!confirm('Remove this server?')) return
    await removeServer(id)
    await loadServers()
    await updateServerConfig()
    onServerChange?.()
  }

  async function handleSaveServer(serverData) {
    if (editingServer) {
      await updateServer(editingServer.id, serverData)
    } else {
      await addServer(serverData)
    }
    await loadServers()
    await updateServerConfig()
    setShowAddModal(false)
    setEditingServer(null)
    onServerChange?.()
  }

  function handleEdit(server) {
    setEditingServer(server)
    setShowAddModal(true)
  }

  async function handleScanQR() {
    if (!BarcodeScanner) {
      setScanError('QR scanner not available')
      return
    }

    try {
      setScanError(null)
      setScanning(true)

      // Check camera permission
      const { camera } = await BarcodeScanner.checkPermissions()
      if (camera !== 'granted') {
        const { camera: newPerm } = await BarcodeScanner.requestPermissions()
        if (newPerm !== 'granted') {
          setScanError('Camera permission required to scan QR codes')
          setScanning(false)
          return
        }
      }

      // Start scanning
      document.body.classList.add('barcode-scanner-active')
      const result = await BarcodeScanner.scan()
      document.body.classList.remove('barcode-scanner-active')

      if (!result.barcodes.length) {
        setScanError('No QR code found')
        setScanning(false)
        return
      }

      // Parse QR data
      const rawValue = result.barcodes[0].rawValue
      let qrData
      try {
        qrData = JSON.parse(rawValue)
      } catch {
        setScanError('Invalid QR code format')
        setScanning(false)
        return
      }

      // Validate QR data
      if (qrData.type !== 'localbooru') {
        setScanError('Not a LocalBooru QR code')
        setScanning(false)
        return
      }

      // Try connecting - local first, then public
      let workingUrl = null
      let urls = []
      if (qrData.local) urls.push(qrData.local)
      if (qrData.public) urls.push(qrData.public)

      for (const url of urls) {
        const result = await testServerConnection(url)
        if (result.success) {
          workingUrl = url
          break
        }
      }

      if (!workingUrl) {
        setScanError('Could not connect to server. Make sure you are on the same network.')
        setScanning(false)
        return
      }

      // Add the server with certificate fingerprint (for HTTPS pinning)
      await addServer({
        name: qrData.name || 'LocalBooru Server',
        url: workingUrl,
        username: null,
        password: null,
        certFingerprint: qrData.cert_fingerprint || null,  // Store cert fingerprint for pinning
        lastConnected: new Date().toISOString()
      })

      await loadServers()
      await updateServerConfig()
      onServerChange?.()

    } catch (err) {
      console.error('Scan error:', err)
      setScanError(err.message || 'Failed to scan QR code')
      document.body.classList.remove('barcode-scanner-active')
    } finally {
      setScanning(false)
    }
  }

  // Don't show on web (non-Capacitor)
  if (!isMobileApp()) {
    return (
      <div className="server-settings">
        <div className="server-info">
          Running as web app - connects to the server that hosts it.
        </div>
      </div>
    )
  }

  if (loading) {
    return <div className="server-settings loading">Loading servers...</div>
  }

  return (
    <div className="server-settings">
      <div className="server-header">
        <h3>Servers</h3>
        <div className="server-header-actions">
          <button
            className="scan-qr-btn"
            onClick={handleScanQR}
            disabled={scanning}
          >
            {scanning ? 'Scanning...' : 'Scan QR'}
          </button>
          <button className="add-server-btn" onClick={() => setShowAddModal(true)}>
            + Add
          </button>
        </div>
      </div>

      {scanError && (
        <div className="scan-error">
          {scanError}
          <button className="dismiss-btn" onClick={() => setScanError(null)}>Dismiss</button>
        </div>
      )}

      {servers.length === 0 ? (
        <div className="no-servers">
          <p>No servers configured.</p>
          <p>Add a server to connect to your LocalBooru library.</p>
        </div>
      ) : (
        <div className="server-list">
          {servers.map(server => (
            <ServerCard
              key={server.id}
              server={server}
              isActive={server.id === activeServerId}
              onSetActive={() => handleSetActive(server.id)}
              onEdit={() => handleEdit(server)}
              onDelete={() => handleDelete(server.id)}
            />
          ))}
        </div>
      )}

      {showAddModal && (
        <AddServerModal
          server={editingServer}
          onSave={handleSaveServer}
          onClose={() => {
            setShowAddModal(false)
            setEditingServer(null)
          }}
        />
      )}
    </div>
  )
}

function ServerCard({ server, isActive, onSetActive, onEdit, onDelete }) {
  const [status, setStatus] = useState(null)
  const [testing, setTesting] = useState(false)

  async function testConnection() {
    setTesting(true)
    const result = await testServerConnection(server.url, server.username, server.password)
    setStatus(result.success ? 'connected' : 'error')
    setTesting(false)
  }

  useEffect(() => {
    testConnection()
  }, [server.url])

  return (
    <div className={`server-card ${isActive ? 'active' : ''}`} onClick={onSetActive}>
      <div className="server-status">
        {testing ? (
          <span className="status-dot testing" title="Testing..."></span>
        ) : status === 'connected' ? (
          <span className="status-dot connected" title="Connected"></span>
        ) : (
          <span className="status-dot error" title="Connection failed"></span>
        )}
      </div>

      <div className="server-info">
        <div className="server-name">{server.name}</div>
        <div className="server-url">{server.url}</div>
        {server.lastConnected && (
          <div className="server-last-connected">
            Last connected: {new Date(server.lastConnected).toLocaleDateString()}
          </div>
        )}
      </div>

      <div className="server-actions">
        {isActive && <span className="active-badge">Active</span>}
        <button className="edit-btn" onClick={(e) => { e.stopPropagation(); onEdit() }}>
          Edit
        </button>
        <button className="delete-btn" onClick={(e) => { e.stopPropagation(); onDelete() }}>
          Delete
        </button>
      </div>
    </div>
  )
}

function AddServerModal({ server, onSave, onClose }) {
  const [name, setName] = useState(server?.name || '')
  const [url, setUrl] = useState(server?.url || '')
  const [username, setUsername] = useState(server?.username || '')
  const [password, setPassword] = useState(server?.password || '')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)

  async function handleTest() {
    if (!url) return

    setTesting(true)
    setTestResult(null)

    // Normalize URL
    let normalizedUrl = url.trim()
    if (!normalizedUrl.startsWith('http://') && !normalizedUrl.startsWith('https://')) {
      normalizedUrl = 'http://' + normalizedUrl
    }
    // Remove trailing slash
    normalizedUrl = normalizedUrl.replace(/\/$/, '')

    setUrl(normalizedUrl)

    const result = await testServerConnection(normalizedUrl, username, password)
    setTestResult(result)
    setTesting(false)
  }

  function handleSave() {
    if (!url || !testResult?.success) return

    // Extract hostname for default name
    let defaultName = name
    if (!defaultName) {
      try {
        const urlObj = new URL(url)
        defaultName = urlObj.hostname
      } catch {
        defaultName = 'LocalBooru Server'
      }
    }

    onSave({
      name: defaultName,
      url: url.replace(/\/$/, ''),
      username: username || null,
      password: password || null,
      lastConnected: new Date().toISOString()
    })
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <h3>{server ? 'Edit Server' : 'Add Server'}</h3>

        <div className="form-group">
          <label>Server URL</label>
          <input
            type="text"
            placeholder="192.168.1.100:8790"
            value={url}
            onChange={e => setUrl(e.target.value)}
          />
          <small>IP address or hostname with port</small>
        </div>

        <div className="form-group">
          <label>Name (optional)</label>
          <input
            type="text"
            placeholder="My Server"
            value={name}
            onChange={e => setName(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Username (optional)</label>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Password (optional)</label>
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
          />
        </div>

        {testResult && (
          <div className={`test-result ${testResult.success ? 'success' : 'error'}`}>
            {testResult.success ? 'Connection successful!' : `Error: ${testResult.error}`}
          </div>
        )}

        <div className="modal-actions">
          <button className="cancel-btn" onClick={onClose}>Cancel</button>
          <button
            className="test-btn"
            onClick={handleTest}
            disabled={testing || !url}
          >
            {testing ? 'Testing...' : 'Test Connection'}
          </button>
          <button
            className="save-btn"
            onClick={handleSave}
            disabled={!testResult?.success}
          >
            Save
          </button>
        </div>
      </div>
    </div>
  )
}
